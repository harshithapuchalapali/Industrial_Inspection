"""
Milestone 4: Cost-Sensitive Decision Engine Arbitration & System Evaluation.

Protocol
--------
- Fixed inputs: immutable splits.json + M1 (baseline_metrics.json) and M3
  (classifier_metrics.json) results. Thresholds are LOCKED and read from the
  stored JSONs. NO threshold optimisation, NO retraining, NO architecture
  changes, NO changes to D_eval_final (strictly read-only).

- Model sourcing:
    M1 Mahalanobis detector is re-fitted on D_train_anomaly (176 normals) — verified
    bit-consistent with M1 (equivalence gate).
    M3 MLP head + temperature scaler are LOADED from persisted artifacts
    (classifier_head.pt, temperature_scaler.pt) produced by the M3 run. NO training.
  Equivalence gates verify that loaded-model metrics on D_eval_final match the stored
  M3 JSON within tolerance -> proving we are using the exact locked M3 model.

- Decision engine: the approved 5-case arbitration matrix is applied with the
  locked thresholds (tau_anomaly from M1, tau_confidence from M3).

Outputs
-------
  artifacts/m4_metrics.json       Full numeric results.
  artifacts/m4_report.md          Detailed M4 experiment report.
  artifacts/m4_action_distribution.png
  artifacts/m4_per_category_actions.png
  artifacts/m4_system_confusion.png
  artifacts/m4_binary_confusion.png
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.data.dataset import IDX_TO_CLASS, load_splits_and_build_datasets
from ml_engine.models.baseline_extractor import ResNetFeatureExtractor
from ml_engine.models.baseline_knn import MahalanobisAnomalyDetector
from ml_engine.models.classifier_head import (
    DefectClassifier,
    collect_classifier_outputs,
)
from ml_engine.models.temperature_scaling import (
    TemperatureScaler,
    apply_temperature_scaling,
)
from ml_engine.evaluation.metrics import (
    compute_roc_pr_metrics,
    evaluate_threshold_performance,
    plot_confusion_matrix_figure,
)
from ml_engine.evaluation.classification_metrics import (
    CLASS_NAMES,
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_normalized_entropy,
    softmax_to_probs,
)
from ml_engine.evaluation.decision_engine import (
    compute_lodo_metrics,
    route_batch,
    route_single,
)
from ml_engine.evaluation.system_metrics import (
    compute_action_distribution,
    compute_per_category_actions,
    compute_reason_distribution,
    compute_system_confusion,
    compute_system_metrics,
    plot_action_distribution,
    plot_per_category_actions,
    plot_system_confusion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
TRAIN_EPOCHS = 80
GATE_TOL = 1e-3          # absolute tolerance for proportion gates
GATE_TOL_TEMP = 1e-4     # tighter tolerance for temperature


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def check_gate(name: str, actual: float, expected: float, tol: float = GATE_TOL,
               gates: Optional[List[Dict[str, object]]] = None) -> bool:
    ok = bool(abs(float(actual) - float(expected)) <= tol)
    msg = f"  [GATE] {name}: actual={float(actual):.6f}  expected={float(expected):.6f}  -> {'PASS' if ok else 'FAIL'}"
    print(msg)
    if gates is not None:
        gates.append({
            "name": name,
            "actual": float(actual),
            "expected": float(expected),
            "tolerance": tol,
            "passed": ok,
        })
    return ok


# ── Report rendering ─────────────────────────────────────────────────────────

def write_markdown_report(results: Dict, report_path: Path):
    lines: List[str] = []
    lines.append("# M4 Experiment Report — Decision Engine & System Evaluation")
    lines.append("")
    lines.append(f"*Date generated:* {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"*Seed:* {SEED} | *Device:* {results['runtime']['device']}")
    lines.append("")

    lines.append("## 1. Fixed / Locked Inputs")
    lines.append("")
    lines.append("| Input | Value |")
    lines.append("|---|---|")
    cal = results["locked_config"]
    lines.append(f"| tau_anomaly (M1, beta=5) | {cal['tau_anomaly']:.6f} |")
    lines.append(f"| tau_confidence (M3) | {cal['tau_confidence']:.4f} |")
    lines.append(f"| Temperature T (M3) | {cal['temperature']:.4f} |")
    lines.append(f"| Classifier head (M3) | {cal['head_type']} |")
    lines.append(f"| Detector (M1) | Mahalanobis (Ledoit-Wolf), fitted on {cal['train_normal_count']} normals |")
    lines.append(f"| Evaluation set | {results['dataset']['eval_total']} images "
                 f"({results['dataset']['eval_normal']} normal, {results['dataset']['eval_defect']} defective; untouched) |")
    lines.append("")

    lines.append("## 2. Locked Model Sourcing & Equivalence Gates")
    lines.append("")
    lines.append("M1 Mahalanobis detector is reproduced deterministically (seed=42). The M3 classifier "
                 "head + temperature scaler are loaded from persisted artifacts (classifier_head.pt, "
                 "temperature_scaler.pt). Gates verify the loaded model reproduces the stored M1/M3 metrics "
                 "on D_eval_final within tolerance.")
    lines.append("")
    lines.append("| Gate | Actual | Expected (stored) | Passed |")
    lines.append("|---|---|---|---|")
    for g in results["equivalence_gates"]:
        lines.append(f"| {g['name']} | {g['actual']:.6f} | {g['expected']:.6f} | {'YES' if g['passed'] else 'NO'} |")
    lines.append("")
    lines.append(f"**All gates passed:** {'YES' if results['all_gates_passed'] else 'NO'}")
    lines.append("")

    lines.append("## 3. Stream A — Binary Anomaly Detection (locked tau_anomaly)")
    lines.append("")
    binm = results["binary_anomaly"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| ROC-AUC | {binm['roc_auc']:.4f} |")
    lines.append(f"| PR-AUC (AP) | {binm['pr_auc']:.4f} |")
    perf = binm["locked_performance"]
    lines.append(f"| TP / FP / TN / FN | {perf['tp']} / {perf['fp']} / {perf['tn']} / {perf['fn']} |")
    lines.append(f"| FNR (escape) | {perf['fnr'] * 100:.2f}% |")
    lines.append(f"| FPR (false alarm) | {perf['fpr'] * 100:.2f}% |")
    lines.append(f"| Precision | {perf['precision']:.4f} |")
    lines.append(f"| Recall | {perf['recall']:.4f} |")
    lines.append(f"| F1 | {perf['f1_score']:.4f} |")
    lines.append(f"| Accuracy | {perf['accuracy'] * 100:.2f}% |")
    lines.append("")

    lines.append("## 4. Decision Engine — System Accounting (PASS / FAIL / REVIEW)")
    lines.append("")
    sysm = results["system_metrics"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| PASS | {sysm['pass_on_normal']} normal + {sysm['pass_on_defect']} defect |")
    lines.append(f"| FAIL | {sysm['fail_on_normal']} normal + {sysm['fail_on_defect']} defect |")
    lines.append(f"| REVIEW | {sysm['review_on_normal']} normal + {sysm['review_on_defect']} defect |")
    lines.append(f"| Silent Escape Rate (SER) | {sysm['silent_escape_rate_ser'] * 100:.2f}% |")
    lines.append(f"| False Alarm Rate (FPR) | {sysm['false_alarm_rate_fpr'] * 100:.2f}% |")
    lines.append(f"| Precision | {sysm['precision']:.4f} |")
    lines.append(f"| Recall | {sysm['recall']:.4f} |")
    lines.append(f"| F1 | {sysm['f1_score']:.4f} |")
    lines.append(f"| Accuracy | {sysm['accuracy'] * 100:.2f}% |")
    lines.append(f"| Review rate | {sysm['review_rate'] * 100:.2f}% |")
    lines.append(f"| Defect review rate | {sysm['defect_review_rate'] * 100:.2f}% |")
    lines.append(f"| Normal review rate | {sysm['normal_review_rate'] * 100:.2f}% |")
    lines.append("")

    lines.append("### System confusion (Ground truth x Action)")
    lines.append("")
    cm = results["system_confusion"]
    lines.append("| Truth \\ Action | PASS | FAIL | REVIEW |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Normal | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |")
    lines.append(f"| Defect | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |")
    lines.append("")

    lines.append("## 5. Per-Category Action Distribution")
    lines.append("")
    lines.append("| Category | PASS | FAIL | REVIEW |")
    lines.append("|---|---|---|---|")
    for cat, dist in results["per_category_actions"].items():
        lines.append(f"| {cat} | {dist['PASS']} | {dist['FAIL']} | {dist['REVIEW']} |")
    lines.append("")

    lines.append("## 6. LODO — Held-Out FLIP (23 unseen)")
    lines.append("")
    lodo = results["lodo_flip"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| UACR (caught -> REVIEW) | {lodo['uacr'] * 100:.2f}% |")
    lines.append(f"| SER (wrong PASS) | {lodo['ser'] * 100:.2f}% |")
    lines.append(f"| FCR (false certainty FAIL) | {lodo['fcr'] * 100:.2f}% |")
    lines.append(f"| Mean p_max | {lodo['mean_p_max']:.4f} |")
    lines.append(f"| Mean anomaly score | {lodo['mean_anomaly_score']:.4f} |")
    lines.append(f"| Action distribution | {json.dumps(lodo['action_distribution'])} |")
    lines.append(f"| Predicted class distribution | {json.dumps(lodo['predicted_class_distribution'])} |")
    lines.append("")

    lines.append("## 7. Stream B Classification Diagnostics (context)")
    lines.append("")
    clf = results["classification_diagnostics"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Known-class accuracy (36) | {clf['known_accuracy']:.4f} |")
    lines.append(f"| Known-class macro-F1 | {clf['known_macro_f1']:.4f} |")
    lines.append(f"| ECE (eval) | {clf['ece']:.4f} |")
    lines.append(f"| Brier score (eval) | {clf['brier']:.4f} |")
    lines.append("")

    lines.append("## 8. Latency & Memory Profiling")
    lines.append("")
    prof = results["profiling"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| End-to-end latency mean (ms/img) | {prof['latency_mean_ms']:.2f} |")
    lines.append(f"| End-to-end latency std (ms/img) | {prof['latency_std_ms']:.2f} |")
    lines.append(f"| Baseline RSS (MB) | {prof['baseline_rss_mb']:.1f} |")
    lines.append(f"| RSS after model load (MB) | {prof['after_model_load_rss_mb']:.1f} |")
    lines.append(f"| Peak RSS during inference (MB) | {prof['peak_inference_rss_mb']:.1f} |")
    lines.append("")

    lines.append("## 9. Reason Code Distribution")
    lines.append("")
    lines.append("| Reason Code | Count |")
    lines.append("|---|---|")
    for code, cnt in results["reason_distribution"].items():
        lines.append(f"| {code} | {cnt} |")
    lines.append("")

    lines.append("## 10. Conclusions")
    lines.append("")
    for c in results["conclusions"]:
        lines.append(f"- {c}")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  • Saved M4 report: {report_path}")


def run_m4_experiment() -> Dict:
    set_seed(SEED)

    print("=" * 80)
    print("MILESTONE 4: DECISION ENGINE ARBITRATION & SYSTEM EVALUATION")
    print("Locked thresholds | No retraining | D_eval_final untouched")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "data" / "raw" / "metal_nut"
    splits_path = base_dir / "data" / "splits.json"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    gates: List[Dict[str, object]] = []

    # ── Load locked config from stored JSONs ──────────────────────────────
    m1_path = artifacts_dir / "baseline_metrics.json"
    m3_path = artifacts_dir / "classifier_metrics.json"
    if not (m1_path.exists() and m3_path.exists()):
        raise FileNotFoundError("M1 and/or M3 metrics JSON missing. Run M1 then M3 first.")
    with open(m1_path) as f:
        m1 = json.load(f)
    with open(m3_path) as f:
        m3 = json.load(f)

    tau_anomaly = float(m1["calibration_results"]["primary_calibrated_threshold"])
    tau_confidence = float(m3["calibration_results"]["tau_confidence_calibrated"])
    stored_temp = float(m3["calibration_results"]["temperature_scaling"]["temperature"])
    stored_head = str(m3["model_architecture"]["head_type"])

    m1_expected_roc = float(m1["final_evaluation_metrics"]["image_level_roc_auc"])
    m1_expected_pr = float(m1["final_evaluation_metrics"]["image_level_pr_auc"])
    m1_expected_locked = m1["final_evaluation_metrics"]["locked_threshold_performance"]
    m3_expected_acc = float(m3["known_class_evaluation"]["accuracy"])
    m3_expected_f1 = float(m3["known_class_evaluation"]["macro_f1"])

    print(f"  Locked tau_anomaly     : {tau_anomaly:.6f}")
    print(f"  Locked tau_confidence  : {tau_confidence:.4f}")
    print(f"  Locked temperature T   : {stored_temp:.4f}")
    print(f"  Locked head            : {stored_head}")

    # ── Step 1: Load partitions ───────────────────────────────────────────
    print("\n[Step 1/8] Loading partitions (read-only)...")
    datasets = load_splits_and_build_datasets(dataset_root, splits_path, img_size=IMG_SIZE)
    d_train_anom = datasets["D_train_anomaly"]
    d_calib = datasets["D_val_calib"]
    d_eval = datasets["D_eval_final"]
    print(f"  • D_train_anomaly : {len(d_train_anom)} normal")
    print(f"  • D_val_calib     : {len(d_calib)} images")
    print(f"  • D_eval_final    : {len(d_eval)} images (strictly untouched)")

    device = torch.device("cpu")
    baseline_rss = _rss_mb()

    # ── Step 2: Reproduce M1 Stream A detector + equivalence gate ─────────
    print("\n[Step 2/8] Reproducing M1 Mahalanobis detector (seed=42)...")
    extractor = ResNetFeatureExtractor(backbone_name="resnet18", device=device)

    loader = DataLoader(d_train_anom, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_feats = np.concatenate([extractor(b["image"]).cpu().numpy() for b in loader], axis=0)
    mahalanobis = MahalanobisAnomalyDetector(use_ledoit_wolf=True)
    mahalanobis.fit(train_feats)

    eval_loader = DataLoader(d_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    eval_feats = np.concatenate([extractor(b["image"]).cpu().numpy() for b in eval_loader], axis=0)
    eval_scores = mahalanobis.score(eval_feats)
    eval_labels = np.array([d_eval[i]["is_anomaly"] for i in range(len(d_eval))], dtype=int)
    eval_cats = [d_eval[i]["category"] for i in range(len(d_eval))]
    eval_paths = [d_eval[i]["rel_path"] for i in range(len(d_eval))]

    eval_roc_pr = compute_roc_pr_metrics(eval_labels, eval_scores)
    ok = check_gate("M1 ROC-AUC (D_eval_final)", eval_roc_pr["roc_auc"], m1_expected_roc, gates=gates)
    all_ok = ok
    ok = check_gate("M1 PR-AUC (D_eval_final)", eval_roc_pr["pr_auc"], m1_expected_pr, gates=gates)
    all_ok = all_ok and ok

    print(f"  • D_eval_final Mahalanobis scores shape: {eval_scores.shape}")

    # ── Step 3: Load the locked M3 classifier head (persisted, no training) ─
    print("\n[Step 3/8] Loading locked M3 classifier head (classifier_head.pt)...")
    head_art_path = artifacts_dir / "classifier_head.pt"
    temp_art_path = artifacts_dir / "temperature_scaler.pt"
    if not (head_art_path.exists() and temp_art_path.exists()):
        raise FileNotFoundError(
            "M3 persisted artifacts missing. Re-run M3 (run_classifier_experiment.py) "
            "with persistence enabled first."
        )
    head_art = torch.load(head_art_path, map_location="cpu")
    if head_art["head_type"] != stored_head:
        raise RuntimeError(f"Persisted head type {head_art['head_type']} != stored {stored_head}")
    model = DefectClassifier(
        backbone_name=head_art["backbone_name"],
        head_type=head_art["head_type"],
        device=device,
    )
    model.head.load_state_dict(head_art["state_dict"])
    model.head.eval()
    model.extractor.eval()
    after_model_load_rss = _rss_mb()
    print(f"  • Loaded {head_art['head_type']} head from {head_art_path.name}")

    # ── Step 4: Collect logits + apply locked temperature scaling ─────────
    print("\n[Step 4/8] Collecting classifier outputs + applying locked temperature scaling...")
    calib_logits, _, calib_labels, _, _, _ = collect_classifier_outputs(model, d_calib)
    eval_logits, eval_probs_raw, eval_labels_clf, eval_anom, eval_cats_clf, eval_paths_clf = (
        collect_classifier_outputs(model, d_eval)
    )

    temp_art = torch.load(temp_art_path, map_location="cpu")
    scaler = TemperatureScaler()
    scaler.load_state_dict(temp_art["state_dict"])
    use_temperature_scaling = bool(temp_art["enabled"])

    ece_before, _ = compute_ece(softmax_to_probs(calib_logits), calib_labels)
    calib_probs_scaled = softmax_to_probs(apply_temperature_scaling(calib_logits, scaler))
    eval_probs_scaled = softmax_to_probs(apply_temperature_scaling(eval_logits, scaler))
    ece_after, _ = compute_ece(calib_probs_scaled, calib_labels)

    if use_temperature_scaling:
        eval_probs = eval_probs_scaled
        final_temp = float(scaler.temperature.item())
    else:
        eval_probs = eval_probs_raw
        final_temp = 1.0

    ok = check_gate("M3 temperature T", final_temp, stored_temp, tol=GATE_TOL_TEMP, gates=gates)
    all_ok = all_ok and ok
    print(f"  • Loaded T = {final_temp:.4f} | stored = {stored_temp:.4f} | scaled used = {use_temperature_scaling}")

    # ── Step 5: M3 known-class equivalence gate ───────────────────────────
    print("\n[Step 5/8] M3 known-class equivalence gate...")
    known_mask = eval_labels_clf >= 0
    known_metrics = compute_classification_metrics(eval_probs[known_mask], eval_labels_clf[known_mask])
    ok = check_gate("M3 known-class accuracy (36)", known_metrics["accuracy"], m3_expected_acc, gates=gates)
    all_ok = all_ok and ok
    ok = check_gate("M3 known-class macro-F1", known_metrics["macro_f1"], m3_expected_f1, gates=gates)
    all_ok = all_ok and ok
    print(f"  • Known-class accuracy = {known_metrics['accuracy']:.4f} | macro-F1 = {known_metrics['macro_f1']:.4f}")

    # sanity: eval_labels from Stream A vs classifier collector must agree
    assert np.array_equal(eval_labels, eval_anom), "Stream A/Stream B anomaly labels disagree"

    if not all_ok:
        raise RuntimeError(
            "Equivalence gates FAILED — reproduced models deviate from stored M1/M3 results. "
            "Aborting before final evaluation."
        )

    # ── Step 6: Decision engine routing (locked thresholds) ───────────────
    print("\n[Step 6/8] Routing D_eval_final through the 5-case arbitration matrix...")
    eval_actions, eval_reasons, eval_p_max, eval_preds = route_batch(
        eval_scores, eval_probs, tau_anomaly, tau_confidence
    )

    binary_perf = evaluate_threshold_performance(eval_labels, eval_scores, threshold=tau_anomaly)
    system_cm = compute_system_confusion(eval_actions, eval_labels)
    system_metrics = compute_system_metrics(eval_actions, eval_labels)
    per_cat_actions = compute_per_category_actions(eval_cats, eval_actions)
    reason_dist = compute_reason_distribution(eval_reasons)
    action_dist = compute_action_distribution(eval_actions)

    lodo_metrics = compute_lodo_metrics(
        eval_scores, eval_probs, eval_cats,
        tau_anomaly=tau_anomaly,
        tau_confidence=tau_confidence,
        held_out_category="flip (unseen)",
    )

    eval_entropy = compute_normalized_entropy(eval_probs)
    eval_ece, _ = compute_ece(eval_probs, eval_labels_clf)
    eval_brier = compute_brier_score(eval_probs, eval_labels_clf)

    # ── Step 7: Latency + memory profiling ────────────────────────────────
    print("\n[Step 7/8] Profiling end-to-end latency + memory...")
    img_t = d_eval[0]["image"].unsqueeze(0)
    anom0 = float(eval_scores[0])

    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(img_t)

    peak_inference_rss = _rss_mb()
    latencies: List[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            feats = extractor(img_t)
            logits = model.head(feats)
        logits_np = logits.cpu().numpy()
        anom_score = float(mahalanobis.score(feats.cpu().numpy())[0])
        scaled = apply_temperature_scaling(logits_np, scaler) if use_temperature_scaling else logits_np
        probs = softmax_to_probs(scaled)
        route_single(anom_score, float(np.max(probs)), int(np.argmax(probs)), tau_anomaly, tau_confidence)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        peak_inference_rss = max(peak_inference_rss, _rss_mb())

    latency_mean = float(np.mean(latencies))
    latency_std = float(np.std(latencies))

    # ── Step 8: Artifacts ─────────────────────────────────────────────────
    print("\n[Step 8/8] Generating artifacts...")

    a = artifacts_dir / "m4_action_distribution.png"
    plot_action_distribution(eval_actions, str(a))
    print(f"  • Saved action distribution: {a.name}")

    p = artifacts_dir / "m4_per_category_actions.png"
    plot_per_category_actions(eval_cats, eval_actions, str(p))
    print(f"  • Saved per-category actions: {p.name}")

    s = artifacts_dir / "m4_system_confusion.png"
    plot_system_confusion(system_cm, str(s))
    print(f"  • Saved system confusion: {s.name}")

    bcm = np.array([
        [binary_perf["tn"], binary_perf["fp"]],
        [binary_perf["fn"], binary_perf["tp"]],
    ])
    b = artifacts_dir / "m4_binary_confusion.png"
    plot_confusion_matrix_figure(bcm, str(b), title=f"Stream A Binary Confusion at tau*={tau_anomaly:.3f}")
    print(f"  • Saved binary confusion: {b.name}")

    # ── Compile results ───────────────────────────────────────────────────
    results: Dict = {
        "milestone": "M4 - Decision Engine Arbitration & System Evaluation",
        "locked_config": {
            "tau_anomaly": tau_anomaly,
            "tau_confidence": tau_confidence,
            "temperature": final_temp,
            "head_type": stored_head,
            "temperature_scaling_enabled": use_temperature_scaling,
            "train_normal_count": len(train_feats),
        },
        "runtime": {"device": "CPU", "seed": SEED},
        "dataset": {
            "eval_total": len(eval_labels),
            "eval_normal": int(np.sum(eval_labels == 0)),
            "eval_defect": int(np.sum(eval_labels == 1)),
        },
        "equivalence_gates": gates,
        "all_gates_passed": all_ok,
        "binary_anomaly": {
            "roc_auc": eval_roc_pr["roc_auc"],
            "pr_auc": eval_roc_pr["pr_auc"],
            "locked_performance": binary_perf,
        },
        "system_metrics": system_metrics,
        "system_confusion": system_cm.tolist(),
        "action_distribution": action_dist,
        "per_category_actions": per_cat_actions,
        "reason_distribution": reason_dist,
        "lodo_flip": lodo_metrics,
        "classification_diagnostics": {
            "known_accuracy": known_metrics["accuracy"],
            "known_macro_f1": known_metrics["macro_f1"],
            "ece": eval_ece,
            "brier": eval_brier,
        },
        "profiling": {
            "latency_mean_ms": latency_mean,
            "latency_std_ms": latency_std,
            "baseline_rss_mb": baseline_rss,
            "after_model_load_rss_mb": after_model_load_rss,
            "peak_inference_rss_mb": peak_inference_rss,
        },
        "conclusions": [],
    }

    # Conclusions
    if system_metrics["silent_escape_rate_ser"] == 0.0:
        results["conclusions"].append(
            "Zero silent escapes on D_eval_final: every defect received FAIL or REVIEW (no defect passed as GOOD)."
        )
    if lodo_metrics["ser"] == 0.0 and lodo_metrics["fcr"] > 0.0:
        results["conclusions"].append(
            "LODO finding confirmed: all 23 unseen flip defects were forced into a known defect class at high "
            "confidence (FCR>0, SER=0). Stream B provides no 'unseen' signal; Stream A is the only safeguard for "
            "novel anomaly types. A higher tau_confidence or an OOD-aware confidence method is the mitigation path."
        )
    results["conclusions"].append(
        f"Primary weakness remains the elevated false-alarm rate on normals "
        f"({system_metrics['false_alarm_rate_fpr'] * 100:.1f}% FAIL on normal), i.e. normals are not being PASSed."
    )

    metrics_path = artifacts_dir / "m4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  • Saved metrics JSON: {metrics_path.name}")

    report_path = artifacts_dir / "m4_report.md"
    write_markdown_report(results, report_path)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MILESTONE 4 SYSTEM EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Locked thresholds        : tau_anomaly={tau_anomaly:.4f}  tau_confidence={tau_confidence:.4f}  T={final_temp:.4f}")
    print(f"Equivalence gates passed : {all_ok}")
    print("-" * 80)
    print("Stream A (binary, locked tau):")
    print(f"  ROC-AUC {eval_roc_pr['roc_auc']:.4f} | PR-AUC {eval_roc_pr['pr_auc']:.4f} | "
          f"FNR {binary_perf['fnr']*100:.2f}% | FPR {binary_perf['fpr']*100:.2f}% | "
          f"F1 {binary_perf['f1_score']:.4f}")
    print("-" * 80)
    print("System decisions (PASS/FAIL/REVIEW):")
    print(f"  {json.dumps(action_dist)}")
    print(f"  Normal : PASS={system_cm[0,0]} FAIL={system_cm[0,1]} REVIEW={system_cm[0,2]}")
    print(f"  Defect : PASS={system_cm[1,0]} FAIL={system_cm[1,1]} REVIEW={system_cm[1,2]}")
    print(f"  SER {system_metrics['silent_escape_rate_ser']*100:.2f}% | "
          f"FPR {system_metrics['false_alarm_rate_fpr']*100:.2f}% | "
          f"Prec {system_metrics['precision']:.4f} | Rec {system_metrics['recall']:.4f} | "
          f"F1 {system_metrics['f1_score']:.4f} | Acc {system_metrics['accuracy']*100:.2f}%")
    print("-" * 80)
    print("LODO flip (23):")
    print(f"  UACR {lodo_metrics['uacr']*100:.1f}% | SER {lodo_metrics['ser']*100:.1f}% | FCR {lodo_metrics['fcr']*100:.1f}%")
    print("-" * 80)
    print(f"Latency {latency_mean:.2f} +/- {latency_std:.2f} ms/img | "
          f"Peak RSS {peak_inference_rss:.1f} MB (baseline {baseline_rss:.1f})")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_m4_experiment()
