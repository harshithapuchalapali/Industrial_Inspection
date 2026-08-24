"""
Milestone 3: Known-Defect Classifier + Calibration + LODO Experiment.

Protocol
--------
- Stream B 4-class classifier trained on D_train_clf (84 images).
- Compare linear probe vs 2-layer MLP; select best head on D_val_calib accuracy.
- Experimental Temperature Scaling on D_val_calib (N=28); keep only if ECE improves.
- Calibrate tau_confidence on D_val_calib using provisional decision engine routing.
- Stream A anomaly scores from M1 Mahalanobis detector (same tau_anomaly from M1).
- LODO evaluation on 23 held-out flip images in D_eval_final.

Outputs
-------
  artifacts/classifier_metrics.json
  artifacts/classifier_confusion_matrix.png
  artifacts/classifier_reliability_diagram.png
  artifacts/classifier_entropy_distribution.png
  artifacts/lodo_flip_summary.png
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
    train_classifier,
)
from ml_engine.models.temperature_scaling import (
    apply_temperature_scaling,
    fit_temperature_scaler,
)
from ml_engine.evaluation.classification_metrics import (
    CLASS_NAMES,
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_normalized_entropy,
    plot_entropy_distribution,
    plot_multiclass_confusion_matrix,
    plot_reliability_diagram,
    softmax_to_probs,
)
from ml_engine.evaluation.decision_engine import (
    compute_lodo_metrics,
    find_tau_confidence,
    route_batch,
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
COST_BETA = 5.0
TRAIN_EPOCHS = 80


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def extract_anomaly_scores(
    extractor: ResNetFeatureExtractor,
    detector: MahalanobisAnomalyDetector,
    dataset,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_scores: List[np.ndarray] = []
    all_anomaly: List[int] = []
    all_categories: List[str] = []

    for batch in loader:
        feats = extractor(batch["image"]).cpu().numpy()
        scores = detector.score(feats)
        all_scores.append(scores)
        all_anomaly.extend(batch["is_anomaly"].numpy().tolist())
        all_categories.extend(batch["category"])

    return (
        np.concatenate(all_scores, axis=0),
        np.array(all_anomaly, dtype=int),
        all_categories,
    )


def select_best_head(
    device: torch.device,
    train_dataset,
    calib_dataset,
) -> Tuple[str, DefectClassifier, Dict[str, float]]:
    """Trains linear and MLP heads; picks the one with higher D_val_calib accuracy."""
    best_head = "mlp"
    best_model: DefectClassifier | None = None
    best_acc = -1.0
    summaries: Dict[str, Dict[str, float]] = {}

    for head_type in ("linear", "mlp"):
        logger.info("Training %s head on D_train_clf...", head_type)
        model = DefectClassifier(backbone_name="resnet18", head_type=head_type, device=device)
        train_summary = train_classifier(
            model,
            train_dataset,
            epochs=TRAIN_EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )

        _, calib_probs, calib_labels, _, _, _ = collect_classifier_outputs(model, calib_dataset)
        valid = calib_labels >= 0
        calib_preds = np.argmax(calib_probs[valid], axis=1)
        calib_acc = float(np.mean(calib_preds == calib_labels[valid]))

        summaries[head_type] = {
            **train_summary,
            "calib_accuracy": calib_acc,
        }
        logger.info("  %s head -> D_val_calib accuracy = %.4f", head_type, calib_acc)

        if calib_acc > best_acc:
            best_acc = calib_acc
            best_head = head_type
            best_model = model

    assert best_model is not None
    return best_head, best_model, summaries


def plot_lodo_summary(lodo_metrics: Dict, save_path: str):
    """Bar chart for UACR / SER / FCR on held-out flip defects."""
    labels = ["UACR\n(caught → REVIEW)", "SER\n(wrong PASS)", "FCR\n(false certainty FAIL)"]
    values = [
        lodo_metrics["uacr"] * 100,
        lodo_metrics["ser"] * 100,
        lodo_metrics["fcr"] * 100,
    ]
    colors = ["#2196f3", "#f44336", "#ff9800"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage of Flip Samples (N=23)", fontsize=11, fontweight="bold")
    ax.set_title("LODO Evaluation on Held-Out FLIP Defects", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_classifier_experiment():
    set_seed(SEED)

    print("=" * 80)
    print("MILESTONE 3: KNOWN-DEFECT CLASSIFIER + CALIBRATION + LODO")
    print("Stream B: 4-Class Classifier (GOOD, BENT, COLOR, SCRATCH)")
    print("LODO Held-Out Category: flip (23 unseen images)")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "data" / "raw" / "metal_nut"
    splits_path = base_dir / "data" / "splits.json"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = artifacts_dir / "baseline_metrics.json"
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(
            "baseline_metrics.json not found. Run M1 (run_baseline_experiment.py) first."
        )
    with open(baseline_metrics_path, "r") as f:
        baseline_metrics = json.load(f)
    tau_anomaly = float(
        baseline_metrics["calibration_results"]["primary_calibrated_threshold"]
    )

    # ── Step 1: Load partitions ─────────────────────────────────────────────
    print("\n[Step 1/7] Loading data partitions from immutable splits.json...")
    datasets = load_splits_and_build_datasets(dataset_root, splits_path, img_size=IMG_SIZE)
    d_train_clf = datasets["D_train_clf"]
    d_calib = datasets["D_val_calib"]
    d_eval = datasets["D_eval_final"]
    d_train_anomaly = datasets["D_train_anomaly"]

    print(f"  • D_train_clf   : {len(d_train_clf):>3} images (Stream B training)")
    print(f"  • D_val_calib   : {len(d_calib):>3} images (calibration)")
    print(f"  • D_eval_final  : {len(d_eval):>3} images (final evaluation + LODO)")
    print(f"  • tau_anomaly*  : {tau_anomaly:.4f} (locked from M1 baseline)")

    device = torch.device("cpu")

    # ── Step 2: Train classifier (head selection) ─────────────────────────────
    print("\n[Step 2/7] Training Stream B classifier heads on D_train_clf...")
    t0 = time.perf_counter()
    best_head, model, head_summaries = select_best_head(device, d_train_clf, d_calib)
    train_time = time.perf_counter() - t0
    print(f"  • Selected head: {best_head} (best D_val_calib accuracy)")
    print(f"  • Training time : {train_time:.2f}s")

    # ── Persist the selected head for downstream milestones ────────────────
    # NOTE: 80-epoch CPU training is NOT run-to-run reproducible; persisting the
    # weights is required so M4 can use the exact locked model without re-training.
    head_save_path = artifacts_dir / "classifier_head.pt"
    torch.save(
        {
            "backbone_name": "resnet18",
            "head_type": best_head,
            "state_dict": model.head.state_dict(),
        },
        head_save_path,
    )
    print(f"  • Persisted classifier head weights: {head_save_path.name}")

    # ── Step 3: Collect logits on calibration & eval ────────────────────────────
    print("\n[Step 3/7] Collecting classifier outputs...")
    calib_logits, calib_probs_raw, calib_labels, calib_anom, calib_cats, _ = (
        collect_classifier_outputs(model, d_calib)
    )
    eval_logits, eval_probs_raw, eval_labels, eval_anom, eval_cats, eval_paths = (
        collect_classifier_outputs(model, d_eval)
    )

    # ── Step 4: Temperature scaling (experimental) ────────────────────────────
    print("\n[Step 4/7] Fitting experimental Temperature Scaling on D_val_calib...")
    scaler, temp_summary = fit_temperature_scaler(calib_logits, calib_labels)

    ece_before, ece_details_before = compute_ece(calib_probs_raw, calib_labels)
    brier_before = compute_brier_score(calib_probs_raw, calib_labels)

    calib_logits_scaled = apply_temperature_scaling(calib_logits, scaler)
    eval_logits_scaled = apply_temperature_scaling(eval_logits, scaler)
    calib_probs = softmax_to_probs(calib_logits_scaled)
    eval_probs = softmax_to_probs(eval_logits_scaled)

    ece_after, ece_details_after = compute_ece(calib_probs, calib_labels)
    brier_after = compute_brier_score(calib_probs, calib_labels)

    use_temperature_scaling = ece_after <= ece_before
    if not use_temperature_scaling:
        calib_probs = calib_probs_raw
        eval_probs = eval_probs_raw
        temp_summary["temperature"] = 1.0
        ece_after = ece_before
        ece_details_after = ece_details_before
        brier_after = brier_before

    print(f"  • Temperature T        : {temp_summary['temperature']:.4f}")
    print(f"  • ECE before / after   : {ece_before:.4f} / {ece_after:.4f}")
    print(f"  • Brier before / after : {brier_before:.4f} / {brier_after:.4f}")
    print(f"  • Using scaled logits  : {use_temperature_scaling}")

    # Persist the temperature scaler (enabled flag + learned T) for M4.
    temp_save_path = artifacts_dir / "temperature_scaler.pt"
    torch.save(
        {
            "enabled": bool(use_temperature_scaling),
            "state_dict": scaler.state_dict(),
        },
        temp_save_path,
    )
    print(f"  • Persisted temperature scaler: {temp_save_path.name}")

    # ── Step 5: Stream A anomaly scores for dual-stream routing ───────────────
    print("\n[Step 5/7] Computing Stream A Mahalanobis scores for routing...")
    extractor = ResNetFeatureExtractor(backbone_name="resnet18", device=device)
    train_loader = DataLoader(d_train_anomaly, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_feats = []
    for batch in train_loader:
        train_feats.append(extractor(batch["image"]).cpu().numpy())
    train_feats = np.concatenate(train_feats, axis=0)

    mahalanobis = MahalanobisAnomalyDetector(use_ledoit_wolf=True)
    mahalanobis.fit(train_feats)

    calib_anomaly_scores, _, _ = extract_anomaly_scores(extractor, mahalanobis, d_calib)
    eval_anomaly_scores, _, _ = extract_anomaly_scores(extractor, mahalanobis, d_eval)

    # ── Step 6: Calibrate tau_confidence ──────────────────────────────────────
    print("\n[Step 6/7] Calibrating tau_confidence on D_val_calib...")
    tau_confidence, tau_conf_stats = find_tau_confidence(
        calib_anomaly_scores,
        calib_probs,
        calib_labels,
        calib_anom,
        tau_anomaly=tau_anomaly,
        beta=COST_BETA,
    )
    print(f"  • tau_confidence* : {tau_confidence:.4f}")
    print(
        f"  • Calib routing cost: {tau_conf_stats['cost']:.1f} "
        f"(false PASS={tau_conf_stats['false_pass']}, false FAIL={tau_conf_stats['false_fail']})"
    )

    # ── Step 7: Final evaluation ──────────────────────────────────────────────
    print("\n[Step 7/7] Evaluating on D_eval_final (known classes + LODO flip)...")

    known_eval_mask = eval_labels >= 0
    known_metrics = compute_classification_metrics(
        eval_probs[known_eval_mask],
        eval_labels[known_eval_mask],
    )
    eval_ece, eval_ece_details = compute_ece(eval_probs, eval_labels)
    eval_brier = compute_brier_score(eval_probs, eval_labels)
    eval_entropy = compute_normalized_entropy(eval_probs)

    lodo_metrics = compute_lodo_metrics(
        eval_anomaly_scores,
        eval_probs,
        eval_cats,
        tau_anomaly=tau_anomaly,
        tau_confidence=tau_confidence,
        held_out_category="flip (unseen)",
    )

    eval_actions, eval_reasons, eval_p_max, eval_preds = route_batch(
        eval_anomaly_scores, eval_probs, tau_anomaly, tau_confidence
    )

    # Latency profiling
    single_img = d_eval[0]["image"].unsqueeze(0)
    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(single_img)
        latencies.append((time.perf_counter() - t_start) * 1000.0)
    latency_mean = float(np.mean(latencies))
    latency_std = float(np.std(latencies))

    # ── Artifacts ─────────────────────────────────────────────────────────────
    print("\n--- Generating Artifacts and Visualizations ---")

    cm_path = artifacts_dir / "classifier_confusion_matrix.png"
    plot_multiclass_confusion_matrix(
        known_metrics["confusion_matrix"],
        str(cm_path),
        title=f"Stream B Confusion Matrix ({best_head} head, known classes only)",
    )
    print(f"  • Saved confusion matrix: {cm_path}")

    rel_path = artifacts_dir / "classifier_reliability_diagram.png"
    plot_reliability_diagram(
        ece_details_after if use_temperature_scaling else ece_details_before,
        str(rel_path),
        ece_value=ece_after,
        title="Stream B Reliability Diagram (D_val_calib)",
    )
    print(f"  • Saved reliability diagram: {rel_path}")

    entropy_path = artifacts_dir / "classifier_entropy_distribution.png"
    plot_entropy_distribution(
        eval_entropy,
        eval_cats,
        str(entropy_path),
        title="Normalized Entropy on D_eval_final",
    )
    print(f"  • Saved entropy distribution: {entropy_path}")

    lodo_plot_path = artifacts_dir / "lodo_flip_summary.png"
    plot_lodo_summary(lodo_metrics, str(lodo_plot_path))
    print(f"  • Saved LODO summary: {lodo_plot_path}")

    # Per-category breakdown on eval
    per_category: Dict[str, Dict] = {}
    for cat in sorted(set(eval_cats)):
        mask = np.array([c == cat for c in eval_cats])
        cat_actions = eval_actions[mask]
        per_category[cat] = {
            "count": int(np.sum(mask)),
            "mean_p_max": float(np.mean(eval_p_max[mask])),
            "mean_entropy": float(np.mean(eval_entropy[mask])),
            "mean_anomaly_score": float(np.mean(eval_anomaly_scores[mask])),
            "action_distribution": {
                "PASS": int(np.sum(cat_actions == "PASS")),
                "FAIL": int(np.sum(cat_actions == "FAIL")),
                "REVIEW": int(np.sum(cat_actions == "REVIEW")),
            },
        }

    results = {
        "milestone": "M3 - Known-Defect Classifier + Calibration + LODO",
        "reproducibility_note": (
            "80-epoch CPU training is not run-to-run reproducible on this machine. "
            "The selected head weights and temperature scaler are therefore persisted for M4."
        ),
        "persisted_artifacts": {
            "classifier_head_weights": "classifier_head.pt",
            "temperature_scaler": "temperature_scaler.pt",
        },
        "model_architecture": {
            "backbone": "ResNet-18 (Pretrained ImageNet-1k, Frozen)",
            "head_type": best_head,
            "num_classes": 4,
            "class_names": CLASS_NAMES,
            "head_selection_summary": head_summaries,
        },
        "timing_and_profiling": {
            "training_time_sec": train_time,
            "single_image_latency_mean_ms": latency_mean,
            "single_image_latency_std_ms": latency_std,
            "device": "CPU",
        },
        "calibration_results": {
            "calibration_partition_size": len(calib_labels),
            "temperature_scaling": {
                "enabled": use_temperature_scaling,
                "temperature": temp_summary["temperature"],
                "nll_before": temp_summary["nll_before"],
                "nll_after": temp_summary["nll_after"],
                "ece_before": ece_before,
                "ece_after": ece_after,
                "brier_before": brier_before,
                "brier_after": brier_after,
            },
            "tau_anomaly_locked_from_m1": tau_anomaly,
            "tau_confidence_calibrated": tau_confidence,
            "tau_confidence_calibration_stats": tau_conf_stats,
        },
        "known_class_evaluation": {
            "partition": "D_eval_final (excluding flip)",
            "sample_count": int(np.sum(known_eval_mask)),
            "accuracy": known_metrics["accuracy"],
            "macro_precision": known_metrics["macro_precision"],
            "macro_recall": known_metrics["macro_recall"],
            "macro_f1": known_metrics["macro_f1"],
            "per_class": known_metrics["per_class"],
            "confusion_matrix": known_metrics["confusion_matrix"].tolist(),
        },
        "full_eval_calibration_diagnostics": {
            "ece": eval_ece,
            "brier_score": eval_brier,
        },
        "lodo_flip_evaluation": lodo_metrics,
        "decision_routing_on_eval_final": {
            "tau_anomaly": tau_anomaly,
            "tau_confidence": tau_confidence,
            "per_category": per_category,
            "overall_action_distribution": {
                "PASS": int(np.sum(eval_actions == "PASS")),
                "FAIL": int(np.sum(eval_actions == "FAIL")),
                "REVIEW": int(np.sum(eval_actions == "REVIEW")),
            },
        },
    }

    metrics_path = artifacts_dir / "classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  • Saved metrics JSON: {metrics_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MILESTONE 3 EMPIRICAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Selected Head              : {best_head}")
    print(f"Known-Class Eval Samples   : {int(np.sum(known_eval_mask))} (excludes 23 flip)")
    print("-" * 80)
    print(f"Accuracy                   : {known_metrics['accuracy']:.4f}")
    print(f"Macro Precision            : {known_metrics['macro_precision']:.4f}")
    print(f"Macro Recall               : {known_metrics['macro_recall']:.4f}")
    print(f"Macro F1                   : {known_metrics['macro_f1']:.4f}")
    print("-" * 80)
    print(f"ECE (calib)                : {ece_after:.4f}")
    print(f"Brier (calib)              : {brier_after:.4f}")
    print(f"Temperature Scaling Used   : {use_temperature_scaling}")
    print("-" * 80)
    print(f"tau_confidence*            : {tau_confidence:.4f}")
    print(f"LODO flip samples          : {lodo_metrics['sample_count']}")
    print(f"  UACR (→ REVIEW)          : {lodo_metrics['uacr']*100:.1f}%")
    print(f"  SER  (wrong PASS)        : {lodo_metrics['ser']*100:.1f}%")
    print(f"  FCR  (false certainty)   : {lodo_metrics['fcr']*100:.1f}%")
    print("-" * 80)
    print(f"Inference Latency (CPU)    : {latency_mean:.2f} +/- {latency_std:.2f} ms/image")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_classifier_experiment()
