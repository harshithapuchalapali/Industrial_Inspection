"""
Milestone 2: PatchCore Anomaly Detection Experiment + Comparative Evaluation.

Protocol
--------
- Same D_train_anomaly, D_val_calib, D_eval_final partitions as M1 (immutable splits.json).
- PatchCore memory bank built from D_train_anomaly using layer2+layer3 hooks on ResNet-18.
- Coreset subsampled to 10% of patches (greedy minimax facility-location).
- Threshold calibrated on D_val_calib using the same cost-sensitive criterion (beta=5.0)
  as M1, ensuring a fair comparison.
- Final evaluation on D_eval_final (untouched holdout).

Outputs
-------
  artifacts/patchcore_metrics.json        — Full numeric results.
  artifacts/patchcore_roc_pr_curve.png    — ROC + PR curves.
  artifacts/patchcore_anomaly_maps.png    — Anomaly map visualisation grid.
  artifacts/m1_vs_m2_comparison.png       — Side-by-side metric comparison bar chart.

NOTE: This script makes NO assumptions about which method will perform better.
      All performance claims are derived from empirical measurement on the
      pre-defined holdout partition (D_eval_final). See baseline_metrics.json for M1 results.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows (prevents UnicodeEncodeError on CP1252 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_engine.data.dataset import load_splits_and_build_datasets
from ml_engine.models.patchcore_extractor import PatchCoreExtractor
from ml_engine.models.patchcore_memory import PatchCoreMemoryBank
from ml_engine.evaluation.metrics import (
    compute_roc_pr_metrics,
    find_cost_sensitive_threshold,
    find_youden_threshold,
    find_f1_threshold,
    evaluate_threshold_performance,
    plot_roc_pr_curves,
    plot_confusion_matrix_figure,
)
from ml_engine.evaluation.localization_metrics import (
    compute_pixel_auroc,
    compute_pro_score,
    plot_anomaly_maps_grid,
    plot_comparison_bar_chart,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
SPATIAL_OUT = (28, 28)          # patch grid resolution
BATCH_SIZE = 8
CORESET_RATIO = 0.10            # keep 10% of patches
COST_BETA = 5.0                 # FN is 5× more expensive than FP


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Data loading helpers ──────────────────────────────────────────────────────

def _collect_patches_from_dataset(
    extractor: PatchCoreExtractor,
    dataset,
    batch_size: int,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Runs the dataset through PatchCoreExtractor and returns:
      - flat_patches : (N_images * N_patches, C) — all patch embeddings stacked.
      - labels       : (N_images,) int array.
      - categories   : List of category strings (length N_images).
      - rel_paths    : List of relative path strings (length N_images).

    The number of patches per image = SPATIAL_OUT[0] * SPATIAL_OUT[1].
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []
    all_cats: List[str] = []
    all_paths: List[str] = []

    for batch in loader:
        imgs = batch["image"]  # (B, 3, H, W)
        # patches: (B, N_patches, C)
        patches_t = extractor.forward(imgs)
        B, N, C = patches_t.shape
        all_patches.append(patches_t.cpu().numpy().reshape(B * N, C))
        all_labels.extend(batch["is_anomaly"].numpy().tolist())
        all_cats.extend(batch["category"])
        all_paths.extend(batch["rel_path"])

    flat_patches = np.concatenate(all_patches, axis=0)  # (N_images*N, C)
    labels_arr = np.array(all_labels, dtype=int)
    return flat_patches, labels_arr, all_cats, all_paths


def _load_raw_images_and_masks(
    dataset,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]]]:
    """
    Loads the raw uint8 RGB images and binary ground-truth masks for
    visualisation and pixel-level evaluation. Masks are None for normal images.

    The dataset returns `mask` as a torch.Tensor (1, H, W) float32.
    We convert it to a (H, W) uint8 numpy array.
    """
    raw_images: List[np.ndarray] = []
    gt_masks: List[Optional[np.ndarray]] = []

    for item in dataset:
        raw_images.append(item["raw_image"])  # already uint8 numpy (H, W, 3)

        mask_raw = item.get("mask", None)
        if mask_raw is not None:
            # mask_raw is a torch.Tensor (1, H, W) float32; convert to (H, W) uint8
            if hasattr(mask_raw, "numpy"):
                mask_np = mask_raw.numpy()  # (1, H, W)
            else:
                mask_np = np.array(mask_raw)
            mask_np = mask_np.squeeze(0).astype(np.uint8)  # (H, W)
            gt_masks.append(mask_np)
        else:
            gt_masks.append(None)

    return raw_images, gt_masks


# ── Main Experiment ───────────────────────────────────────────────────────────

def run_patchcore_experiment():
    set_seed(SEED)

    print("=" * 80)
    print("MILESTONE 2: PATCHCORE ANOMALY DETECTION EXPERIMENT")
    print("Backbone  : Frozen ResNet-18 | Hooks: layer2 + layer3")
    print(f"Patch Grid: {SPATIAL_OUT[0]}x{SPATIAL_OUT[1]} = {SPATIAL_OUT[0]*SPATIAL_OUT[1]} patches/image")
    print(f"Coreset   : {int(CORESET_RATIO*100)}% of training patches (Greedy Minimax)")
    print("=" * 80)

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "data" / "raw" / "metal_nut"
    splits_path = base_dir / "data" / "splits.json"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load Data ─────────────────────────────────────────────────
    print("\n[Step 1/6] Loading data partitions from immutable splits.json...")
    datasets = load_splits_and_build_datasets(dataset_root, splits_path, img_size=IMG_SIZE)

    d_train = datasets["D_train_anomaly"]
    d_calib = datasets["D_val_calib"]
    d_eval = datasets["D_eval_final"]

    print(f"  • D_train_anomaly : {len(d_train):>3} normal images (memory bank source)")
    print(f"  • D_val_calib     : {len(d_calib):>3} images (threshold calibration)")
    print(f"  • D_eval_final    : {len(d_eval):>3} images (untouched holdout)")

    n_patches_per_image = SPATIAL_OUT[0] * SPATIAL_OUT[1]

    # ── Step 2: Initialize PatchCore extractor ────────────────────────────
    print("\n[Step 2/6] Initialising PatchCore patch-feature extractor (ResNet-18)...")
    device = torch.device("cpu")
    extractor = PatchCoreExtractor(
        backbone_name="resnet18",
        hook_layers=["layer2", "layer3"],
        target_spatial=SPATIAL_OUT,
        device=device,
    )
    print(f"  • Patch embedding dimension : {extractor.embedding_dim}-d  "
          f"(layer2={128}-d + layer3={256}-d)")

    # ── Step 3: Build Memory Bank ─────────────────────────────────────────
    print("\n[Step 3/6] Extracting training patches and building coreset memory bank...")
    t0 = time.perf_counter()

    train_patches, train_labels, _, _ = _collect_patches_from_dataset(
        extractor, d_train, BATCH_SIZE, desc="Training"
    )
    extract_time = time.perf_counter() - t0
    print(f"  • Extracted {len(train_patches):,} training patches in {extract_time:.1f}s")
    print(f"    (from {len(d_train)} images × {n_patches_per_image} patches/image)")

    memory_bank = PatchCoreMemoryBank(
        n_neighbors=9,
        coreset_ratio=CORESET_RATIO,
        coreset_max_samples=30_000,
        random_seed=SEED,
    )
    t1 = time.perf_counter()
    memory_bank.fit(train_patches)
    coreset_time = time.perf_counter() - t1
    print(f"  • Coreset built: {memory_bank.bank_size:,} patches retained "
          f"({memory_bank.bank_size/len(train_patches)*100:.1f}% of {len(train_patches):,}) "
          f"in {coreset_time:.1f}s")

    # ── Step 4: Score Calibration + Eval Sets ─────────────────────────────
    print("\n[Step 4/6] Computing patch scores for calibration and evaluation sets...")
    t2 = time.perf_counter()

    calib_patches, calib_labels, calib_cats, _ = _collect_patches_from_dataset(
        extractor, d_calib, BATCH_SIZE
    )
    eval_patches, eval_labels, eval_cats, eval_paths = _collect_patches_from_dataset(
        extractor, d_eval, BATCH_SIZE
    )
    score_extract_time = time.perf_counter() - t2
    print(f"  • Feature extraction (calib + eval) : {score_extract_time:.1f}s")

    # Image-level scores (max patch distance per image)
    calib_scores, calib_amaps = memory_bank.score_dataset(
        calib_patches, n_patches_per_image, SPATIAL_OUT
    )
    eval_scores, eval_amaps = memory_bank.score_dataset(
        eval_patches, n_patches_per_image, SPATIAL_OUT
    )
    print(f"  • Calibration set scored : {len(calib_scores)} images")
    print(f"  • Evaluation set scored  : {len(eval_scores)} images")

    # ── Step 5: Threshold Calibration ─────────────────────────────────────
    print(f"\n[Step 5/6] Calibrating decision threshold on D_val_calib (beta={COST_BETA})...")

    tau_primary, cost_primary, stats_primary = find_cost_sensitive_threshold(
        calib_labels, calib_scores, beta=COST_BETA
    )
    tau_youden, j_val = find_youden_threshold(calib_labels, calib_scores)
    tau_f1, f1_val = find_f1_threshold(calib_labels, calib_scores)

    print(f"  • Cost-Sensitive (beta={COST_BETA}) : tau* = {tau_primary:.4f} "
          f"(Val FN={stats_primary['fn']}, FP={stats_primary['fp']})")
    print(f"  • Youden-J                       : tau* = {tau_youden:.4f} (J={j_val:.4f})")
    print(f"  • F1-Max                         : tau* = {tau_f1:.4f} (F1={f1_val:.4f})")

    # Sensitivity analysis across beta values
    print("\n--- Threshold Sensitivity Analysis ---")
    sensitivity_results = {}
    for beta_val in [2.0, 5.0, 10.0, 20.0]:
        tau_b, cost_b, stats_b = find_cost_sensitive_threshold(calib_labels, calib_scores, beta=beta_val)
        sensitivity_results[f"beta_{beta_val}"] = {
            "beta": beta_val,
            "threshold": float(tau_b),
            "cost": float(cost_b),
            "validation_fn": int(stats_b["fn"]),
            "validation_fp": int(stats_b["fp"]),
        }
        print(f"  • beta = {beta_val:>4.1f} → tau* = {tau_b:.4f} | Val FN = {stats_b['fn']} | Val FP = {stats_b['fp']}")

    # ── Step 6: Final Evaluation on D_eval_final ──────────────────────────
    print("\n[Step 6/6] Evaluating PatchCore on untouched D_eval_final...")

    # Image-level metrics
    eval_roc_pr = compute_roc_pr_metrics(eval_labels, eval_scores)
    final_perf = evaluate_threshold_performance(eval_labels, eval_scores, threshold=tau_primary)

    # Pixel-level metrics (only on defective images that have GT masks)
    print("  • Loading raw images and GT masks for pixel-level evaluation...")
    raw_images, gt_masks = _load_raw_images_and_masks(d_eval)

    # Only score pixel-level on defective images (those with non-None masks)
    defect_mask_list = [gt_masks[i] for i in range(len(eval_labels)) if eval_labels[i] == 1]
    defect_amap_list = [eval_amaps[i] for i in range(len(eval_labels)) if eval_labels[i] == 1]

    if any(m is not None for m in defect_mask_list):
        print("  • Computing pixel-level AUROC...")
        pixel_auroc = compute_pixel_auroc(defect_mask_list, defect_amap_list, target_size=(IMG_SIZE, IMG_SIZE))
        print(f"    Pixel-Level AUROC : {pixel_auroc:.4f}")

        print("  • Computing Per-Region Overlap (PRO) score [FPR ≤ 0.30]...")
        # PRO needs full image context (normal + defect maps + masks)
        pro_score = compute_pro_score(gt_masks, list(eval_amaps), target_size=(IMG_SIZE, IMG_SIZE), fpr_limit=0.30)
        print(f"    PRO Score (FPR≤30%): {pro_score:.4f}")
    else:
        print("  ! No GT pixel masks available for pixel-level evaluation. Skipping.")
        pixel_auroc = float("nan")
        pro_score = float("nan")

    # Inference latency profiling
    sample_img = d_eval[0]["image"].unsqueeze(0)
    latencies = []
    for _ in range(50):
        t_s = time.perf_counter()
        with torch.no_grad():
            patches = extractor.forward(sample_img).cpu().numpy().reshape(-1, extractor.embedding_dim)
        _ = memory_bank.score_patches(patches)
        latencies.append((time.perf_counter() - t_s) * 1000.0)
    latency_mean = float(np.mean(latencies))
    latency_std = float(np.std(latencies))

    # Per-category breakdown
    cat_scores: Dict[str, np.ndarray] = {}
    unique_cats = sorted(set(eval_cats))
    for c in unique_cats:
        mask = np.array([x == c for x in eval_cats])
        cat_scores[c] = eval_scores[mask]

    # ── Artifacts ─────────────────────────────────────────────────────────
    print("\n--- Generating Artifacts ---")

    # ROC / PR curves
    roc_pr_path = artifacts_dir / "patchcore_roc_pr_curve.png"
    plot_roc_pr_curves(eval_roc_pr, str(roc_pr_path), title_suffix="PatchCore (ResNet-18 Patch)")
    print(f"  • ROC/PR Curves      → {roc_pr_path.name}")

    # Confusion matrix
    cm = np.array([
        [final_perf["tn"], final_perf["fp"]],
        [final_perf["fn"], final_perf["tp"]],
    ])
    cm_path = artifacts_dir / "patchcore_confusion_matrix.png"
    plot_confusion_matrix_figure(cm, str(cm_path),
                                 title=f"PatchCore Confusion Matrix at tau*={tau_primary:.3f}")
    print(f"  • Confusion Matrix   → {cm_path.name}")

    # Anomaly map grid
    amap_path = artifacts_dir / "patchcore_anomaly_maps.png"
    plot_anomaly_maps_grid(
        images=raw_images,
        anomaly_maps=list(eval_amaps),
        gt_masks=gt_masks,
        image_scores=list(eval_scores),
        labels=list(eval_labels),
        save_path=str(amap_path),
        n_cols=5,
        target_size=(IMG_SIZE, IMG_SIZE),
        title="PatchCore Anomaly Maps — Top Scoring Defects & High-Score Normals",
    )
    print(f"  • Anomaly Map Grid   → {amap_path.name}")

    # M1 vs M2 comparison bar chart (load M1 metrics from saved JSON)
    m1_json_path = artifacts_dir / "baseline_metrics.json"
    if m1_json_path.exists():
        with open(m1_json_path) as f:
            m1_data = json.load(f)
        m1_eval = m1_data["final_evaluation_metrics"]
        m1_locked = m1_eval["locked_threshold_performance"]
        baseline_chart_metrics = {
            "ROC-AUC": round(m1_eval["image_level_roc_auc"], 4),
            "PR-AUC": round(m1_eval["image_level_pr_auc"], 4),
            "Recall": round(m1_locked["recall"], 4),
            "Precision": round(m1_locked["precision"], 4),
            "1 - FPR": round(1 - m1_locked["fpr"], 4),
            "1 - FNR": round(1 - m1_locked["fnr"], 4),
        }
        patchcore_chart_metrics = {
            "ROC-AUC": round(eval_roc_pr["roc_auc"], 4),
            "PR-AUC": round(eval_roc_pr["pr_auc"], 4),
            "Recall": round(final_perf["recall"], 4),
            "Precision": round(final_perf["precision"], 4),
            "1 - FPR": round(1 - final_perf["fpr"], 4),
            "1 - FNR": round(1 - final_perf["fnr"], 4),
        }
        compare_path = artifacts_dir / "m1_vs_m2_comparison.png"
        plot_comparison_bar_chart(baseline_chart_metrics, patchcore_chart_metrics, str(compare_path))
        print(f"  • M1 vs M2 Comparison → {compare_path.name}")
    else:
        logger.warning("baseline_metrics.json not found — skipping comparison chart.")
        baseline_chart_metrics = {}
        patchcore_chart_metrics = {}

    # ── Metrics JSON ──────────────────────────────────────────────────────
    results_summary = {
        "milestone": "M2 - PatchCore Anomaly Detection",
        "model_architecture": {
            "backbone": "ResNet-18 (Pretrained ImageNet-1k, Frozen)",
            "hook_layers": ["layer2", "layer3"],
            "patch_grid": f"{SPATIAL_OUT[0]}x{SPATIAL_OUT[1]}",
            "patches_per_image": n_patches_per_image,
            "embedding_dim_per_patch": extractor.embedding_dim,
            "coreset_ratio": CORESET_RATIO,
            "memory_bank_size": memory_bank.bank_size,
            "n_neighbors_knn": 9,
        },
        "timing_and_profiling": {
            "training_patch_extraction_sec": round(extract_time, 2),
            "coreset_build_sec": round(coreset_time, 2),
            "inference_latency_mean_ms": round(latency_mean, 3),
            "inference_latency_std_ms": round(latency_std, 3),
            "device": "CPU",
        },
        "calibration_results": {
            "dataset_size": len(calib_labels),
            "primary_beta": COST_BETA,
            "primary_calibrated_threshold": float(tau_primary),
            "sensitivity_analysis": sensitivity_results,
            "secondary_youden_threshold": float(tau_youden),
            "secondary_f1_threshold": float(tau_f1),
        },
        "final_evaluation_metrics": {
            "dataset_size": len(eval_labels),
            "normal_count": int(np.sum(eval_labels == 0)),
            "defect_count": int(np.sum(eval_labels == 1)),
            "image_level_roc_auc": round(eval_roc_pr["roc_auc"], 6),
            "image_level_pr_auc": round(eval_roc_pr["pr_auc"], 6),
            "pixel_level_auroc": round(pixel_auroc, 6) if not np.isnan(pixel_auroc) else None,
            "pro_score_fpr30": round(pro_score, 6) if not np.isnan(pro_score) else None,
            "locked_threshold_performance": final_perf,
            "per_category_scores": {
                c: {
                    "count": int(len(s)),
                    "mean_score": float(np.mean(s)),
                    "std_score": float(np.std(s)),
                    "min_score": float(np.min(s)),
                    "max_score": float(np.max(s)),
                    "flagged_as_anomaly_rate": float(np.mean(s >= tau_primary)),
                }
                for c, s in cat_scores.items()
            },
        },
    }

    metrics_json_path = artifacts_dir / "patchcore_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  • Metrics JSON       → {metrics_json_path.name}")

    # ── Final Summary Table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("MILESTONE 2  —  PATCHCORE EMPIRICAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Memory Bank Size             : {memory_bank.bank_size:,} patches  "
          f"(from {len(train_patches):,} raw patches)")
    print(f"Patch Embedding Dim          : {extractor.embedding_dim}-d "
          f"(layer2 128-d ‖ layer3 256-d)")
    print(f"Evaluation Samples           : {len(eval_labels)} "
          f"({int(np.sum(eval_labels==0))} Normal, {int(np.sum(eval_labels==1))} Defective)")
    print("-" * 80)
    print(f"Image-Level ROC-AUC          : {eval_roc_pr['roc_auc']:.4f}")
    print(f"Image-Level PR-AUC           : {eval_roc_pr['pr_auc']:.4f}")
    if not np.isnan(pixel_auroc):
        print(f"Pixel-Level AUROC            : {pixel_auroc:.4f}")
        print(f"PRO Score (FPR ≤ 30%)        : {pro_score:.4f}")
    print("-" * 80)
    print(f"Locked Threshold (tau*)      : {tau_primary:.4f}  (beta={COST_BETA})")
    print(f"True Positives  (TP)         : {final_perf['tp']:>2} / {int(np.sum(eval_labels==1))}")
    print(f"True Negatives  (TN)         : {final_perf['tn']:>2} / {int(np.sum(eval_labels==0))}")
    print(f"False Positives (FP)         : {final_perf['fp']:>2} / {int(np.sum(eval_labels==0))}  "
          f"(False Alarm Rate: {final_perf['fpr']*100:.2f}%)")
    print(f"False Negatives (FN)         : {final_perf['fn']:>2} / {int(np.sum(eval_labels==1))}  "
          f"(Escape Rate / FNR: {final_perf['fnr']*100:.2f}%)")
    print(f"Precision                    : {final_perf['precision']:.4f}")
    print(f"Recall (Sensitivity)         : {final_perf['recall']:.4f}")
    print(f"F1-Score                     : {final_perf['f1_score']:.4f}")
    print(f"Overall Accuracy             : {final_perf['accuracy']*100:.2f}%")
    print("-" * 80)
    print("Per-Category Breakdown:")
    for c, info in results_summary["final_evaluation_metrics"]["per_category_scores"].items():
        print(f"  • {c:<16}: Mean = {info['mean_score']:>7.4f} | "
              f"Flagged = {info['flagged_as_anomaly_rate']*100:>5.1f}% ({info['count']} samples)")
    print("-" * 80)
    print(f"Inference Latency (CPU)      : {latency_mean:.2f} ± {latency_std:.2f} ms/image")
    print("=" * 80)

    # ── Comparison Summary ────────────────────────────────────────────────
    if m1_json_path.exists():
        m1_roc = m1_eval["image_level_roc_auc"]
        m2_roc = eval_roc_pr["roc_auc"]
        m1_fpr = m1_locked["fpr"]
        m2_fpr = final_perf["fpr"]
        m1_fnr = m1_locked["fnr"]
        m2_fnr = final_perf["fnr"]
        print("\nM1 BASELINE vs M2 PATCHCORE — MEASURED DELTA")
        print("-" * 80)
        print(f"  ROC-AUC  : M1={m1_roc:.4f}  →  M2={m2_roc:.4f}  "
              f"Δ={m2_roc-m1_roc:+.4f}")
        print(f"  PR-AUC   : M1={m1_eval['image_level_pr_auc']:.4f}  →  "
              f"M2={eval_roc_pr['pr_auc']:.4f}  "
              f"Δ={eval_roc_pr['pr_auc']-m1_eval['image_level_pr_auc']:+.4f}")
        print(f"  FPR      : M1={m1_fpr*100:.2f}%  →  M2={m2_fpr*100:.2f}%  "
              f"Δ={( m2_fpr-m1_fpr)*100:+.2f}pp")
        print(f"  FNR      : M1={m1_fnr*100:.2f}%  →  M2={m2_fnr*100:.2f}%  "
              f"Δ={(m2_fnr-m1_fnr)*100:+.2f}pp")
        print("=" * 80)

    return results_summary


if __name__ == "__main__":
    run_patchcore_experiment()
