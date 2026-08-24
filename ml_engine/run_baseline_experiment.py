import sys
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_engine.data.dataset import load_splits_and_build_datasets
from ml_engine.models.baseline_extractor import ResNetFeatureExtractor
from ml_engine.models.baseline_knn import MahalanobisAnomalyDetector, KNNAnomalyDetector
from ml_engine.evaluation.metrics import (
    compute_roc_pr_metrics,
    find_cost_sensitive_threshold,
    find_youden_threshold,
    find_f1_threshold,
    evaluate_threshold_performance,
    plot_roc_pr_curves,
    plot_score_distributions_by_category,
    plot_confusion_matrix_figure
)

def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_features_from_dataset(
    feature_extractor: ResNetFeatureExtractor,
    dataset,
    batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extracts feature embeddings, anomaly labels, categories, and relative paths.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings = []
    all_labels = []
    all_categories = []
    all_paths = []
    
    for batch in loader:
        imgs = batch["image"]
        embeddings = feature_extractor(imgs).cpu().numpy()
        all_embeddings.append(embeddings)
        all_labels.extend(batch["is_anomaly"].numpy().tolist())
        all_categories.extend(batch["category"])
        all_paths.extend(batch["rel_path"])
        
    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.array(all_labels, dtype=int)
    return embeddings_arr, labels_arr, all_categories, all_paths

def run_experiment():
    set_deterministic_seed(42)
    
    print("=" * 80)
    print("MILESTONE 1: BASELINE ANOMALY DETECTION EXPERIMENT")
    print("Backbone: Frozen Pretrained ResNet-18 (ImageNet-1k, 512-dim embeddings)")
    print("=" * 80)
    
    base_dir = Path(__file__).resolve().parent
    dataset_root = base_dir / "data" / "raw" / "metal_nut"
    splits_path = base_dir / "data" / "splits.json"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data Partitions
    print("\n[Step 1/5] Loading isolated data partitions from splits.json...")
    datasets = load_splits_and_build_datasets(dataset_root, splits_path, img_size=224)
    print(f"  • D_train_anomaly : {len(datasets['D_train_anomaly']):>3} normal images (for fitting)")
    print(f"  • D_val_norm      : {len(datasets['D_val_norm']):>3} normal images")
    print(f"  • D_val_calib     : {len(datasets['D_val_calib']):>3} images (14 normal + 14 known defects)")
    print(f"  • D_eval_final    : {len(datasets['D_eval_final']):>3} images (22 normal + 14 known + 23 unseen flip)")
    
    # 2. Extract Features
    print("\n[Step 2/5] Initializing feature extractor and computing embeddings...")
    device = torch.device("cpu")
    extractor = ResNetFeatureExtractor(backbone_name="resnet18", device=device)
    
    t0 = time.perf_counter()
    train_feats, train_labels, _, _ = extract_features_from_dataset(extractor, datasets["D_train_anomaly"])
    val_calib_feats, val_calib_labels, val_calib_cats, _ = extract_features_from_dataset(extractor, datasets["D_val_calib"])
    eval_feats, eval_labels, eval_cats, eval_paths = extract_features_from_dataset(extractor, datasets["D_eval_final"])
    extract_time = time.perf_counter() - t0
    
    print(f"  • Feature Extraction Complete ({extract_time:.2f}s total)")
    print(f"  • Training Embeddings Shape   : {train_feats.shape}")
    print(f"  • Calibration Embeddings Shape: {val_calib_feats.shape}")
    print(f"  • Evaluation Embeddings Shape : {eval_feats.shape}")
    
    # 3. Fit Candidate Distance Models on D_train_anomaly
    print("\n[Step 3/5] Fitting distance-based anomaly models on D_train_anomaly...")
    models = {
        "Mahalanobis": MahalanobisAnomalyDetector(use_ledoit_wolf=True),
        "kNN_k1": KNNAnomalyDetector(n_neighbors=1),
        "kNN_k3": KNNAnomalyDetector(n_neighbors=3),
        "kNN_k5": KNNAnomalyDetector(n_neighbors=5),
    }
    
    # Validation AUC & performance comparison on D_val_calib
    print("\n--- Comparing Candidate Models on Calibration Partition (D_val_calib) ---")
    val_metrics = {}
    for name, model in models.items():
        model.fit(train_feats)
        calib_scores = model.score(val_calib_feats)
        metrics = compute_roc_pr_metrics(val_calib_labels, calib_scores)
        val_metrics[name] = metrics
        print(f"  • {name:<14} -> Validation ROC-AUC: {metrics['roc_auc']:.4f} | Validation PR-AUC: {metrics['pr_auc']:.4f}")
        
    # Select primary model (kNN_k5 or Mahalanobis based on validation AUC)
    primary_model_name = "kNN_k5" if val_metrics["kNN_k5"]["roc_auc"] >= val_metrics["Mahalanobis"]["roc_auc"] else "Mahalanobis"
    primary_model = models[primary_model_name]
    print(f"\n[OK] Selected Primary Baseline Model based on validation: {primary_model_name}")
    
    # 4. Threshold Calibration on D_val_calib (Primary & Sensitivity Analysis)
    print("\n[Step 4/5] Calibrating decision threshold on D_val_calib...")
    primary_calib_scores = primary_model.score(val_calib_feats)
    
    # Primary: Cost-Sensitive (beta = 5.0)
    tau_primary, cost_primary, stats_primary = find_cost_sensitive_threshold(
        val_calib_labels, primary_calib_scores, beta=5.0
    )
    print(f"  • Primary Cost-Sensitive Threshold (beta=5.0) : tau* = {tau_primary:.4f}")
    print(f"    Validation Cost: {cost_primary:.2f} (FN: {stats_primary['fn']}, FP: {stats_primary['fp']})")
    
    # Sensitivity Analysis on beta in [2.0, 5.0, 10.0, 20.0]
    print("\n--- Predefined Sensitivity Analysis on Calibration Set (D_val_calib) ---")
    sensitivity_results = {}
    for beta_val in [2.0, 5.0, 10.0, 20.0]:
        tau_b, cost_b, stats_b = find_cost_sensitive_threshold(val_calib_labels, primary_calib_scores, beta=beta_val)
        sensitivity_results[f"beta_{beta_val}"] = {
            "beta": beta_val,
            "threshold": tau_b,
            "cost": cost_b,
            "validation_fn": stats_b["fn"],
            "validation_fp": stats_b["fp"]
        }
        print(f"  • beta = {beta_val:>4.1f} -> tau* = {tau_b:.4f} | Val FN = {stats_b['fn']} | Val FP = {stats_b['fp']}")
        
    # Secondary Validation Criteria (Youden J & F1)
    tau_youden, j_val = find_youden_threshold(val_calib_labels, primary_calib_scores)
    tau_f1, f1_val = find_f1_threshold(val_calib_labels, primary_calib_scores)
    print(f"  • Secondary Criterion (Youden J) : tau* = {tau_youden:.4f} (Val J = {j_val:.4f})")
    print(f"  • Secondary Criterion (F1-Max)   : tau* = {tau_f1:.4f} (Val F1 = {f1_val:.4f})")
    
    # 5. Final Evaluation on Untouched D_eval_final (59 images)
    print("\n[Step 5/5] Evaluating Primary Baseline on Untouched Final Evaluation Set (D_eval_final)...")
    eval_scores = primary_model.score(eval_feats)
    
    # Compute Full ROC & PR curves on final evaluation set
    eval_roc_pr = compute_roc_pr_metrics(eval_labels, eval_scores)
    
    # Evaluate primary locked threshold
    final_perf = evaluate_threshold_performance(eval_labels, eval_scores, threshold=tau_primary)
    
    # Per-category score breakdown
    cat_scores = {}
    unique_cats = sorted(list(set(eval_cats)))
    for c in unique_cats:
        mask = np.array([x == c for x in eval_cats])
        cat_scores[c] = eval_scores[mask]
        
    # Profile single-image inference latency
    single_img_tensor = datasets["D_eval_final"][0]["image"].unsqueeze(0)
    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        with torch.no_grad():
            feat = extractor(single_img_tensor).cpu().numpy()
            _ = primary_model.score(feat)
        latencies.append((time.perf_counter() - t_start) * 1000.0) # ms
        
    latency_mean = float(np.mean(latencies))
    latency_std = float(np.std(latencies))
    
    # 6. Generate Figures & Artifacts
    print("\n--- Generating Artifacts and Visualizations ---")
    roc_pr_fig_path = artifacts_dir / "baseline_roc_pr_curve.png"
    plot_roc_pr_curves(
        eval_roc_pr,
        str(roc_pr_fig_path),
        title_suffix=f"Baseline {primary_model_name} (ResNet-18)"
    )
    print(f"  • Saved ROC/PR Curves to: {roc_pr_fig_path}")
    
    dist_fig_path = artifacts_dir / "baseline_score_distributions.png"
    plot_score_distributions_by_category(
        cat_scores,
        threshold=tau_primary,
        save_path=str(dist_fig_path),
        title=f"Baseline Anomaly Scores Across Categories ({primary_model_name})"
    )
    print(f"  • Saved Score Distributions to: {dist_fig_path}")
    
    cm = np.array([
        [final_perf["tn"], final_perf["fp"]],
        [final_perf["fn"], final_perf["tp"]]
    ])
    cm_fig_path = artifacts_dir / "baseline_confusion_matrix.png"
    plot_confusion_matrix_figure(
        cm,
        str(cm_fig_path),
        title=f"Baseline Confusion Matrix at tau* = {tau_primary:.3f}"
    )
    print(f"  • Saved Confusion Matrix to: {cm_fig_path}")
    
    # 7. Save Metrics JSON
    results_summary = {
        "milestone": "M1 - Baseline Anomaly Detection",
        "model_architecture": {
            "backbone": "ResNet-18 (Pretrained ImageNet-1k, Frozen)",
            "embedding_dim": 512,
            "detector_type": primary_model_name
        },
        "timing_and_profiling": {
            "single_image_latency_mean_ms": latency_mean,
            "single_image_latency_std_ms": latency_std,
            "device": "CPU"
        },
        "calibration_results": {
            "dataset_size": len(val_calib_labels),
            "primary_beta": 5.0,
            "primary_calibrated_threshold": tau_primary,
            "sensitivity_analysis": sensitivity_results,
            "secondary_youden_threshold": tau_youden,
            "secondary_f1_threshold": tau_f1
        },
        "final_evaluation_metrics": {
            "dataset_size": len(eval_labels),
            "normal_count": int(np.sum(eval_labels == 0)),
            "defect_count": int(np.sum(eval_labels == 1)),
            "image_level_roc_auc": eval_roc_pr["roc_auc"],
            "image_level_pr_auc": eval_roc_pr["pr_auc"],
            "locked_threshold_performance": final_perf,
            "per_category_scores": {
                c: {
                    "count": len(scores),
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "min_score": float(np.min(scores)),
                    "max_score": float(np.max(scores)),
                    "flagged_as_anomaly_rate": float(np.mean(scores >= tau_primary))
                }
                for c, scores in cat_scores.items()
            }
        }
    }
    
    metrics_json_path = artifacts_dir / "baseline_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  • Saved Full Metrics JSON to: {metrics_json_path}")
    
    # Print Final Summary Table
    print("\n" + "=" * 80)
    print("MILESTONE 1 BASELINE EMPIRICAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Primary Detector Model       : {primary_model_name} (ResNet-18 GAP 512-dim)")
    print(f"Untouched Evaluation Samples : {len(eval_labels)} (22 Normal, 37 Defective)")
    print("-" * 80)
    print(f"Image-Level ROC-AUC          : {eval_roc_pr['roc_auc']:.4f}")
    print(f"Image-Level PR-AUC           : {eval_roc_pr['pr_auc']:.4f}")
    print("-" * 80)
    print(f"Locked Threshold (tau*)      : {tau_primary:.4f} (Calibrated on D_val_calib with beta=5.0)")
    print(f"True Positives (TP)          : {final_perf['tp']:>2} / 37")
    print(f"True Negatives (TN)          : {final_perf['tn']:>2} / 22")
    print(f"False Positives (FP)         : {final_perf['fp']:>2} / 22 (False Alarm Rate: {final_perf['fpr']*100:.2f}%)")
    print(f"False Negatives (FN)         : {final_perf['fn']:>2} / 37 (Escape Rate / FNR: {final_perf['fnr']*100:.2f}%)")
    print(f"Precision                    : {final_perf['precision']:.4f}")
    print(f"Recall (Sensitivity)         : {final_perf['recall']:.4f}")
    print(f"F1-Score                     : {final_perf['f1_score']:.4f}")
    print(f"Overall Accuracy             : {final_perf['accuracy']*100:.2f}%")
    print("-" * 80)
    print("Per-Category Anomaly Detection Breakdown:")
    for c, info in results_summary["final_evaluation_metrics"]["per_category_scores"].items():
        print(f"  • {c:<16}: Mean Score = {info['mean_score']:>7.3f} | Flagged Anomaly Rate = {info['flagged_as_anomaly_rate']*100:>5.1f}% ({info['count']} samples)")
    print("-" * 80)
    print(f"Inference Latency on CPU     : {latency_mean:.2f} +/- {latency_std:.2f} ms per image")
    print("=" * 80)
    return results_summary

if __name__ == "__main__":
    run_experiment()
