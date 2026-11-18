"""
train_gradient_boosting.py - Trains and validates Teammate 3's Gradient Boosting model.
Performs Group-Aware 5-Fold Cross-Validation on the 80 training trips,
fits the full model, evaluates on student-testing trips, and exports model artifacts.
"""

from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gradient_boosting import PathSnapGradientBoosting, FEATURE_COLUMNS
from src.models.evaluation import (
    compute_candidate_metrics,
    compute_top1_observation_accuracy,
)


def run_training():
    train_path = PROJECT_ROOT / "data" / "processed" / "train_features.csv.gz"
    test_path = PROJECT_ROOT / "data" / "processed" / "student_test_features.csv.gz"

    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Student-test features not found at {test_path}")

    print("[*] Loading processed training data...")
    t0 = time.time()
    train_df = pd.read_csv(train_path)
    print(f"[+] Loaded {len(train_df):,} training samples from {train_df['trip_id'].nunique()} trips ({time.time() - t0:.2f}s)")

    X = train_df[FEATURE_COLUMNS]
    y = train_df["is_true_arc"].values
    groups = train_df["trip_id"].values

    # 1. Group-Aware 5-Fold Cross-Validation
    print("\n" + "=" * 60)
    print("[*] Running Group-Aware 5-Fold Cross-Validation (by trip_id)...")
    print("=" * 60)

    gkf = GroupKFold(n_splits=5)
    cv_records = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        fold_start = time.time()
        X_train, y_train = train_df.iloc[train_idx], y[train_idx]
        X_val, y_val = train_df.iloc[val_idx], y[val_idx]

        model = PathSnapGradientBoosting(
            learning_rate=0.08,
            max_iter=120,
            max_leaf_nodes=45,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42 + fold,
        )
        model.fit(X_train, y_train)

        # Validation predictions
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        metrics = compute_candidate_metrics(y_val, val_preds, val_probs)

        # Observation-level Top-1 accuracy
        val_eval_df = X_val[["trip_id", "obs_id", "is_true_arc"]].copy()
        val_eval_df["score"] = val_probs
        top1_acc, miss_rate = compute_top1_observation_accuracy(val_eval_df)

        fold_time = time.time() - fold_start
        record = {
            "fold": fold,
            "val_trips": int(X_val["trip_id"].nunique()),
            "val_samples": len(X_val),
            "macro_f1": metrics["f1_macro"],
            "binary_f1": metrics["f1_binary"],
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "brier_score": metrics["brier_score"],
            "top1_acc": top1_acc,
            "cand_miss_rate": miss_rate,
            "fold_time_s": fold_time,
        }
        cv_records.append(record)
        print(f"  [Fold {fold}] Macro F1: {metrics['f1_macro']:.4f} | Top-1 Acc: {top1_acc:.2f}% | LogLoss: {metrics['log_loss']:.4f} | Time: {fold_time:.1f}s")

    cv_df = pd.DataFrame(cv_records)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(reports_dir / "gradient_boosting_cv_results.csv", index=False)

    print("-" * 60)
    print(f"[+] CV Average Macro F1: {cv_df['macro_f1'].mean():.4f} +/- {cv_df['macro_f1'].std():.4f}")
    print(f"[+] CV Average Top-1 Acc: {cv_df['top1_acc'].mean():.2f}% +/- {cv_df['top1_acc'].std():.2f}%")
    print(f"[+] CV Average Log Loss: {cv_df['log_loss'].mean():.4f}")

    # 2. Fit Full Model on all 80 Training Trips
    print("\n" + "=" * 60)
    print("[*] Fitting final model on all 80 training trips...")
    print("=" * 60)
    full_start = time.time()
    final_model = PathSnapGradientBoosting(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=45,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(train_df, y)
    full_train_time = time.time() - full_start
    print(f"[+] Full model fitted in {full_train_time:.2f}s")

    model_dir = PROJECT_ROOT / "models"
    final_model.save(model_dir / "gradient_boosting.joblib")

    # 3. Evaluate on 10 Student-Test Trips
    print("\n" + "=" * 60)
    print("[*] Evaluating on 10 Student-Testing trips...")
    print("=" * 60)
    test_df = pd.read_csv(test_path)
    inf_start = time.time()
    test_probs = final_model.predict_proba(test_df)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    inf_time = time.time() - inf_start

    test_metrics = compute_candidate_metrics(test_df["is_true_arc"].values, test_preds, test_probs)

    test_eval_df = test_df[["trip_id", "obs_id", "is_true_arc"]].copy()
    test_eval_df["score"] = test_probs
    test_top1_acc, test_miss_rate = compute_top1_observation_accuracy(test_eval_df)

    print(f"  Student-Test Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"  Student-Test Binary F1: {test_metrics['f1_binary']:.4f}")
    print(f"  Student-Test Candidate Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Student-Test Observation Top-1 Accuracy: {test_top1_acc:.2f}%")
    print(f"  Student-Test Candidate Miss Rate: {test_miss_rate:.2f}%")
    print(f"  Student-Test Log Loss: {test_metrics['log_loss']:.4f}")
    print(f"  Student-Test Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"  Inference Time for {len(test_df):,} candidates: {inf_time:.3f}s ({inf_time / 10.0 * 1000:.1f}ms/trip)")

    summary = {
        "model_name": "Gradient Boosting (HistGradientBoosting)",
        "algorithm_family": "Boosting",
        "key_configuration": final_model.params,
        "cv_macro_f1_mean": float(cv_df["macro_f1"].mean()),
        "cv_macro_f1_std": float(cv_df["macro_f1"].std()),
        "cv_top1_acc_mean": float(cv_df["top1_acc"].mean()),
        "cv_top1_acc_std": float(cv_df["top1_acc"].std()),
        "student_test_macro_f1": float(test_metrics["f1_macro"]),
        "student_test_binary_f1": float(test_metrics["f1_binary"]),
        "student_test_top1_accuracy": float(test_top1_acc),
        "student_test_candidate_miss_rate": float(test_miss_rate),
        "student_test_log_loss": float(test_metrics["log_loss"]),
        "student_test_brier_score": float(test_metrics["brier_score"]),
        "full_training_time_s": float(full_train_time),
        "student_test_inference_time_s": float(inf_time),
    }

    with open(reports_dir / "gradient_boosting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Full metrics saved to {reports_dir / 'gradient_boosting_summary.json'}")
    return summary


if __name__ == "__main__":
    run_training()
