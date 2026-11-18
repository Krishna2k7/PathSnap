"""
PathSnap Logistic Regression Training & Evaluation Pipeline
===========================================================
Algorithm Family: Linear / Probabilistic
Model: Logistic Regression with L2 Regularization (scikit-learn)

Implements:
1. Shared data loading from precomputed processed_data/ parquet files.
2. 5-Fold Group-Aware Cross-Validation (cv_fold 0 to 4):
   - Strict trip-level isolation.
   - Per-fold evaluation of Log Loss, Macro F1, Observation-Level Top-1 Accuracy.
   - Note: lbfgs is a quasi-Newton batch optimization method without epoch histories;
     in accordance with guidelines, no artificial iteration loss curves are generated.
3. Full Model Training on all 80 training trips:
   - Export model weights to models/logistic_regression/logreg_model.joblib.
   - Export configuration to models/logistic_regression/config.json.
4. Evaluation on datasets/student_test (10 development-stage trips):
   - Candidate row-level Precision, Recall, Macro F1, and Binary Log Loss.
   - Observation-Level Top-1 Accuracy and Top-1 Macro F1.
   - Route Continuity across all 10 student-test trips using raw road-network arcs.
5. Visualizations:
   - visualizations/logreg_confusion_matrix.png (candidate confusion matrix heatmap)
   - visualizations/logreg_coefficients.png (standardized feature coefficients bar chart)
6. Metrics logging to results/logistic_regression_metrics.json.
"""

import os
import sys
import json
import time
import glob
import tempfile
import warnings
from typing import Dict, List, Tuple, Any

# Ensure matplotlib uses writable temporary directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_pathsnap")
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def load_datasets(
    feature_meta_path: str = "processed_data/feature_columns.json",
    train_parquet_path: str = "processed_data/train_features.parquet",
    test_parquet_path: str = "processed_data/student_test_features.parquet",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads precomputed parquet datasets and feature columns metadata.
    """
    print("[Data] Loading feature metadata and standardized datasets...")
    with open(feature_meta_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]

    train_df = pd.read_parquet(train_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)

    print(f"[Data] Loaded Train: {len(train_df):,} rows ({train_df['trip_id'].nunique()} trips)")
    print(f"[Data] Loaded Student Test: {len(test_df):,} rows ({test_df['trip_id'].nunique()} trips)")
    print(f"[Data] Number of features: {len(feature_cols)}")
    return train_df, test_df, feature_cols


def compute_observation_top1_accuracy(
    df: pd.DataFrame,
    pred_probs: np.ndarray
) -> Tuple[float, float, pd.DataFrame]:
    """
    Groups predictions by (trip_id, obs_idx), picks the candidate arc with
    maximum predicted probability, and calculates Top-1 accuracy against ground truth.
    """
    df_eval = df[["trip_id", "obs_idx", "candidate_arc_id", "label"]].copy().reset_index(drop=True)
    df_eval["pred_prob"] = pred_probs

    # Find candidate with max predicted probability per observation
    idx_max = df_eval.groupby(["trip_id", "obs_idx"])["pred_prob"].idxmax()
    top1_df = df_eval.loc[idx_max].copy()
    top1_df = top1_df.sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    # Correct if the top predicted candidate has label == 1
    top1_accuracy = float(top1_df["label"].mean() * 100.0)

    # Compute candidate-level macro F1 where top-1 candidate = 1, others = 0
    is_top1_mask = np.zeros(len(df_eval), dtype=int)
    is_top1_mask[idx_max.values] = 1
    top1_macro_f1 = float(f1_score(df_eval["label"], is_top1_mask, average="macro"))

    return top1_accuracy, top1_macro_f1, top1_df


def compute_route_continuity(
    top1_df: pd.DataFrame,
    student_test_dir: str = "datasets/student_test"
) -> Tuple[float, Dict[str, float]]:
    """
    Calculates route continuity for each student-test trip:
    The percentage of consecutive Top-1 predicted road segments that are either:
    1. Identical: arc_t == arc_{t+1} (vehicle stayed on same segment)
    2. Directly connected in the road network: to_node[arc_t] == from_node[arc_{t+1}]
    """
    per_trip_continuity = {}

    for trip_id, trip_group in top1_df.groupby("trip_id"):
        trip_group = trip_group.sort_values("obs_idx")
        pred_arcs = trip_group["candidate_arc_id"].values

        arcs_file = os.path.join(student_test_dir, str(trip_id), f"{trip_id}.arcs")
        if not os.path.exists(arcs_file):
            print(f"[Warning] Arcs file not found for trip {trip_id}: {arcs_file}")
            continue

        arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
        u_nodes = arcs_df["u"].values
        v_nodes = arcs_df["v"].values

        if len(pred_arcs) <= 1:
            per_trip_continuity[trip_id] = 100.0
            continue

        a_curr = pred_arcs[:-1]
        a_next = pred_arcs[1:]

        same_arc = (a_curr == a_next)
        connected_junction = (v_nodes[a_curr] == u_nodes[a_next])
        valid_transitions = same_arc | connected_junction

        trip_cont_pct = float(np.mean(valid_transitions) * 100.0)
        per_trip_continuity[trip_id] = round(trip_cont_pct, 2)

    mean_continuity = float(np.mean(list(per_trip_continuity.values()))) if per_trip_continuity else 0.0
    return round(mean_continuity, 2), per_trip_continuity


def run_cross_validation(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    model_params: Dict[str, Any],
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Executes 5-fold group-aware cross-validation across cv_fold 0 to 4.
    """
    print(f"\n[CV] Starting {n_splits}-Fold Group-Aware Cross-Validation (Logistic Regression)...")
    fold_results = []
    start_cv_time = time.time()

    for fold in range(n_splits):
        fold_start = time.time()
        train_mask = (train_df["cv_fold"] != fold)
        val_mask = (train_df["cv_fold"] == fold)

        X_train_fold = train_df.loc[train_mask, feature_cols].values
        y_train_fold = train_df.loc[train_mask, "label"].values

        X_val_fold = train_df.loc[val_mask, feature_cols].values
        y_val_fold = train_df.loc[val_mask, "label"].values
        val_sub_df = train_df.loc[val_mask]

        fold_model = LogisticRegression(**model_params)
        fold_model.fit(X_train_fold, y_train_fold)

        val_probs = fold_model.predict_proba(X_val_fold)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        val_logloss = float(log_loss(y_val_fold, val_probs))
        val_macro_f1 = float(f1_score(y_val_fold, val_preds, average="macro"))
        val_precision = float(precision_score(y_val_fold, val_preds, zero_division=0))
        val_recall = float(recall_score(y_val_fold, val_preds, zero_division=0))
        val_acc = float(accuracy_score(y_val_fold, val_preds))

        val_top1_acc, val_top1_f1, _ = compute_observation_top1_accuracy(val_sub_df, val_probs)
        fold_time = time.time() - fold_start

        fold_record = {
            "fold": fold,
            "val_trips": int(val_sub_df["trip_id"].nunique()),
            "val_rows": len(val_sub_df),
            "log_loss": round(val_logloss, 5),
            "macro_f1": round(val_macro_f1, 4),
            "top1_accuracy": round(val_top1_acc, 2),
            "top1_macro_f1": round(val_top1_f1, 4),
            "precision": round(val_precision, 4),
            "recall": round(val_recall, 4),
            "accuracy": round(val_acc, 4),
            "iterations_to_converge": int(fold_model.n_iter_[0]),
            "training_time_sec": round(fold_time, 2),
        }
        fold_results.append(fold_record)

        print(
            f"  -> Fold {fold}: LogLoss = {val_logloss:.5f} | "
            f"Macro F1 = {val_macro_f1:.4f} | "
            f"Top-1 Acc = {val_top1_acc:.2f}% | "
            f"Iters = {fold_model.n_iter_[0]} | "
            f"Time = {fold_time:.2f}s"
        )

    total_cv_time = time.time() - start_cv_time

    macro_f1s = [f["macro_f1"] for f in fold_results]
    log_losses = [f["log_loss"] for f in fold_results]
    top1_accs = [f["top1_accuracy"] for f in fold_results]
    top1_f1s = [f["top1_macro_f1"] for f in fold_results]

    cv_summary = {
        "folds": fold_results,
        "mean_macro_f1": round(float(np.mean(macro_f1s)), 4),
        "std_macro_f1": round(float(np.std(macro_f1s)), 4),
        "mean_log_loss": round(float(np.mean(log_losses)), 5),
        "std_log_loss": round(float(np.std(log_losses)), 5),
        "mean_top1_accuracy": round(float(np.mean(top1_accs)), 2),
        "std_top1_accuracy": round(float(np.std(top1_accs)), 2),
        "mean_top1_macro_f1": round(float(np.mean(top1_f1s)), 4),
        "total_cv_time_sec": round(total_cv_time, 2),
    }

    print(
        f"[CV Completed] 5-Fold Mean Macro F1: {cv_summary['mean_macro_f1']:.4f} (+/- {cv_summary['std_macro_f1']:.4f}) | "
        f"Mean LogLoss: {cv_summary['mean_log_loss']:.5f} | "
        f"Mean Top-1: {cv_summary['mean_top1_accuracy']:.2f}%"
    )
    return cv_summary


def train_full_model_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_params: Dict[str, Any],
    output_model_dir: str = "models/logistic_regression"
) -> Tuple[LogisticRegression, Dict[str, Any], Dict[str, Any]]:
    """
    Fits Logistic Regression on all 80 training trips, saves model weights and config,
    and evaluates on datasets/student_test.
    """
    os.makedirs(output_model_dir, exist_ok=True)

    print("\n[Full Fit] Training Logistic Regression on all 80 training trips (100% training data)...")
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    final_model = LogisticRegression(**model_params)
    t0_train = time.time()
    final_model.fit(X_train, y_train)
    full_train_time = time.time() - t0_train
    print(f"[Full Fit] Model successfully trained in {full_train_time:.2f} seconds (converged in {final_model.n_iter_[0]} iterations).")

    # Save model weights & config
    model_save_path = os.path.join(output_model_dir, "logreg_model.joblib")
    config_save_path = os.path.join(output_model_dir, "config.json")
    joblib.dump(final_model, model_save_path)

    config_dict = {
        "model_type": "Logistic Regression Classifier",
        "algorithm_family": "Linear and probabilistic",
        "hyperparameters": {
            "penalty": model_params["penalty"],
            "C": model_params["C"],
            "solver": model_params["solver"],
            "max_iter": model_params["max_iter"],
            "random_state": model_params["random_state"],
        },
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "intercept": float(final_model.intercept_[0]),
        "coefficients": {col: float(coef) for col, coef in zip(feature_cols, final_model.coef_[0])},
        "iterations_converged": int(final_model.n_iter_[0]),
        "saved_at": pd.Timestamp.now().isoformat(),
        "model_path": model_save_path,
    }
    with open(config_save_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[Artifacts] Saved model to {model_save_path} and config to {config_save_path}")

    # Evaluate on Student Test
    print("[Student Test] Evaluating final model on student_test partition (10 trips)...")
    t0_inf = time.time()
    test_probs = final_model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - t0_inf
    test_preds = (test_probs >= 0.5).astype(int)

    # 1. Candidate Row-level Metrics
    test_logloss = float(log_loss(y_test, test_probs))
    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_binary_f1 = float(f1_score(y_test, test_preds, average="binary"))
    test_accuracy = float(accuracy_score(y_test, test_preds))

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # 2. Observation-Level Top-1 Accuracy
    top1_acc, top1_macro_f1, top1_df = compute_observation_top1_accuracy(test_df, test_probs)

    # 3. Route Continuity across Student-Test Trips
    mean_continuity, per_trip_continuity = compute_route_continuity(top1_df)

    student_test_metrics = {
        "candidate_metrics": {
            "log_loss": round(test_logloss, 5),
            "macro_f1": round(test_macro_f1, 4),
            "binary_f1": round(test_binary_f1, 4),
            "precision": round(test_precision, 4),
            "recall": round(test_recall, 4),
            "accuracy": round(test_accuracy, 4),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
        },
        "observation_top1_metrics": {
            "top1_accuracy_pct": round(top1_acc, 2),
            "top1_macro_f1": round(top1_macro_f1, 4),
            "total_observations": len(top1_df),
            "correct_observations": int(top1_df["label"].sum()),
        },
        "route_continuity": {
            "mean_route_continuity_pct": round(mean_continuity, 2),
            "per_trip_continuity_pct": per_trip_continuity,
        },
        "timing": {
            "full_training_time_sec": round(full_train_time, 2),
            "student_test_inference_time_sec": round(inference_time, 3),
            "inference_time_per_observation_ms": round((inference_time / len(top1_df)) * 1000.0, 4),
        },
    }

    # Diagnostics: Train vs Student-Test comparison
    train_probs_sample = final_model.predict_proba(X_train[:50000])[:, 1]
    train_logloss_sample = float(log_loss(y_train[:50000], train_probs_sample))
    train_macro_f1_sample = float(f1_score(y_train[:50000], (train_probs_sample >= 0.5).astype(int), average="macro"))

    diagnostics = {
        "train_sample_logloss": round(train_logloss_sample, 5),
        "train_sample_macro_f1": round(train_macro_f1_sample, 4),
        "test_logloss": round(test_logloss, 5),
        "test_macro_f1": round(test_macro_f1, 4),
        "generalization_gap_logloss": round(test_logloss - train_logloss_sample, 5),
        "generalization_gap_f1": round(train_macro_f1_sample - test_macro_f1, 4),
    }

    return final_model, student_test_metrics, diagnostics


def generate_visualizations(
    final_model: LogisticRegression,
    feature_cols: List[str],
    student_test_metrics: Dict[str, Any],
    output_dir: str = "visualizations"
):
    """
    Generates and saves:
    1. visualizations/logreg_confusion_matrix.png (confusion matrix heatmap)
    2. visualizations/logreg_coefficients.png (standardized feature weights bar chart)
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Confusion Matrix Heatmap
    cm_dict = student_test_metrics["candidate_metrics"]["confusion_matrix"]
    cm = np.array([
        [cm_dict["true_negative"], cm_dict["false_positive"]],
        [cm_dict["false_negative"], cm_dict["true_positive"]]
    ])
    total = np.sum(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Negative (0)", "Positive (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Student-Test Candidate Confusion Matrix (Logistic Regression)",
        ylabel="True Reference Arc",
        xlabel="Predicted Arc Candidate"
    )

    thresh = cm.max() / 2.0
    labels_text = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"]
    ]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100.0
            desc = labels_text[i][j]
            ax.text(
                j, i,
                f"{desc}\n{val:,}\n({pct:.2f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "logreg_confusion_matrix.png")
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"[Visualization] Saved {cm_path}")

    # 2. Standardized Feature Coefficients Bar Chart
    coefs = final_model.coef_[0]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("coefficient", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_df["coefficient"]]
    bars = ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="black", height=0.65)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_title("Standardized Feature Coefficients (Logistic Regression L2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Learned Coefficient Weight (Log-Odds Impact)", fontsize=11)
    ax.set_ylabel("Standardized Feature", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Annotate values
    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = 0.03 if width >= 0 else -0.03
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.3f}",
            ha=ha, va="center", fontsize=8.5, fontweight="bold"
        )

    # Adjust xlim for text padding
    min_x, max_x = coef_df["coefficient"].min(), coef_df["coefficient"].max()
    ax.set_xlim(min_x - 0.25, max_x + 0.25)

    plt.tight_layout()
    coef_path = os.path.join(output_dir, "logreg_coefficients.png")
    fig.savefig(coef_path, dpi=200)
    plt.close(fig)
    print(f"[Visualization] Saved {coef_path}")


def run_logreg_pipeline():
    """
    Main execution pipeline for Logistic Regression training, CV, testing, and metrics saving.
    """
    overall_start = time.time()
    print("=" * 75)
    print("      PATHSNAP LOGISTIC REGRESSION (L2) TRAINING & EVALUATION")
    print("=" * 75)

    # 1. Model Configuration
    model_params = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 500,
        "random_state": 42,
    }

    # 2. Load Datasets
    train_df, test_df, feature_cols = load_datasets()

    # 3. 5-Fold Group-Aware CV
    cv_summary = run_cross_validation(
        train_df, feature_cols, model_params, n_splits=5
    )

    # 4. Fit Full Model & Evaluate on Student Test
    final_model, student_test_metrics, diagnostics = train_full_model_and_evaluate(
        train_df, test_df, feature_cols, model_params
    )

    # 5. Visualizations
    generate_visualizations(
        final_model, feature_cols, student_test_metrics
    )

    # 6. Save Consolidated Metrics JSON
    total_time = time.time() - overall_start

    complete_metrics = {
        "model_name": "Logistic Regression (L2 Regularized)",
        "algorithm_family": "Linear and probabilistic",
        "timestamp": pd.Timestamp.now().isoformat(),
        "hyperparameters": model_params,
        "feature_coefficients": {col: round(float(c), 5) for col, c in zip(feature_cols, final_model.coef_[0])},
        "intercept": round(float(final_model.intercept_[0]), 5),
        "cross_validation_5fold": cv_summary,
        "student_test_evaluation": student_test_metrics,
        "generalization_diagnostics": diagnostics,
        "total_pipeline_time_sec": round(total_time, 2),
    }

    os.makedirs("results", exist_ok=True)
    results_path = "results/logistic_regression_metrics.json"
    with open(results_path, "w") as f:
        json.dump(complete_metrics, f, indent=2)
    print(f"\n[Results] Metrics summary successfully saved to {results_path}")

    # 7. Print Consolidated Summary Report
    print("\n" + "=" * 75)
    print(" LOGISTIC REGRESSION (L2) PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f"Total Execution Time:        {total_time:.2f} s")
    print(f"5-Fold CV Training Time:     {cv_summary['total_cv_time_sec']:.2f} s")
    print(f"Full Model Training Time:    {student_test_metrics['timing']['full_training_time_sec']:.2f} s")
    print(f"Student-Test Inference Time: {student_test_metrics['timing']['student_test_inference_time_sec']:.3f} s "
          f"({student_test_metrics['timing']['inference_time_per_observation_ms']:.4f} ms/obs)")
    print("-" * 75)
    print("CROSS-VALIDATION RESULTS (5-Fold GroupKFold on 80 Training Trips):")
    print(f"  CV Macro F1-score:         {cv_summary['mean_macro_f1']:.4f} +/- {cv_summary['std_macro_f1']:.4f}")
    print(f"  CV Binary Log Loss:        {cv_summary['mean_log_loss']:.5f} +/- {cv_summary['std_log_loss']:.5f}")
    print(f"  CV Top-1 Accuracy:         {cv_summary['mean_top1_accuracy']:.2f}% +/- {cv_summary['std_top1_accuracy']:.2f}%")
    print(f"  CV Top-1 Macro F1:         {cv_summary['mean_top1_macro_f1']:.4f}")
    print("-" * 75)
    print("STUDENT-TEST EVALUATION RESULTS (10 Development-Stage Trips):")
    cand_m = student_test_metrics["candidate_metrics"]
    top1_m = student_test_metrics["observation_top1_metrics"]
    cont_m = student_test_metrics["route_continuity"]
    print(f"  Candidate Top-1 Accuracy:  {top1_m['top1_accuracy_pct']:.2f}% ({top1_m['correct_observations']:,} / {top1_m['total_observations']:,} GPS obs)")
    print(f"  Candidate Top-1 Macro F1:  {top1_m['top1_macro_f1']:.4f}")
    print(f"  Candidate Macro F1-score:  {cand_m['macro_f1']:.4f}")
    print(f"  Candidate Precision:       {cand_m['precision']:.4f}")
    print(f"  Candidate Recall:          {cand_m['recall']:.4f}")
    print(f"  Candidate Binary Log Loss: {cand_m['log_loss']:.5f}")
    print(f"  Route Continuity:          {cont_m['mean_route_continuity_pct']:.2f}%")
    print("-" * 75)
    print("PER-TRIP ROUTE CONTINUITY (STUDENT TEST):")
    for tid, val in sorted(cont_m["per_trip_continuity_pct"].items()):
        print(f"  Trip {tid}: {val:.2f}%")
    print("-" * 75)
    print("LEARNED FEATURE COEFFICIENTS (TOP IMPACT):")
    sorted_coefs = sorted(complete_metrics["feature_coefficients"].items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, weight in sorted_coefs[:8]:
        direction = "Positive (Increases True Arc Odds)" if weight > 0 else "Negative (Decreases True Arc Odds)"
        print(f"  {feat:<25} : {weight:+.4f} | {direction}")
    print("-" * 75)
    print("GENERALIZATION & OVERFITTING DIAGNOSTICS:")
    print(f"  Train Sample Log Loss:     {diagnostics['train_sample_logloss']:.5f}")
    print(f"  Student Test Log Loss:     {diagnostics['test_logloss']:.5f}")
    print(f"  Log Loss Gap:              {diagnostics['generalization_gap_logloss']:+.5f}")
    print(f"  Train Sample Macro F1:     {diagnostics['train_sample_macro_f1']:.4f}")
    print(f"  Student Test Macro F1:     {diagnostics['test_macro_f1']:.4f}")
    print(f"  Macro F1 Gap:              {diagnostics['generalization_gap_f1']:+.4f}")
    print("=" * 75)


if __name__ == "__main__":
    run_logreg_pipeline()
