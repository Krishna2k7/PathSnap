"""
PathSnap Bayesian Hyperparameter Optimization (Optuna) for LightGBM
===================================================================
Conducts Bayesian optimization using Tree-structured Parzen Estimators (TPE)
to optimize the 5-fold Group-Aware Cross-Validation Macro F1-score of the
LightGBM classifier on datasets/train without data leakage.

Implements:
1. TPESampler search over justified hyperparameter distributions (25 trials).
2. Optimization of 5-fold GroupKFold Macro F1 using early stopping.
3. Export of optimization history to results/optuna_trials.csv and results/bayesian_opt_summary.json.
4. Refitting the winning model on 100% of all 80 training trips:
   - Export to models/gradient_boosting/tuned_lgbm_model.joblib
   - Export to models/gradient_boosting/tuned_config.json
5. Systematic comparison: Untuned vs. Tuned LightGBM on CV and Student-Test metrics.
6. Publication plots:
   - visualizations/optuna_optimization_history.png
   - visualizations/optuna_param_importance.png
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
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import lightgbm as lgb
from sklearn.metrics import (
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_datasets():
    print("[Data] Loading feature metadata and preprocessed datasets...")
    with open("processed_data/feature_columns.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]

    train_df = pd.read_parquet("processed_data/train_features.parquet")
    test_df = pd.read_parquet("processed_data/student_test_features.parquet")

    print(f"[Data] Train: {len(train_df):,} rows ({train_df['trip_id'].nunique()} trips)")
    print(f"[Data] Student Test: {len(test_df):,} rows ({test_df['trip_id'].nunique()} trips)")
    return train_df, test_df, feature_cols


def compute_observation_top1_accuracy(df: pd.DataFrame, pred_probs: np.ndarray) -> Tuple[float, float, pd.DataFrame]:
    df_eval = df[["trip_id", "obs_idx", "candidate_arc_id", "label"]].copy().reset_index(drop=True)
    df_eval["pred_prob"] = pred_probs

    idx_max = df_eval.groupby(["trip_id", "obs_idx"])["pred_prob"].idxmax()
    top1_df = df_eval.loc[idx_max].copy().sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    top1_accuracy = float(top1_df["label"].mean() * 100.0)

    is_top1_mask = np.zeros(len(df_eval), dtype=int)
    is_top1_mask[idx_max.values] = 1
    top1_macro_f1 = float(f1_score(df_eval["label"], is_top1_mask, average="macro"))

    return top1_accuracy, top1_macro_f1, top1_df


def compute_route_continuity(top1_df: pd.DataFrame, student_test_dir: str = "datasets/student_test") -> Tuple[float, Dict[str, float]]:
    per_trip_continuity = {}
    for trip_id, trip_group in top1_df.groupby("trip_id"):
        trip_group = trip_group.sort_values("obs_idx")
        pred_arcs = trip_group["candidate_arc_id"].values

        arcs_file = os.path.join(student_test_dir, str(trip_id), f"{trip_id}.arcs")
        if not os.path.exists(arcs_file):
            continue

        arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
        u_nodes = arcs_df["u"].values
        v_nodes = arcs_df["v"].values

        if len(pred_arcs) <= 1:
            per_trip_continuity[trip_id] = 100.0
            continue

        a_curr = pred_arcs[:-1]
        a_next = pred_arcs[1:]
        valid_transitions = (a_curr == a_next) | (v_nodes[a_curr] == u_nodes[a_next])
        per_trip_continuity[trip_id] = round(float(np.mean(valid_transitions) * 100.0), 2)

    mean_cont = float(np.mean(list(per_trip_continuity.values()))) if per_trip_continuity else 0.0
    return round(mean_cont, 2), per_trip_continuity


def create_objective(sample_df: pd.DataFrame, feature_cols: List[str]):
    X = sample_df[feature_cols].values
    y = sample_df["label"].values
    folds = sample_df["cv_fold"].values

    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space
        num_leaves = trial.suggest_int("num_leaves", 15, 127)
        max_depth = trial.suggest_int("max_depth", 4, 14)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        n_estimators = trial.suggest_int("n_estimators", 150, 600)
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": 1 if subsample < 1.0 else 0,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "max_bin": 128,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_f1_scores = []
        fold_loglosses = []

        for f in range(5):
            tr_idx = (folds != f)
            val_idx = (folds == f)

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X[tr_idx],
                y[tr_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )

            val_probs = model.predict_proba(X[val_idx])[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)

            f1 = f1_score(y[val_idx], val_preds, average="macro")
            ll = log_loss(y[val_idx], val_probs)

            fold_f1_scores.append(f1)
            fold_loglosses.append(ll)

            # Report intermediate score for pruning
            trial.report(f1, f)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_macro_f1 = float(np.mean(fold_f1_scores))
        mean_log_loss = float(np.mean(fold_loglosses))

        # Store attributes
        trial.set_user_attr("mean_log_loss", round(mean_log_loss, 5))
        trial.set_user_attr("mean_macro_f1", round(mean_macro_f1, 4))
        return mean_macro_f1

    return objective


def run_bayesian_optimization(n_trials: int = 25):
    t_start = time.time()
    print("=" * 75)
    print("      BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA) - LIGHTGBM")
    print("=" * 75)

    train_df, test_df, feature_cols = load_datasets()

    # Create stratified, trip-preserving sample for rapid 25-trial exploration (~5-10 min)
    sample_df = train_df.iloc[::4].copy().reset_index(drop=True)
    print(f"[Optuna] Evaluation sample: {len(sample_df):,} rows across all 80 trips in 5 folds.")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lightgbm_tuning",
    )

    obj_func = create_objective(sample_df, feature_cols)

    print(f"\n[Optuna] Launching {n_trials} trials with TPE Sampler...")

    def trial_callback(s, t):
        status = t.state.name
        val_score = t.value if t.value is not None else 0.0
        dur = t.duration.total_seconds() if t.duration else 0.0
        best_so_far = s.best_value if len(s.best_trials) > 0 else val_score
        print(
            f"  [Trial {t.number+1:02d}/{n_trials:02d}] {status:<8} | "
            f"Macro F1 = {val_score:.4f} | "
            f"Best So Far = {best_so_far:.4f} | "
            f"Time = {dur:.1f}s"
        )

    study.optimize(obj_func, n_trials=n_trials, callbacks=[trial_callback])

    trial_records = []
    for t in study.trials:
        dur = t.duration.total_seconds() if t.duration else 0.0
        rec = {
            "trial_id": t.number,
            "status": t.state.name,
            "macro_f1": round(t.value, 4) if t.value is not None else None,
            "log_loss": t.user_attrs.get("mean_log_loss", None),
            "duration_sec": round(dur, 2),
            **t.params,
        }
        trial_records.append(rec)

    optuna_duration = time.time() - t_start
    best_trial = study.best_trial
    best_params = best_trial.params
    print("\n" + "-" * 75)
    print(f"[Optuna Completed in {optuna_duration:.2f}s]")
    print(f"  Best Trial Number: {best_trial.number}")
    print(f"  Best Validation Macro F1: {best_trial.value:.4f}")
    print("  Optimal Hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print("-" * 75)

    # Save trials dataframe
    os.makedirs("results", exist_ok=True)
    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv("results/optuna_trials.csv", index=False)
    print("[Results] Saved results/optuna_trials.csv")

    # -------------------------------------------------------------
    # Step 2: Refit Tuned Model on 100% of Training Data (80 Trips)
    # -------------------------------------------------------------
    print("\n[Refit] Retraining tuned model on all 80 training trips (100% data)...")
    full_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": best_params["num_leaves"],
        "max_depth": best_params["max_depth"],
        "learning_rate": best_params["learning_rate"],
        "n_estimators": best_params["n_estimators"],
        "min_child_samples": best_params["min_child_samples"],
        "subsample": best_params["subsample"],
        "subsample_freq": 1 if best_params["subsample"] < 1.0 else 0,
        "colsample_bytree": best_params["colsample_bytree"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "max_bin": 128,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Evaluate tuned model on full 5-fold CV
    print("[Evaluation] Computing full 5-fold CV metrics on full 100% data with tuned parameters...")
    X_full = train_df[feature_cols].values
    y_full = train_df["label"].values
    folds_full = train_df["cv_fold"].values

    tuned_cv_f1s = []
    tuned_cv_loglosses = []
    tuned_cv_top1s = []

    for f in range(5):
        tr_idx = (folds_full != f)
        val_idx = (folds_full == f)
        val_sub = train_df.loc[val_idx]

        m_fold = lgb.LGBMClassifier(**full_params)
        m_fold.fit(X_full[tr_idx], y_full[tr_idx], eval_set=[(X_full[val_idx], y_full[val_idx])], callbacks=[lgb.early_stopping(30, verbose=False)])

        p_val = m_fold.predict_proba(X_full[val_idx])[:, 1]
        tuned_cv_loglosses.append(log_loss(y_full[val_idx], p_val))
        tuned_cv_f1s.append(f1_score(y_full[val_idx], (p_val >= 0.5).astype(int), average="macro"))

        t1_acc, _, _ = compute_observation_top1_accuracy(val_sub, p_val)
        tuned_cv_top1s.append(t1_acc)

    tuned_cv_macro_f1 = float(np.mean(tuned_cv_f1s))
    tuned_cv_logloss = float(np.mean(tuned_cv_loglosses))
    tuned_cv_top1_acc = float(np.mean(tuned_cv_top1s))

    # Train final tuned model on all 80 trips
    t_fit_start = time.time()
    tuned_final_model = lgb.LGBMClassifier(**full_params)
    tuned_final_model.fit(X_full, y_full)
    refit_duration = time.time() - t_fit_start
    print(f"[Refit] Full refit complete in {refit_duration:.2f} seconds.")

    # Save tuned model weights and config
    os.makedirs("models/gradient_boosting", exist_ok=True)
    tuned_model_path = "models/gradient_boosting/tuned_lgbm_model.joblib"
    tuned_config_path = "models/gradient_boosting/tuned_config.json"
    joblib.dump(tuned_final_model, tuned_model_path)

    tuned_config_dict = {
        "model_name": "Tuned LightGBM Classifier (Bayesian TPE)",
        "hyperparameters": full_params,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "best_trial_number": best_trial.number,
        "best_cv_macro_f1": round(tuned_cv_macro_f1, 4),
        "saved_at": pd.Timestamp.now().isoformat(),
        "model_path": tuned_model_path,
    }
    with open(tuned_config_path, "w") as f:
        json.dump(tuned_config_dict, f, indent=2)
    print(f"[Artifacts] Saved {tuned_model_path} and {tuned_config_path}")

    # -------------------------------------------------------------
    # Step 3: Evaluate on Student Test & Compare Untuned vs Tuned
    # -------------------------------------------------------------
    print("[Student Test] Evaluating Tuned Model on datasets/student_test (10 trips)...")
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    t_inf_start = time.time()
    test_probs = tuned_final_model.predict_proba(X_test)[:, 1]
    test_inf_time = time.time() - t_inf_start
    test_preds = (test_probs >= 0.5).astype(int)

    test_logloss = float(log_loss(y_test, test_probs))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_acc = float(accuracy_score(y_test, test_preds))

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    top1_acc, top1_macro_f1, top1_df = compute_observation_top1_accuracy(test_df, test_probs)
    mean_continuity, per_trip_cont = compute_route_continuity(top1_df)

    # Load untuned metrics for direct comparison
    with open("results/gradient_boosting_metrics.json", "r") as f:
        untuned_data = json.load(f)

    untuned_cv = untuned_data["cross_validation_5fold"]
    untuned_test_cand = untuned_data["student_test_evaluation"]["candidate_metrics"]
    untuned_test_top1 = untuned_data["student_test_evaluation"]["observation_top1_metrics"]
    untuned_test_cont = untuned_data["student_test_evaluation"]["route_continuity"]
    untuned_test_time = untuned_data["student_test_evaluation"]["timing"]

    comparison = {
        "metric_comparison": {
            "cv_macro_f1": {
                "untuned": untuned_cv["mean_macro_f1"],
                "tuned": round(tuned_cv_macro_f1, 4),
                "delta": round(tuned_cv_macro_f1 - untuned_cv["mean_macro_f1"], 4),
            },
            "cv_log_loss": {
                "untuned": untuned_cv["mean_log_loss"],
                "tuned": round(tuned_cv_logloss, 5),
                "delta": round(tuned_cv_logloss - untuned_cv["mean_log_loss"], 5),
            },
            "student_test_top1_accuracy": {
                "untuned": untuned_test_top1["top1_accuracy_pct"],
                "tuned": round(top1_acc, 2),
                "delta": round(top1_acc - untuned_test_top1["top1_accuracy_pct"], 2),
            },
            "student_test_top1_macro_f1": {
                "untuned": untuned_test_top1["top1_macro_f1"],
                "tuned": round(top1_macro_f1, 4),
                "delta": round(top1_macro_f1 - untuned_test_top1["top1_macro_f1"], 4),
            },
            "student_test_candidate_macro_f1": {
                "untuned": untuned_test_cand["macro_f1"],
                "tuned": round(test_macro_f1, 4),
                "delta": round(test_macro_f1 - untuned_test_cand["macro_f1"], 4),
            },
            "student_test_log_loss": {
                "untuned": untuned_test_cand["log_loss"],
                "tuned": round(test_logloss, 5),
                "delta": round(test_logloss - untuned_test_cand["log_loss"], 5),
            },
            "student_test_route_continuity": {
                "untuned": untuned_test_cont["mean_route_continuity_pct"],
                "tuned": round(mean_continuity, 2),
                "delta": round(mean_continuity - untuned_test_cont["mean_route_continuity_pct"], 2),
            },
            "student_test_inference_time_sec": {
                "untuned": untuned_test_time["student_test_inference_time_sec"],
                "tuned": round(test_inf_time, 3),
            },
        },
        "tuned_student_test_details": {
            "precision": round(test_precision, 4),
            "recall": round(test_recall, 4),
            "accuracy": round(test_acc, 4),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "per_trip_continuity": per_trip_cont,
        },
        "best_hyperparameters": best_params,
        "optuna_search_time_sec": round(optuna_duration, 2),
    }

    with open("results/bayesian_opt_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("[Results] Saved results/bayesian_opt_summary.json")

    # -------------------------------------------------------------
    # Step 4: Visualizations
    # -------------------------------------------------------------
    os.makedirs("visualizations", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Optimization History
    fig, ax = plt.subplots(figsize=(9, 5))
    trials_clean = [t for t in study.trials if t.value is not None]
    trial_nums = [t.number + 1 for t in trials_clean]
    trial_vals = [t.value for t in trials_clean]

    # Compute running maximum (best so far)
    running_max = np.maximum.accumulate(trial_vals)

    ax.scatter(trial_nums, trial_vals, color="#1f77b4", s=50, alpha=0.8, edgecolors="black", label="Trial Objective (CV Macro F1)")
    ax.plot(trial_nums, running_max, color="#d62728", linewidth=2.5, linestyle="--", label="Best So Far")
    ax.set_title("Bayesian Hyperparameter Optimization History (25 Trials)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylabel("5-Fold CV Macro F1-score", fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    hist_path = "visualizations/optuna_optimization_history.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)
    print(f"[Visualization] Saved {hist_path}")

    # Plot 2: Parameter Importance
    try:
        importances = optuna.importance.get_param_importances(study)
        params_sorted = list(importances.keys())[::-1]
        scores_sorted = [importances[k] for k in params_sorted]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(params_sorted, scores_sorted, color="#2ca02c", edgecolor="black", height=0.6)
        ax.set_title("Hyperparameter Importance for CV Macro F1 (Optuna TPE)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Relative Importance", fontsize=11)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%", ha="left", va="center", fontsize=9.5, fontweight="bold")

        ax.set_xlim(0, max(scores_sorted) * 1.15)
        plt.tight_layout()
        imp_path = "visualizations/optuna_param_importance.png"
        fig.savefig(imp_path, dpi=200)
        plt.close(fig)
        print(f"[Visualization] Saved {imp_path}")
    except Exception as e:
        print(f"[Warning] Could not generate param importance plot: {e}")

    # -------------------------------------------------------------
    # Step 5: Print Consolidated Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("      UNTUNED VS. TUNED LIGHTGBM PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Evaluation Metric':<35} | {'Untuned Model':<15} | {'Tuned Model':<15} | {'Delta':<10}")
    print("-" * 80)
    m_comp = comparison["metric_comparison"]
    for metric_name, vals in m_comp.items():
        delta_str = f"{vals.get('delta', 'N/A'):+}" if "delta" in vals else "N/A"
        u_val = f"{vals['untuned']}"
        t_val = f"{vals['tuned']}"
        print(f"{metric_name:<35} | {u_val:<15} | {t_val:<15} | {delta_str:<10}")
    print("=" * 80)


if __name__ == "__main__":
    run_bayesian_optimization(n_trials=25)
