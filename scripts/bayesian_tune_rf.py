"""
bayesian_tune_rf.py - Bayesian Hyperparameter Optimization for Teammate 2's Random Forest.
Optimizes n_estimators, max_depth, min_samples_split, and min_samples_leaf
using Gaussian Process Expected Improvement across Group-Aware Cross-Validation splits.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.random_forest import PathSnapRandomForest
from src.models.bayesian_optimizer import BayesianOptimizer
from src.models.evaluation import compute_candidate_metrics


def run_bayesian_optimization(n_trials: int = 12):
    train_path = PROJECT_ROOT / "data" / "processed" / "train_features.csv.gz"
    print(f"[*] Loading training dataset from {train_path}...")
    train_df = pd.read_csv(train_path)

    # Use a representative 25-trip subset for tuning speed
    unique_trips = train_df["trip_id"].unique()[:25]
    subset_df = train_df[train_df["trip_id"].isin(unique_trips)].copy()
    print(f"[+] Tuning on {len(subset_df):,} candidate samples from {len(unique_trips)} trips")

    y = subset_df["is_true_arc"].values
    groups = subset_df["trip_id"].values

    gkf = GroupKFold(n_splits=3)
    train_val_splits = list(gkf.split(subset_df, y, groups=groups))

    def objective_fn(config: dict) -> float:
        f1_scores = []
        for train_idx, val_idx in train_val_splits:
            X_tr, y_tr = subset_df.iloc[train_idx], y[train_idx]
            X_v, y_v = subset_df.iloc[val_idx], y[val_idx]

            model = PathSnapRandomForest(
                n_estimators=int(config["n_estimators"]),
                max_depth=int(config["max_depth"]),
                min_samples_split=int(config["min_samples_split"]),
                min_samples_leaf=int(config["min_samples_leaf"]),
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            val_probs = model.predict_proba(X_v)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            metrics = compute_candidate_metrics(y_v, val_preds, val_probs)
            f1_scores.append(metrics["f1_macro"])

        return float(np.mean(f1_scores))

    # Parameter search bounds
    param_bounds = {
        "n_estimators": (30, 80, "int"),
        "max_depth": (10, 20, "int"),
        "min_samples_split": (10, 40, "int"),
        "min_samples_leaf": (5, 25, "int"),
    }

    print("\n" + "=" * 65)
    print(f"[*] Starting Bayesian Optimization ({n_trials} Trials)...")
    print("=" * 65)

    opt = BayesianOptimizer(param_bounds, n_initial=4, random_state=42)
    best_config, best_score, trials_df = opt.optimize(objective_fn, n_trials=n_trials)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = reports_dir / "bayesian_trials_rf.csv"
    trials_df.to_csv(trials_csv, index=False)

    print("\n" + "=" * 65)
    print("[+] BAYESIAN OPTIMIZATION COMPLETE")
    print(f"  Best Validation Macro F1: {best_score:.4f}")
    print(f"  Best Configuration: {best_config}")
    print(f"  Saved trial search history to: {trials_csv}")
    print("=" * 65)

    return best_config, best_score, trials_df


if __name__ == "__main__":
    run_bayesian_optimization()
