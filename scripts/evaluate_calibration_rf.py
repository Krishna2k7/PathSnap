"""
evaluate_calibration_rf.py - Evaluates probability calibration for Teammate 2's Random Forest.
Computes Brier Score, Expected Calibration Error (ECE), reliability curve coordinates,
and fits a Platt Sigmoid calibrator.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.random_forest import PathSnapRandomForest, FEATURE_COLUMNS
from src.models.calibration import evaluate_calibration


def run_calibration_evaluation():
    test_path = PROJECT_ROOT / "data" / "processed" / "student_test_features.csv.gz"
    model_path = PROJECT_ROOT / "models" / "random_forest.joblib"

    if not model_path.exists():
        print(f"[!] Random forest model not found at {model_path}")
        return

    print("[*] Loading trained Random Forest model and student-test data...")
    rf_model = PathSnapRandomForest.load(model_path)
    test_df = pd.read_csv(test_path)

    y_test = test_df["is_true_arc"].values
    raw_probs = rf_model.predict_candidate_scores(test_df)

    # 1. Evaluate uncalibrated probabilities
    raw_calib = evaluate_calibration(y_test, raw_probs, n_bins=10)

    print("\n" + "=" * 60)
    print("PROBABILITY CALIBRATION ANALYSIS (RANDOM FOREST)")
    print("=" * 60)
    print(f"  Uncalibrated Brier Score: {raw_calib['brier_score']:.4f}")
    print(f"  Uncalibrated Log Loss:    {raw_calib['log_loss']:.4f}")
    print(f"  Uncalibrated ECE:         {raw_calib['ece']:.4f}")

    # Reliability curve points
    print("\n  Reliability Diagram (10 Bins):")
    print("  Bin Pred Prob  |  True Fraction")
    print("  " + "-" * 32)
    for p_pred, p_true in zip(raw_calib["prob_pred"], raw_calib["prob_true"]):
        print(f"     {p_pred:6.3f}      |     {p_true:6.3f}")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "calibration_analysis_rf.json"

    with open(out_file, "w") as f:
        json.dump({
            "model": "Random Forest",
            "uncalibrated": raw_calib,
        }, f, indent=2)

    print(f"\n[+] Calibration report saved to {out_file}")
    return raw_calib


if __name__ == "__main__":
    run_calibration_evaluation()
