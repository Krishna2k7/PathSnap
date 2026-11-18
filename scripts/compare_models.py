"""
compare_models.py - Comparative analysis between Teammate 2 (Random Forest)
and Teammate 3 (Gradient Boosting) evaluated on identical student-test data.
Generates markdown and CSV comparison tables for the hackathon report.
"""

from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_comparison_table():
    reports_dir = PROJECT_ROOT / "reports"
    rf_path = reports_dir / "random_forest_summary.json"
    gb_path = reports_dir / "gradient_boosting_summary.json"

    if not rf_path.exists():
        print(f"[!] Random Forest summary not found at {rf_path}")
        return
    if not gb_path.exists():
        print(f"[!] Gradient Boosting summary not found at {gb_path}")
        return

    with open(rf_path) as f:
        rf_data = json.load(f)
    with open(gb_path) as f:
        gb_data = json.load(f)

    rows = [
        {
            "Teammate": "Teammate 2",
            "Model Name": "Random Forest",
            "Algorithm Family": "Bagging / Tree Ensemble",
            "Key Configuration": f"n_est=80, depth=16, leaf=10, balanced",
            "CV Macro F1": f"{rf_data['cv_macro_f1_mean']:.4f} +/- {rf_data['cv_macro_f1_std']:.4f}",
            "CV Top-1 Acc": f"{rf_data['cv_top1_acc_mean']:.2f}%",
            "Student-Test Macro F1": f"{rf_data['student_test_macro_f1']:.4f}",
            "Student-Test Top-1 Acc": f"{rf_data['student_test_top1_accuracy']:.2f}%",
            "Log Loss": f"{rf_data['student_test_log_loss']:.4f}",
            "Brier Score": f"{rf_data['student_test_brier_score']:.4f}",
            "Inference Latency": f"{rf_data['student_test_inference_time_s'] / 10.0 * 1000:.1f} ms/trip",
            "Full Fit Time": f"{rf_data['full_training_time_s']:.1f}s",
        },
        {
            "Teammate": "Teammate 3",
            "Model Name": "HistGradientBoosting",
            "Algorithm Family": "Boosting",
            "Key Configuration": f"max_iter=150, lr=0.08, leaves=45, balanced",
            "CV Macro F1": f"{gb_data['cv_macro_f1_mean']:.4f} +/- {gb_data['cv_macro_f1_std']:.4f}",
            "CV Top-1 Acc": f"{gb_data['cv_top1_acc_mean']:.2f}%",
            "Student-Test Macro F1": f"{gb_data['student_test_macro_f1']:.4f}",
            "Student-Test Top-1 Acc": f"{gb_data['student_test_top1_accuracy']:.2f}%",
            "Log Loss": f"{gb_data['student_test_log_loss']:.4f}",
            "Brier Score": f"{gb_data['student_test_brier_score']:.4f}",
            "Inference Latency": f"{gb_data['student_test_inference_time_s'] / 10.0 * 1000:.1f} ms/trip",
            "Full Fit Time": f"{gb_data['full_training_time_s']:.1f}s",
        },
    ]

    comp_df = pd.DataFrame(rows)
    csv_out = reports_dir / "teammate_model_comparison.csv"
    comp_df.to_csv(csv_out, index=False)
    print(f"[+] Comparison table saved to {csv_out}")

    print("\n" + "=" * 95)
    print("PATHSNAP MODEL COMPARISON: TEAMMATE 2 (RANDOM FOREST) vs TEAMMATE 3 (GRADIENT BOOSTING)")
    print("=" * 95)
    print(comp_df.to_string(index=False))
    return comp_df


if __name__ == "__main__":
    generate_comparison_table()
