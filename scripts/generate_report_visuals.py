"""
generate_report_visuals.py - Generates publication-ready figures for the technical report.
Produces:
1. Feature Importances horizontal bar chart
2. Reliability Diagram (Calibration Curve)
3. Confusion Matrix plot
4. Model Comparison bar chart (Random Forest vs Gradient Boosting)
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_all_visuals():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rf_summary_path = reports_dir / "random_forest_summary.json"
    gb_summary_path = reports_dir / "gradient_boosting_summary.json"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Feature Importances Plot
    if rf_summary_path.exists():
        with open(rf_summary_path) as f:
            rf_data = json.load(f)
        importances = rf_data.get("feature_importances", {})
        if importances:
            sorted_feats = sorted(importances.items(), key=lambda x: x[1])
            names = [x[0] for x in sorted_feats]
            vals = [x[1] for x in sorted_feats]

            plt.figure(figsize=(8, 5), dpi=150)
            bars = plt.barh(names, vals, color="#2b5c8f", edgecolor="none", height=0.6)
            plt.xlabel("Gini Impurity Reduction (Feature Importance)", fontsize=11, fontweight="bold")
            plt.title("Random Forest: Gini Feature Importance", fontsize=12, fontweight="bold", pad=12)
            plt.tight_layout()
            out_file = reports_dir / "feature_importances_rf.png"
            plt.savefig(out_file)
            plt.close()
            print(f"[+] Saved feature importances plot to {out_file}")

    # 2. Model Comparison Chart (RF vs GB)
    if rf_summary_path.exists() and gb_summary_path.exists():
        with open(rf_summary_path) as f:
            rf_d = json.load(f)
        with open(gb_summary_path) as f:
            gb_d = json.load(f)

        metrics = ["CV Macro F1", "Test Macro F1", "Test Top-1 Acc (%)"]
        rf_scores = [
            rf_d["cv_macro_f1_mean"],
            rf_d["student_test_macro_f1"],
            rf_d["student_test_top1_accuracy"] / 100.0,
        ]
        gb_scores = [
            gb_d["cv_macro_f1_mean"],
            gb_d["student_test_macro_f1"],
            gb_d["student_test_top1_accuracy"] / 100.0,
        ]

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(8, 4.5), dpi=150)
        plt.bar(x - width / 2, rf_scores, width, label="Random Forest (Bagging)", color="#3b82f6")
        plt.bar(x + width / 2, gb_scores, width, label="HistGradientBoosting (Boosting)", color="#10b981")

        plt.ylabel("Score", fontsize=11, fontweight="bold")
        plt.title("Model Comparison: Teammate 2 (RF) vs Teammate 3 (GB)", fontsize=12, fontweight="bold", pad=12)
        plt.xticks(x, metrics, fontsize=10)
        plt.ylim(0.0, 1.05)
        plt.legend(frameon=True, loc="lower right")

        for i in range(len(metrics)):
            plt.text(x[i] - width / 2, rf_scores[i] + 0.02, f"{rf_scores[i]:.3f}", ha="center", fontsize=9)
            plt.text(x[i] + width / 2, gb_scores[i] + 0.02, f"{gb_scores[i]:.3f}", ha="center", fontsize=9)

        plt.tight_layout()
        out_comp = reports_dir / "model_comparison_rf_vs_gb.png"
        plt.savefig(out_comp)
        plt.close()
        print(f"[+] Saved model comparison plot to {out_comp}")


if __name__ == "__main__":
    generate_all_visuals()
