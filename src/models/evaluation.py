"""
evaluation.py - Comprehensive evaluation suite for PathSnap map-matching models.
Computes candidate-level metrics, observation-level Top-1 accuracy,
candidate miss rates, and calibration metrics.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


def compute_candidate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Computes standard binary classification and calibration metrics on candidate pairs.
    """
    # Clip probabilities for stable log-loss
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    loss = float(log_loss(y_true, y_prob_clipped))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1_macro,
        "f1_binary": f1_binary,
        "log_loss": loss,
        "brier_score": brier,
    }


def compute_top1_observation_accuracy(df_with_scores: pd.DataFrame) -> Tuple[float, float]:
    """
    Computes observation-level Top-1 accuracy and candidate miss rate.
    
    Args:
        df_with_scores: DataFrame with ['trip_id', 'obs_id', 'score', 'is_true_arc'].
        
    Returns:
        (top1_accuracy, candidate_miss_rate) in percentages [0.0, 100.0].
    """
    # Group by observation
    grouped = df_with_scores.groupby(["trip_id", "obs_id"])

    total_obs = len(grouped)
    if total_obs == 0:
        return 0.0, 0.0

    top1_correct = 0
    candidate_misses = 0

    for (trip_id, obs_id), group in grouped:
        has_true = group["is_true_arc"].sum() > 0
        if not has_true:
            candidate_misses += 1

        # Select candidate with highest predicted score
        best_row = group.loc[group["score"].idxmax()]
        if best_row["is_true_arc"] == 1:
            top1_correct += 1

    top1_acc = (top1_correct / total_obs) * 100.0
    miss_rate = (candidate_misses / total_obs) * 100.0
    return top1_acc, miss_rate


def get_calibration_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Computes reliability diagram data (fraction of positives vs mean predicted value)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred
