"""
calibration.py - Probability calibration evaluation and calibrator fitting for PathSnap.
Evaluates reliability curves, Brier scores, and fits Platt Scaling (Sigmoid)
and Isotonic Regression calibrators to ensure reliable HMM emission likelihoods.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def evaluate_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, object]:
    """
    Evaluates probability calibration quality.
    
    Returns:
        dict containing:
            'brier_score': float
            'log_loss': float
            'expected_calibration_error': float
            'prob_true': np.ndarray
            'prob_pred': np.ndarray
    """
    eps = 1e-7
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)

    brier = float(brier_score_loss(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob_clipped))

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        in_bin = bin_assignments == i
        count = np.sum(in_bin)
        if count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (count / N) * abs(bin_acc - bin_conf)

    return {
        "brier_score": brier,
        "log_loss": loss,
        "ece": float(ece),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def fit_calibrator(
    base_model, X_val: pd.DataFrame, y_val: np.ndarray, method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """
    Fits a post-hoc probability calibrator on validation fold predictions.
    
    Args:
        base_model: Pre-fitted base classifier (e.g. RandomForest).
        X_val: Validation feature DataFrame.
        y_val: Validation target labels.
        method: 'sigmoid' (Platt scaling) or 'isotonic' (Isotonic regression).
        
    Returns:
        CalibratedClassifierCV wrapper fitted on the validation set.
    """
    calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated
