"""
Models package for PathSnap.
Exposes Teammate 3's Gradient Boosting model and evaluation metrics.
"""

from src.models.gradient_boosting import PathSnapGradientBoosting, FEATURE_COLUMNS
from src.models.evaluation import (
    compute_candidate_metrics,
    compute_top1_observation_accuracy,
    get_calibration_data,
)

__all__ = [
    "PathSnapGradientBoosting",
    "FEATURE_COLUMNS",
    "compute_candidate_metrics",
    "compute_top1_observation_accuracy",
    "get_calibration_data",
]
