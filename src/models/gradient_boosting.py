"""
gradient_boosting.py - Teammate 3's Gradient Boosting model implementation.
Uses Scikit-learn's C-accelerated HistGradientBoostingClassifier with balanced weights,
probability calibration, and feature importances.
"""

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURE_COLUMNS = [
    "d_perp",
    "proj_ratio",
    "heading_diff",
    "speed_ratio",
    "arc_length",
    "from_deg",
    "to_deg",
    "cand_rank",
    "time_delta",
    "gps_speed",
]


class PathSnapGradientBoosting:
    """
    Gradient Boosting map-matching candidate evaluator.
    Predicts the posterior probability that a candidate arc is the vehicle's true active segment.
    """

    def __init__(
        self,
        learning_rate: float = 0.08,
        max_iter: int = 150,
        max_leaf_nodes: int = 63,
        min_samples_leaf: int = 25,
        l2_regularization: float = 1.0,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.params = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
            "class_weight": class_weight,
            "random_state": random_state,
        }
        self.model = HistGradientBoostingClassifier(**self.params)
        self.feature_names = FEATURE_COLUMNS
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PathSnapGradientBoosting":
        """Fits the HistGradientBoostingClassifier on candidate features."""
        features = X[self.feature_names]
        self.model.fit(features, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary candidate labels (0 = negative, 1 = true arc)."""
        features = X[self.feature_names]
        return self.model.predict(features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Outputs calibrated posterior candidate probabilities [P(y=0), P(y=1)].
        """
        features = X[self.feature_names]
        return self.model.predict_proba(features)

    def predict_candidate_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the probability score P(is_true_arc=1) for ranking."""
        proba = self.predict_proba(X)
        return proba[:, 1]

    def save(self, filepath: Path):
        """Saves trained model and parameter metadata."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "params": self.params,
            "feature_names": self.feature_names,
            "model": self.model,
        }, filepath)
        print(f"[+] Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "PathSnapGradientBoosting":
        """Loads trained model from disk."""
        data = joblib.load(filepath)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = True
        return instance
