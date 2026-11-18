"""
random_forest.py - Teammate 2's Bagging & Tree Ensemble Model Implementation.
Uses Scikit-learn's parallelized RandomForestClassifier with balanced class weights,
probability calibration, and feature importances.
"""

from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.gradient_boosting import FEATURE_COLUMNS


class PathSnapRandomForest:
    """
    Teammate 2: Random Forest candidate evaluator.
    Predicts the posterior probability that a candidate road segment is the true active arc
    using bootstrap aggregation of decision trees.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: str = "sqrt",
        class_weight: str = "balanced",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = FEATURE_COLUMNS
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "PathSnapRandomForest":
        """Fits the RandomForestClassifier on candidate features."""
        features = X[self.feature_names]
        self.model.fit(features, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary candidate labels (0 = negative candidate, 1 = true arc)."""
        features = X[self.feature_names]
        return self.model.predict(features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Outputs posterior candidate probabilities [P(y=0), P(y=1)]."""
        features = X[self.feature_names]
        return self.model.predict_proba(features)

    def predict_candidate_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Returns the probability score P(is_true_arc=1) for ranking and HMM emission."""
        proba = self.predict_proba(X)
        return proba[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns Gini feature importances across all bagged trees."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save(self, filepath: Path):
        """Saves model weights and parameter metadata."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "params": self.params,
            "feature_names": self.feature_names,
            "model": self.model,
        }, filepath)
        print(f"[+] Saved Random Forest model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "PathSnapRandomForest":
        """Loads trained Random Forest model from disk."""
        data = joblib.load(filepath)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_fitted = True
        return instance
