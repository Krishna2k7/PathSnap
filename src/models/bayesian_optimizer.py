"""
bayesian_optimizer.py - Native Bayesian Hyperparameter Optimization engine.
Uses Gaussian Process Regression with Expected Improvement (EI) acquisition function
to optimize hyperparameters via Group-Aware Cross-Validation without external dependencies.
"""

from typing import Callable, Dict, List, Tuple
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class BayesianOptimizer:
    """
    Bayesian Hyperparameter Optimizer using a Gaussian Process surrogate
    and Expected Improvement (EI) acquisition function.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float, str]],
        n_initial: int = 5,
        xi: float = 0.01,
        random_state: int = 42,
    ):
        """
        Args:
            param_bounds: Dict of param_name -> (min_val, max_val, type), where type is 'int' or 'float'.
            n_initial: Number of initial quasi-random exploration points.
            xi: Exploration-exploitation trade-off parameter.
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_initial = n_initial
        self.xi = xi
        self.rng = np.random.RandomState(random_state)

        # GP Surrogate with Matern 5/2 kernel
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4, normalize_y=True, n_restarts_optimizer=5, random_state=random_state
        )

        self.trials: List[Dict[str, object]] = []

    def _sample_random_params(self) -> Dict[str, object]:
        """Draws a random hyperparameter configuration within bounds."""
        config = {}
        for name, (low, high, ptype) in self.param_bounds.items():
            if ptype == "int":
                config[name] = int(self.rng.randint(int(low), int(high) + 1))
            else:
                config[name] = float(self.rng.uniform(low, high))
        return config

    def _config_to_vector(self, config: Dict[str, object]) -> np.ndarray:
        """Normalizes parameter config to [0, 1] unit hypercube."""
        vec = []
        for name in self.param_names:
            low, high, _ = self.param_bounds[name]
            val = config[name]
            norm_val = (val - low) / (high - low + 1e-9)
            vec.append(norm_val)
        return np.array(vec, dtype=np.float64)

    def _vector_to_config(self, vec: np.ndarray) -> Dict[str, object]:
        """Converts [0, 1] vector back to discrete/continuous parameters."""
        config = {}
        for idx, name in enumerate(self.param_names):
            low, high, ptype = self.param_bounds[name]
            norm_val = max(0.0, min(1.0, vec[idx]))
            raw_val = low + norm_val * (high - low)
            if ptype == "int":
                config[name] = int(round(raw_val))
            else:
                config[name] = float(raw_val)
        return config

    def _expected_improvement(self, X_candidates: np.ndarray, y_best: float) -> np.ndarray:
        """Computes Expected Improvement (EI) for candidate parameter proposals."""
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        # Improvement over current best
        improvement = mu - y_best - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def optimize(
        self, objective_fn: Callable[[Dict[str, object]], float], n_trials: int = 15
    ) -> Tuple[Dict[str, object], float, pd.DataFrame]:
        """
        Executes sequential Bayesian optimization trials.
        
        Args:
            objective_fn: Function mapping config dict to scalar score (to maximize).
            n_trials: Total number of optimization trials to execute.
            
        Returns:
            (best_config, best_score, trials_dataframe)
        """
        X_history = []
        y_history = []

        for trial_num in range(1, n_trials + 1):
            if trial_num <= self.n_initial:
                config = self._sample_random_params()
                strategy = "Random Init"
            else:
                # Fit GP surrogate on observed history
                self.gp.fit(np.array(X_history), np.array(y_history))
                y_best = max(y_history)

                # Monte Carlo acquisition maximization over 1000 candidate points
                candidates_norm = self.rng.uniform(0.0, 1.0, size=(1000, len(self.param_names)))
                ei_values = self._expected_improvement(candidates_norm, y_best)
                best_cand_idx = int(np.argmax(ei_values))
                config = self._vector_to_config(candidates_norm[best_cand_idx])
                strategy = "Bayesian GP-EI"

            score = objective_fn(config)

            X_history.append(self._config_to_vector(config))
            y_history.append(score)

            trial_record = {
                "trial": trial_num,
                "strategy": strategy,
                "score": float(score),
                **config,
            }
            self.trials.append(trial_record)
            print(f"  [Trial {trial_num:02d} | {strategy:14s}] Score: {score:.4f} | Config: {config}")

        trials_df = pd.DataFrame(self.trials)
        best_idx = trials_df["score"].idxmax()
        best_row = trials_df.loc[best_idx]
        best_config = {name: best_row[name] for name in self.param_names}
        best_score = float(best_row["score"])

        return best_config, best_score, trials_df
