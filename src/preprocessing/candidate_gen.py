"""
candidate_gen.py - Spatial road candidate generation using KDTree.
Retrieves candidate road segments for each GPS observation within a metric threshold
and guarantees a minimum top-K candidate set.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.preprocessing.projection import point_to_segment_metrics


class CandidateGenerator:
    """
    Builds a spatial index over road network arcs for a given trip,
    and queries candidate segments for each GPS observation.
    """

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        arcs_df: pd.DataFrame,
        search_radius: float = 60.0,
        min_k: int = 5,
        max_k: int = 15,
    ):
        """
        Args:
            node_x: Projected x-coordinates of road network nodes.
            node_y: Projected y-coordinates of road network nodes.
            arcs_df: DataFrame with 'u' (from_node) and 'v' (to_node).
            search_radius: Metric distance threshold (meters) for candidate search.
            min_k: Minimum number of candidate arcs to retain per observation.
            max_k: Maximum number of candidate arcs to retain per observation.
        """
        self.node_x = node_x
        self.node_y = node_y
        self.u = arcs_df["u"].values
        self.v = arcs_df["v"].values
        self.num_arcs = len(arcs_df)
        self.search_radius = search_radius
        self.min_k = min_k
        self.max_k = max_k

        # Calculate arc geometry in 2D metric space
        x1, y1 = self.node_x[self.u], self.node_y[self.u]
        x2, y2 = self.node_x[self.v], self.node_y[self.v]
        self.mid_x = (x1 + x2) / 2.0
        self.mid_y = (y1 + y2) / 2.0
        self.half_len = np.hypot(x2 - x1, y2 - y1) / 2.0
        self.max_half_len = float(self.half_len.max()) if len(self.half_len) > 0 else 0.0

        # Build KDTree on arc midpoints
        self.tree = KDTree(np.column_stack([self.mid_x, self.mid_y]))

    def get_candidates_for_point(self, px: float, py: float) -> List[Dict[str, float]]:
        """
        Queries candidate arcs for a single projected GPS point (px, py).
        
        Returns:
            List of candidate dictionaries sorted by perpendicular distance:
            [{
                'arc_id': int,
                'dist': float,
                'proj_ratio': float,
                'seg_length': float,
                'seg_heading': float
            }, ...]
        """
        # Query radius accounts for distance from midpoint to arc endpoints
        query_r = self.search_radius + min(self.max_half_len, 100.0)
        cand_indices = self.tree.query_ball_point([px, py], r=query_r)

        # Fallback to nearest neighbors if ball query is too sparse
        if len(cand_indices) < self.min_k:
            k_query = min(self.min_k, self.num_arcs)
            _, nearest_idx = self.tree.query([px, py], k=k_query)
            if np.isscalar(nearest_idx):
                nearest_idx = [nearest_idx]
            cand_indices = list(set(list(cand_indices) + list(nearest_idx)))

        # Evaluate exact geometric distance for each retrieved candidate
        results = []
        for arc_id in cand_indices:
            u_id = self.u[arc_id]
            v_id = self.v[arc_id]
            x1, y1 = self.node_x[u_id], self.node_y[u_id]
            x2, y2 = self.node_x[v_id], self.node_y[v_id]

            dist, proj_ratio, seg_len, seg_heading = point_to_segment_metrics(px, py, x1, y1, x2, y2)
            results.append({
                "arc_id": int(arc_id),
                "dist": dist,
                "proj_ratio": proj_ratio,
                "seg_length": seg_len,
                "seg_heading": seg_heading,
            })

        # Sort candidates by perpendicular distance
        results.sort(key=lambda item: item["dist"])

        # Filter: keep within search radius, but ensure at least min_k candidates
        filtered = [c for c in results if c["dist"] <= self.search_radius]
        if len(filtered) < self.min_k:
            filtered = results[: self.min_k]

        return filtered[: self.max_k]
