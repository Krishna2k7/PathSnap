"""
labeler.py - Ground-truth route sequence mapping and target label generator.
Associates each continuous GPS observation in a trip with its active traversed arc
from the chronological .route ground-truth sequence.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.preprocessing.projection import point_to_segment_metrics


def assign_observation_ground_truth(
    track_x: np.ndarray,
    track_y: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    arcs_df: pd.DataFrame,
    route: List[int],
    search_window: int = 15,
) -> np.ndarray:
    """
    Assigns the ground-truth active arc_id for each GPS observation by tracking
    forward progression along the ordered sequence of traversed route arcs.
    
    Args:
        track_x: Projected x-coordinates of GPS fixes.
        track_y: Projected y-coordinates of GPS fixes.
        node_x: Projected x-coordinates of road network nodes.
        node_y: Projected y-coordinates of road network nodes.
        arcs_df: DataFrame with 'u' (from_node) and 'v' (to_node).
        route: Chronological sequence of ground-truth arc_ids.
        search_window: Lookahead window in the route sequence to match progression.
        
    Returns:
        Array of ground-truth arc_ids of shape (len(track_x),), matching each GPS point.
    """
    num_points = len(track_x)
    num_route_arcs = len(route)
    assigned_arcs = np.zeros(num_points, dtype=np.int32)

    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values

    curr_route_idx = 0

    for i in range(num_points):
        px, py = track_x[i], track_y[i]

        # Search window in route: look slightly behind and forward
        start_idx = max(0, curr_route_idx - 2)
        end_idx = min(num_route_arcs, curr_route_idx + search_window)

        best_idx = curr_route_idx
        best_dist = float("inf")

        for r_idx in range(start_idx, end_idx):
            arc_id = route[r_idx]
            u_id, v_id = u_nodes[arc_id], v_nodes[arc_id]
            x1, y1 = node_x[u_id], node_y[u_id]
            x2, y2 = node_x[v_id], node_y[v_id]

            dist, proj_ratio, _, _ = point_to_segment_metrics(px, py, x1, y1, x2, y2)

            # Prioritize matching arcs that have reasonable projection along segment
            penalty = 0.0
            if proj_ratio <= 0.01 or proj_ratio >= 0.99:
                penalty = 1.5  # Slight penalty if past endpoints

            effective_dist = dist + penalty

            if effective_dist < best_dist:
                best_dist = effective_dist
                best_idx = r_idx

        # Advance route index monotonically (prevent jumping backwards)
        curr_route_idx = max(curr_route_idx, best_idx)
        assigned_arcs[i] = route[curr_route_idx]

    return assigned_arcs
