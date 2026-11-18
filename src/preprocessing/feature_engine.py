"""
feature_engine.py - Standardized feature extraction for candidate road segments.
Generates the shared feature matrix across all 5 teammates for training and inference.
"""

from pathlib import Path
from typing import Dict, List, Optional
import math
import numpy as np
import pandas as pd

from src.preprocessing.projection import (
    LocalMetricProjector,
    angular_difference,
    compute_heading,
)
from src.preprocessing.candidate_gen import CandidateGenerator
from src.preprocessing.labeler import assign_observation_ground_truth


def extract_trip_candidate_features(
    trip_data: Dict[str, object],
    search_radius: float = 60.0,
    min_k: int = 5,
    max_k: int = 15,
) -> pd.DataFrame:
    """
    Extracts tabular features for all candidate arcs across every GPS observation
    for a single trip.
    
    Args:
        trip_data: Dictionary returned by load_trip_data (track, nodes, arcs, route).
        search_radius: Candidate search radius in meters.
        min_k: Minimum candidates per observation.
        max_k: Maximum candidates per observation.
        
    Returns:
        pd.DataFrame containing feature matrix with metadata and target columns.
    """
    trip_id = trip_data["trip_id"]
    track_df = trip_data["track"]
    nodes_df = trip_data["nodes"]
    arcs_df = trip_data["arcs"]
    route = trip_data.get("route")

    # Local metric projection
    lon0 = float(track_df["lon"].mean())
    lat0 = float(track_df["lat"].mean())
    projector = LocalMetricProjector(lon0, lat0)

    t_x, t_y = projector.project(track_df["lon"].values, track_df["lat"].values)
    n_x, n_y = projector.project(nodes_df["lon"].values, nodes_df["lat"].values)

    num_obs = len(track_df)
    times = track_df["time"].values

    # Precalculate road network graph degrees
    out_degrees = arcs_df["u"].value_counts().to_dict()
    in_degrees = arcs_df["v"].value_counts().to_dict()

    # Precalculate instantaneous GPS headings and speeds
    gps_headings = np.zeros(num_obs, dtype=np.float32)
    gps_speeds = np.zeros(num_obs, dtype=np.float32)
    time_deltas = np.ones(num_obs, dtype=np.float32)

    for i in range(num_obs):
        if i == 0:
            if num_obs > 1:
                dt = max(0.1, times[1] - times[0])
                dx = t_x[1] - t_x[0]
                dy = t_y[1] - t_y[0]
                gps_headings[i] = compute_heading(0, 0, dx, dy)
                gps_speeds[i] = math.hypot(dx, dy) / dt
                time_deltas[i] = dt
        else:
            dt = max(0.1, times[i] - times[i - 1])
            dx = t_x[i] - t_x[i - 1]
            dy = t_y[i] - t_y[i - 1]
            dist = math.hypot(dx, dy)
            gps_headings[i] = compute_heading(0, 0, dx, dy)
            gps_speeds[i] = dist / dt
            time_deltas[i] = dt

    # Ground-truth labels if route is available
    true_arcs = None
    if route is not None and len(route) > 0:
        true_arcs = assign_observation_ground_truth(t_x, t_y, n_x, n_y, arcs_df, route)

    # Initialize spatial candidate generator
    cand_gen = CandidateGenerator(n_x, n_y, arcs_df, search_radius=search_radius, min_k=min_k, max_k=max_k)

    records = []
    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values

    for obs_idx in range(num_obs):
        px, py = t_x[obs_idx], t_y[obs_idx]
        gps_head = gps_headings[obs_idx]
        gps_spd = gps_speeds[obs_idx]
        dt = time_deltas[obs_idx]
        true_arc = true_arcs[obs_idx] if true_arcs is not None else -1

        candidates = cand_gen.get_candidates_for_point(px, py)

        for rank, cand in enumerate(candidates, start=1):
            arc_id = cand["arc_id"]
            d_perp = cand["dist"]
            proj_ratio = cand["proj_ratio"]
            seg_len = cand["seg_length"]
            seg_head = cand["seg_heading"]

            head_diff = angular_difference(gps_head, seg_head)

            # Ratio between observed velocity and implied segment velocity
            implied_spd = seg_len / dt if dt > 0 else gps_spd
            speed_ratio = gps_spd / (implied_spd + 1e-4)

            u_id = u_nodes[arc_id]
            v_id = v_nodes[arc_id]
            from_deg = out_degrees.get(u_id, 1)
            to_deg = in_degrees.get(v_id, 1)

            is_true = 1 if (true_arcs is not None and arc_id == true_arc) else 0

            records.append({
                "trip_id": str(trip_id),
                "obs_id": int(obs_idx),
                "cand_arc_id": int(arc_id),
                "d_perp": float(d_perp),
                "proj_ratio": float(proj_ratio),
                "heading_diff": float(head_diff),
                "speed_ratio": float(speed_ratio),
                "arc_length": float(seg_len),
                "from_deg": int(from_deg),
                "to_deg": int(to_deg),
                "cand_rank": int(rank),
                "time_delta": float(dt),
                "gps_speed": float(gps_spd),
                "is_true_arc": int(is_true),
            })

    return pd.DataFrame.from_records(records)
