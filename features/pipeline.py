"""
PathSnap Feature Engineering & Preprocessing Pipeline
=====================================================
Modular pipeline for:
1. Spatial candidate generation:
   - Queries metric buffer around each GPS observation using shapely.STRtree.
   - Fallback to nearest arcs for sparse or dropout points.
   - Extracts top-K candidates ranked by perpendicular distance.
   - Logs candidate miss rate (% of GPS observations whose true route arc is missing).
2. Feature extraction per (trip_id, obs_idx, candidate_arc_id):
   - Geometric: perpendicular distance (m), distance rank, longitudinal offset (m),
     projection ratio (t in [0, 1]), start/end node distances (m), arc length (m).
   - Kinematic & Heading: GPS speed (m/s), acceleration (m/s^2), timestamp gap (s),
     heading difference (deg), cos(heading difference), stationary indicator,
     speed-to-arc-length ratio.
   - Topological: from/to node in/out-degrees, sequential transition connectivity
     to top-1 candidate at t-1 (is_connected_from_prev, is_same_arc_as_prev,
     topological graph distance from prev).
   - Target label: 1 if candidate_arc_id == ground_truth_arc_id, else 0.
3. Leak-Free Scaling & Cross-Validation:
   - Deterministic 5-fold GroupKFold split on train grouped strictly by trip_id.
   - StandardScaler fitted EXCLUSIVELY on datasets/train.
   - Exports:
     - processed_data/scaler.joblib
     - processed_data/feature_columns.json
     - processed_data/train_features.parquet
     - processed_data/student_test_features.parquet
4. Standalone FastAPI inference transform:
   - transform_single_trip(trip_dir, scaler, feature_cols)
"""

import os
import glob
import json
import joblib
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely
from shapely import linestrings, points, STRtree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from data_processing.spatial_utils import (
    get_transformer,
    project_coords,
    compute_bearing,
    compute_angle_difference,
    project_points_to_segments,
)


FEATURE_COLUMNS = [
    "perp_dist",
    "dist_rank",
    "proj_offset_ratio",
    "longitudinal_offset",
    "dist_to_start_node",
    "dist_to_end_node",
    "arc_length",
    "gps_speed",
    "gps_accel",
    "time_gap",
    "heading_diff_deg",
    "cos_heading_diff",
    "is_stationary",
    "speed_vs_length_ratio",
    "from_node_in_degree",
    "from_node_out_degree",
    "to_node_in_degree",
    "to_node_out_degree",
    "is_connected_from_prev",
    "is_same_arc_as_prev",
    "graph_dist_from_prev",
]


def match_ground_truth_route(
    gps_x: np.ndarray,
    gps_y: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    u_nodes: np.ndarray,
    v_nodes: np.ndarray,
    route: np.ndarray,
) -> np.ndarray:
    """
    Associates each GPS observation with its ground-truth arc_id from the .route sequence.
    Finds the closest route segment to each GPS point.
    """
    r_u = u_nodes[route]
    r_v = v_nodes[route]
    p1x, p1y = nx[r_u], ny[r_u]
    p2x, p2y = nx[r_v], ny[r_v]
    dx = p2x - p1x
    dy = p2y - p1y
    l2 = np.where(dx * dx + dy * dy == 0.0, 1e-9, dx * dx + dy * dy)

    true_arc_ids = []
    for x, y in zip(gps_x, gps_y):
        t_val = np.clip(((x - p1x) * dx + (y - p1y) * dy) / l2, 0.0, 1.0)
        proj_x = p1x + t_val * dx
        proj_y = p1y + t_val * dy
        d = np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)
        best_r = np.argmin(d)
        true_arc_ids.append(route[best_r])

    return np.array(true_arc_ids, dtype=np.int64)


def extract_features_single_trip(
    trip_dir: str,
    search_radius: float = 80.0,
    top_k: int = 7,
    has_ground_truth: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts candidate road segments and rich features for a single trip.
    
    Parameters
    ----------
    trip_dir : str
        Path to trip folder containing .track, .nodes, .arcs, and optional .route.
    search_radius : float
        Metric buffer radius (meters) for candidate search. Default 80m.
    top_k : int
        Number of top candidate arcs to retain per GPS point. Default 7.
    has_ground_truth : bool
        Whether to parse .route and compute candidate miss rate and binary label.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Candidate features DataFrame and metadata metrics (e.g. miss count).
    """
    trip_id = os.path.basename(trip_dir)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    # Load raw data
    track = pd.read_csv(
        track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype=np.float64
    )
    nodes = pd.read_csv(
        nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64
    )
    arcs = pd.read_csv(
        arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64
    )

    route = None
    if has_ground_truth and os.path.exists(route_path):
        route = pd.read_csv(route_path, header=None, names=["arc_id"], dtype=np.int64)[
            "arc_id"
        ].values

    n_gps = len(track)
    n_nodes = len(nodes)
    u_nodes = arcs["u"].values
    v_nodes = arcs["v"].values

    # 1. Coordinate projection to metric UTM
    mean_lon, mean_lat = float(track["lon"].mean()), float(track["lat"].mean())
    transformer, epsg = get_transformer(mean_lon, mean_lat)
    nx, ny = project_coords(nodes["lon"].values, nodes["lat"].values, transformer)
    gx, gy = project_coords(track["lon"].values, track["lat"].values, transformer)
    gt = track["t"].values

    # 2. Kinematics along GPS trajectory
    diff_t = np.diff(gt)
    diff_t = np.where(diff_t <= 0.0, 1.0, diff_t)
    diff_gx = np.diff(gx)
    diff_gy = np.diff(gy)
    step_dist = np.sqrt(diff_gx ** 2 + diff_gy ** 2)

    gps_speeds = np.zeros(n_gps, dtype=np.float64)
    gps_speeds[:-1] = step_dist / diff_t
    gps_speeds[-1] = gps_speeds[-2] if n_gps > 1 else 0.0

    gps_accels = np.zeros(n_gps, dtype=np.float64)
    diff_speed = np.diff(gps_speeds)
    gps_accels[:-1] = diff_speed / diff_t
    gps_accels[-1] = 0.0

    time_gaps = np.zeros(n_gps, dtype=np.float64)
    time_gaps[:-1] = diff_t
    time_gaps[-1] = diff_t[-1] if len(diff_t) > 0 else 1.0

    gps_bearings = np.zeros(n_gps, dtype=np.float64)
    gps_bearings[:-1] = compute_bearing(gx[:-1], gy[:-1], gx[1:], gy[1:])
    gps_bearings[-1] = gps_bearings[-2] if n_gps > 1 else 0.0

    stationary_mask = np.zeros(n_gps, dtype=bool)
    stationary_mask[:-1] = (step_dist < 0.5) | (gps_speeds[:-1] < 0.3)
    stationary_mask[-1] = stationary_mask[-2] if n_gps > 1 else False

    # 3. Road-network node degrees & adjacency
    in_degrees = np.bincount(v_nodes, minlength=n_nodes)
    out_degrees = np.bincount(u_nodes, minlength=n_nodes)

    # Local outgoing adjacency for fast topological step distance
    outgoing_adj: Dict[int, set] = defaultdict(set)
    for u_val, v_val in zip(u_nodes, v_nodes):
        outgoing_adj[u_val].add(v_val)

    # 4. Spatial index for candidate generation
    coords = np.stack([nx, ny], axis=1)
    seg_coords = np.stack([coords[u_nodes], coords[v_nodes]], axis=1)
    tree = STRtree(linestrings(seg_coords))

    # 5. Ground truth mapping if available
    true_arc_ids = None
    if route is not None:
        true_arc_ids = match_ground_truth_route(gx, gy, nx, ny, u_nodes, v_nodes, route)

    # 6. Spatial candidate search
    gps_pts = points(gx, gy)
    buffers = shapely.buffer(gps_pts, search_radius)
    q_pts, q_arcs = tree.query(buffers, predicate="intersects")

    cands_by_pt: Dict[int, List[int]] = defaultdict(list)
    for p_idx, a_idx in zip(q_pts, q_arcs):
        cands_by_pt[p_idx].append(a_idx)

    # Build per-candidate rows
    rows = []
    miss_count = 0
    prev_top_candidate_arc = -1
    prev_top_candidate_v = -1

    for i in range(n_gps):
        c_list = cands_by_pt[i]
        # Fallback if no candidate within search radius
        if len(c_list) < 2:
            nearest_arcs = list(tree.query_nearest(gps_pts[i], max_distance=500.0))
            for na in nearest_arcs:
                if na not in c_list:
                    c_list.append(na)

        px, py = gx[i], gy[i]
        p_speed = gps_speeds[i]
        p_accel = gps_accels[i]
        p_dt = time_gaps[i]
        p_bearing = gps_bearings[i]
        p_stat = stationary_mask[i]

        cand_arr = np.array(c_list, dtype=np.int64)
        c_u = u_nodes[cand_arr]
        c_v = v_nodes[cand_arr]
        p1x, p1y = nx[c_u], ny[c_u]
        p2x, p2y = nx[c_v], ny[c_v]

        proj_res = project_points_to_segments(px, py, p1x, p1y, p2x, p2y)
        perp_dists = proj_res["perp_dist"]

        # Rank candidates by perpendicular distance
        order = np.argsort(perp_dists)
        sorted_cands = cand_arr[order]
        sorted_dists = perp_dists[order]

        # Check candidate miss rate
        if true_arc_ids is not None:
            top_selected = sorted_cands[:top_k]
            if true_arc_ids[i] not in top_selected:
                miss_count += 1

        # Select top-K candidates
        k_count = min(top_k, len(sorted_cands))
        top_indices = order[:k_count]

        # Extract features for each candidate
        for rank, cand_pos in enumerate(top_indices):
            c_arc_id = int(cand_arr[cand_pos])
            u_i = int(c_u[cand_pos])
            v_i = int(c_v[cand_pos])

            dist_m = float(perp_dists[cand_pos])
            arc_len = float(proj_res["seg_len"][cand_pos])
            long_offset = float(proj_res["longitudinal_offset"][cand_pos])
            ratio_offset = float(proj_res["proj_offset_ratio"][cand_pos])
            d_p1 = float(proj_res["dist_to_p1"][cand_pos])
            d_p2 = float(proj_res["dist_to_p2"][cand_pos])
            arc_bearing = float(proj_res["arc_bearing"][cand_pos])

            # Heading difference
            if p_stat:
                h_diff = 0.0
                cos_h = 1.0
            else:
                h_diff = float(compute_angle_difference(p_bearing, arc_bearing))
                cos_h = float(np.cos(np.radians(h_diff)))

            # Topological connectivity from previous top-1 candidate
            if i == 0 or prev_top_candidate_arc == -1:
                is_connected = 1
                is_same_arc = 0
                graph_dist = 0.0
            else:
                is_same_arc = 1 if c_arc_id == prev_top_candidate_arc else 0
                if is_same_arc:
                    is_connected = 1
                    graph_dist = 0.0
                elif u_i == prev_top_candidate_v:
                    is_connected = 1
                    graph_dist = arc_len
                elif u_i in outgoing_adj.get(prev_top_candidate_v, set()):
                    is_connected = 1
                    graph_dist = 2.0 * arc_len
                else:
                    is_connected = 0
                    graph_dist = 500.0

            speed_len_ratio = p_speed / (arc_len + 1e-3)

            row_data = {
                "trip_id": trip_id,
                "obs_idx": i,
                "t": float(gt[i]),
                "candidate_arc_id": c_arc_id,
                "perp_dist_raw": dist_m,
                "perp_dist": dist_m,
                "dist_rank": rank,
                "proj_offset_ratio": ratio_offset,
                "longitudinal_offset": long_offset,
                "dist_to_start_node": d_p1,
                "dist_to_end_node": d_p2,
                "arc_length": arc_len,
                "gps_speed": p_speed,
                "gps_accel": p_accel,
                "time_gap": p_dt,
                "heading_diff_deg": h_diff,
                "cos_heading_diff": cos_h,
                "is_stationary": int(p_stat),
                "speed_vs_length_ratio": speed_len_ratio,
                "from_node_in_degree": int(in_degrees[u_i]),
                "from_node_out_degree": int(out_degrees[u_i]),
                "to_node_in_degree": int(in_degrees[v_i]),
                "to_node_out_degree": int(out_degrees[v_i]),
                "is_connected_from_prev": is_connected,
                "is_same_arc_as_prev": is_same_arc,
                "graph_dist_from_prev": graph_dist,
            }

            if true_arc_ids is not None:
                row_data["label"] = 1 if (c_arc_id == true_arc_ids[i]) else 0

            rows.append(row_data)

        # Update top candidate for next timestamp
        prev_top_candidate_arc = int(sorted_cands[0])
        prev_top_candidate_v = int(v_nodes[sorted_cands[0]])

    df = pd.DataFrame(rows)
    metrics = {
        "trip_id": trip_id,
        "n_gps_points": n_gps,
        "n_candidates_generated": len(df),
        "miss_count": miss_count,
        "miss_rate_pct": round(float(miss_count / n_gps * 100.0), 3) if n_gps > 0 else 0.0,
    }
    return df, metrics


def build_partition_features(
    partition_dir: str,
    top_k: int = 7,
    search_radius: float = 80.0,
    has_ground_truth: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds the complete feature dataframe for all trips in a partition.
    """
    trip_dirs = sorted(glob.glob(os.path.join(partition_dir, "*")))
    all_dfs = []
    total_gps = 0
    total_misses = 0

    print(f"[Pipeline] Processing {len(trip_dirs)} trips in {partition_dir}...")
    for idx, td in enumerate(trip_dirs):
        df, m = extract_features_single_trip(
            td, search_radius=search_radius, top_k=top_k, has_ground_truth=has_ground_truth
        )
        all_dfs.append(df)
        total_gps += m["n_gps_points"]
        total_misses += m["miss_count"]
        if (idx + 1) % 20 == 0 or (idx + 1) == len(trip_dirs):
            print(
                f"  -> Processed {idx + 1}/{len(trip_dirs)} trips. "
                f"Cumulative GPS: {total_gps}, Misses: {total_misses} "
                f"({total_misses / total_gps * 100.0:.2f}%)"
            )

    combined_df = pd.concat(all_dfs, ignore_index=True)
    summary = {
        "trip_count": len(trip_dirs),
        "total_gps_points": total_gps,
        "total_candidate_rows": len(combined_df),
        "total_misses": total_misses,
        "candidate_miss_rate_pct": round(float(total_misses / total_gps * 100.0), 3)
        if total_gps > 0
        else 0.0,
        "avg_candidates_per_point": round(float(len(combined_df) / total_gps), 2)
        if total_gps > 0
        else 0.0,
    }
    return combined_df, summary


def create_group_cv_folds(
    df: pd.DataFrame,
    group_col: str = "trip_id",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Creates deterministic 5-fold GroupKFold splits grouped strictly by trip_id.
    Guarantees no trip is ever leaked across train/val folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    df = df.copy()
    df["cv_fold"] = -1

    unique_groups = df[group_col].unique()
    dummy_x = np.zeros(len(df))
    dummy_y = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(dummy_x, dummy_y, groups=df[group_col])
    ):
        df.loc[val_idx, "cv_fold"] = fold

    # Verification: assert zero overlap of trips across folds
    for f in range(n_splits):
        fold_trips = set(df[df["cv_fold"] == f][group_col].unique())
        other_trips = set(df[df["cv_fold"] != f][group_col].unique())
        assert (
            len(fold_trips.intersection(other_trips)) == 0
        ), f"Data leakage detected in fold {f}!"

    return df


def fit_and_save_pipeline(
    train_dir: str = "datasets/train",
    student_test_dir: str = "datasets/student_test",
    output_dir: str = "processed_data",
    top_k: int = 7,
    search_radius: float = 80.0,
) -> Dict[str, Any]:
    """
    Executes the complete leak-free feature extraction and scaling pipeline:
    1. Extracts raw features for datasets/train and datasets/student_test.
    2. Assigns deterministic 5-fold GroupKFold CV splits to train data.
    3. Fits StandardScaler EXCLUSIVELY on datasets/train.
    4. Transforms both train and student_test features.
    5. Saves scaler, column schema, and Parquet datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract features
    train_df, train_stats = build_partition_features(
        train_dir, top_k=top_k, search_radius=search_radius, has_ground_truth=True
    )
    test_df, test_stats = build_partition_features(
        student_test_dir, top_k=top_k, search_radius=search_radius, has_ground_truth=True
    )

    # 2. GroupKFold CV on train
    train_df = create_group_cv_folds(train_df, group_col="trip_id", n_splits=5)

    # 3. Fit scaler EXCLUSIVELY on train data
    print("[Pipeline] Fitting StandardScaler exclusively on training partition...")
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)

    # 4. Transform features
    train_scaled_vals = scaler.transform(train_df[FEATURE_COLUMNS].values)
    test_scaled_vals = scaler.transform(test_df[FEATURE_COLUMNS].values)

    for idx, col in enumerate(FEATURE_COLUMNS):
        train_df[col] = train_scaled_vals[:, idx]
        test_df[col] = test_scaled_vals[:, idx]

    # 5. Save artifacts
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    features_meta_path = os.path.join(output_dir, "feature_columns.json")
    train_parquet_path = os.path.join(output_dir, "train_features.parquet")
    test_parquet_path = os.path.join(output_dir, "student_test_features.parquet")

    joblib.dump(scaler, scaler_path)
    with open(features_meta_path, "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "n_features": len(FEATURE_COLUMNS),
                "top_k": top_k,
                "search_radius_meters": search_radius,
            },
            f,
            indent=2,
        )

    print(f"[Pipeline] Saving {train_parquet_path} ({len(train_df)} rows)...")
    train_df.to_parquet(train_parquet_path, index=False)

    print(f"[Pipeline] Saving {test_parquet_path} ({len(test_df)} rows)...")
    test_df.to_parquet(test_parquet_path, index=False)

    summary = {
        "train_stats": train_stats,
        "student_test_stats": test_stats,
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "train_rows": len(train_df),
        "student_test_rows": len(test_df),
        "class_balance_train": {
            "label_1": int((train_df["label"] == 1).sum()),
            "label_0": int((train_df["label"] == 0).sum()),
            "pos_ratio": round(float((train_df["label"] == 1).mean()), 4),
        },
        "class_balance_student_test": {
            "label_1": int((test_df["label"] == 1).sum()),
            "label_0": int((test_df["label"] == 0).sum()),
            "pos_ratio": round(float((test_df["label"] == 1).mean()), 4),
        },
        "cv_folds_distribution": train_df.groupby("cv_fold")["trip_id"].nunique().to_dict(),
    }

    return summary


def transform_single_trip(
    trip_dir: str,
    scaler: Optional[StandardScaler] = None,
    feature_cols: Optional[List[str]] = None,
    top_k: int = 7,
    search_radius: float = 80.0,
) -> pd.DataFrame:
    """
    Production-ready inference function for FastAPI backend.
    Directly transforms a trip directory into model-ready candidate features.
    Does NOT require or access .route ground truth.
    """
    if scaler is None:
        scaler = joblib.load("processed_data/scaler.joblib")
    if feature_cols is None:
        with open("processed_data/feature_columns.json", "r") as f:
            feature_cols = json.load(f)["feature_columns"]

    df, _ = extract_features_single_trip(
        trip_dir,
        search_radius=search_radius,
        top_k=top_k,
        has_ground_truth=False,
    )

    scaled_matrix = scaler.transform(df[feature_cols].values)
    for idx, col in enumerate(feature_cols):
        df[col] = scaled_matrix[:, idx]

    return df


if __name__ == "__main__":
    fit_and_save_pipeline()
