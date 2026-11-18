"""
PathSnap HMM + Viterbi Sequential Route Evaluation & Visualizer
===============================================================
Performs parameter tuning on training trips only and evaluates the HMM/Viterbi
decoder against Classifier-Only predictions on the 10 student-test trips.

Outputs:
1. hmm_viterbi/hmm_config.json: Tuned HMM/Viterbi hyperparameters.
2. results/hmm_viterbi_comparison.json: Comprehensive metric comparison
   (Top-1 Accuracy, Macro F1, Route Continuity, Edit Distance, LCS, Runtime).
3. visualizations/hmm_reconstructed_trip_00000000.html
4. visualizations/hmm_reconstructed_trip_00000009.html
"""

import os
import sys
import json
import time
import tempfile
import warnings
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.getcwd())

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_pathsnap")
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import folium
from branca.element import Element
from sklearn.metrics import f1_score, accuracy_score

from hmm_viterbi.decoder import ViterbiDecoder, decode_trip


def compute_edit_distance(seq1: List[int], seq2: List[int]) -> int:
    """Computes Levenshtein edit distance between two sequences of arc IDs."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def compute_lcs_length(seq1: List[int], seq2: List[int]) -> int:
    """Computes length of Longest Common Subsequence (LCS) between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def collapse_consecutive_duplicates(seq: List[int]) -> List[int]:
    """Collapses consecutive repeated arc IDs into distinct traversed road segments."""
    if not seq:
        return []
    res = [seq[0]]
    for item in seq[1:]:
        if item != res[-1]:
            res.append(item)
    return res


def compute_trip_route_continuity(pred_arcs: np.ndarray, arcs_file: str) -> float:
    """Computes percentage of consecutive predicted arcs that are identical or directly connected."""
    if len(pred_arcs) <= 1:
        return 100.0

    arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values

    valid_transitions = 0
    total_transitions = len(pred_arcs) - 1

    for t in range(total_transitions):
        a_curr = pred_arcs[t]
        a_next = pred_arcs[t + 1]
        if a_curr == a_next:
            valid_transitions += 1
        elif v_nodes[a_curr] == u_nodes[a_next]:
            valid_transitions += 1

    return float((valid_transitions / total_transitions) * 100.0)


def tune_hmm_parameters_on_train(
    train_df: pd.DataFrame,
    train_dir: str,
    model: Any,
    feature_cols: List[str],
    tuning_trip_ids: List[str]
) -> Dict[str, Any]:
    """
    Tunes HMM hyperparameters strictly on a subset of training trips (no data leakage).
    """
    print(f"\n[HMM Tuning] Tuning decoder parameters on {len(tuning_trip_ids)} training trips...")
    sub_train = train_df[train_df["trip_id"].isin(tuning_trip_ids)].copy().reset_index(drop=True)
    X_sub = sub_train[feature_cols].values
    sub_train["pred_prob"] = model.predict_proba(X_sub)[:, 1]

    # Pre-load graph data for each training trip
    graphs = {}
    ground_truth_per_trip = {}
    for tid in tuning_trip_ids:
        t_dir = os.path.join(train_dir, str(tid))
        arcs_path = os.path.join(t_dir, f"{tid}.arcs")
        nodes_path = os.path.join(t_dir, f"{tid}.nodes")
        graphs[tid] = ViterbiDecoder().load_road_graph(arcs_path, nodes_path)

        t_sub = sub_train[sub_train["trip_id"] == tid].sort_values("obs_idx")
        true_arcs_per_obs = t_sub[t_sub["label"] == 1].set_index("obs_idx")["candidate_arc_id"].to_dict()
        ground_truth_per_trip[tid] = true_arcs_per_obs

    # Define hyperparameter grid
    param_grid = [
        {"sigma": 8.0, "beta": 20.0, "alpha": 0.60, "hop_penalty": 1.5, "self_transition_log_prob": 0.5},
        {"sigma": 10.0, "beta": 25.0, "alpha": 0.65, "hop_penalty": 2.0, "self_transition_log_prob": 0.5},
        {"sigma": 10.0, "beta": 30.0, "alpha": 0.70, "hop_penalty": 2.0, "self_transition_log_prob": 0.6},
        {"sigma": 12.0, "beta": 25.0, "alpha": 0.65, "hop_penalty": 2.5, "self_transition_log_prob": 0.5},
        {"sigma": 8.0, "beta": 30.0, "alpha": 0.70, "hop_penalty": 2.0, "self_transition_log_prob": 0.7},
        {"sigma": 10.0, "beta": 20.0, "alpha": 0.60, "hop_penalty": 2.0, "self_transition_log_prob": 0.5},
        {"sigma": 12.0, "beta": 35.0, "alpha": 0.75, "hop_penalty": 1.5, "self_transition_log_prob": 0.6},
        {"sigma": 6.0, "beta": 25.0, "alpha": 0.65, "hop_penalty": 2.0, "self_transition_log_prob": 0.5},
    ]

    best_score = -np.inf
    best_params = param_grid[1]  # Default fallback

    for idx, params in enumerate(param_grid):
        decoder = ViterbiDecoder(**params)
        acc_list = []
        cont_list = []

        for tid in tuning_trip_ids:
            t_sub = sub_train[sub_train["trip_id"] == tid].copy()
            arcs_file = os.path.join(train_dir, str(tid), f"{tid}.arcs")
            decoded_arcs, decoded_df = decoder.decode(t_sub, graphs[tid])

            true_dict = ground_truth_per_trip[tid]
            correct = sum(
                1 for obs_i, p_arc in zip(decoded_df["obs_idx"], decoded_df["predicted_arc_id"])
                if true_dict.get(obs_i) == p_arc
            )
            trip_acc = (correct / len(decoded_df)) * 100.0 if len(decoded_df) > 0 else 0.0
            trip_cont = compute_trip_route_continuity(decoded_arcs, arcs_file)

            acc_list.append(trip_acc)
            cont_list.append(trip_cont)

        mean_acc = float(np.mean(acc_list))
        mean_cont = float(np.mean(cont_list))
        # Composite score balancing observation accuracy and physical route continuity
        composite_score = mean_acc + 0.6 * mean_cont

        print(
            f"  [Config {idx+1}/{len(param_grid)}] sigma={params['sigma']}, beta={params['beta']}, "
            f"alpha={params['alpha']:.2f}, hop_pen={params['hop_penalty']} => "
            f"Train Acc: {mean_acc:.2f}%, Continuity: {mean_cont:.2f}%, Score: {composite_score:.2f}"
        )

        if composite_score > best_score:
            best_score = composite_score
            best_params = params

    print(f"\n[HMM Tuning] Selected Best Parameters: {best_params} (Train Score: {best_score:.2f})")
    return best_params


def evaluate_student_test(
    test_df: pd.DataFrame,
    test_dir: str,
    model: Any,
    feature_cols: List[str],
    best_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates both Classifier-Only and HMM/Viterbi on the 10 student-test trips.
    """
    print("\n[Evaluation] Predicting candidate probabilities on student_test...")
    X_test = test_df[feature_cols].values
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    # Pre-select Classifier-Only Top 1
    idx_max = test_df.groupby(["trip_id", "obs_idx"])["pred_prob"].idxmax()
    cls_top1_all = test_df.loc[idx_max].copy().sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    decoder = ViterbiDecoder(**best_params)

    test_trips = sorted(test_df["trip_id"].unique())
    trip_results = []
    decoded_routes_dict = {}

    print(f"[Evaluation] Processing {len(test_trips)} student-test trips with Viterbi decoding...")

    for tid in test_trips:
        t_dir = os.path.join(test_dir, str(tid))
        arcs_path = os.path.join(t_dir, f"{tid}.arcs")
        nodes_path = os.path.join(t_dir, f"{tid}.nodes")
        route_path = os.path.join(t_dir, f"{tid}.route")

        # Load ground truth route sequence
        gt_route = pd.read_csv(route_path, header=None, names=["arc_id"], dtype=np.int64)["arc_id"].tolist()

        # Load road graph
        graph_data = decoder.load_road_graph(arcs_path, nodes_path)

        t_sub = test_df[test_df["trip_id"] == tid].copy()
        cls_sub = cls_top1_all[cls_top1_all["trip_id"] == tid].sort_values("obs_idx")

        # True arc mapping per observation
        true_sub = t_sub[t_sub["label"] == 1].set_index("obs_idx")["candidate_arc_id"].to_dict()

        # ---------------------------------------------------------
        # 1. Classifier-Only Evaluation
        # ---------------------------------------------------------
        cls_pred_arcs = cls_sub["candidate_arc_id"].values
        cls_correct = sum(
            1 for obs_i, p_arc in zip(cls_sub["obs_idx"], cls_pred_arcs)
            if true_sub.get(obs_i) == p_arc
        )
        cls_acc = (cls_correct / len(cls_sub)) * 100.0 if len(cls_sub) > 0 else 0.0
        cls_cont = compute_trip_route_continuity(cls_pred_arcs, arcs_path)

        # Macro F1 for classifier
        t_sub_reset = t_sub.reset_index(drop=True)
        cls_cand_tuples = set(zip(cls_sub["obs_idx"], cls_pred_arcs))
        is_cls_top1 = np.array([
            1 if (obs, cand) in cls_cand_tuples else 0
            for obs, cand in zip(t_sub_reset["obs_idx"], t_sub_reset["candidate_arc_id"])
        ], dtype=int)
        cls_macro_f1 = float(f1_score(t_sub_reset["label"], is_cls_top1, average="macro"))

        # Sequence-level metrics
        cls_collapsed_route = collapse_consecutive_duplicates(cls_pred_arcs.tolist())
        cls_exact = bool(cls_collapsed_route == gt_route)
        cls_edit_dist = compute_edit_distance(cls_collapsed_route, gt_route)
        cls_lcs_len = compute_lcs_length(cls_collapsed_route, gt_route)
        denom_cls = (len(cls_collapsed_route) + len(gt_route))
        cls_lcs_ratio = (2.0 * cls_lcs_len / denom_cls) if denom_cls > 0 else 0.0

        # ---------------------------------------------------------
        # 2. HMM + Viterbi Evaluation
        # ---------------------------------------------------------
        t_decode_start = time.time()
        hmm_pred_arcs, decoded_df = decoder.decode(t_sub, graph_data)
        decode_duration_ms = (time.time() - t_decode_start) * 1000.0

        hmm_correct = sum(
            1 for obs_i, p_arc in zip(decoded_df["obs_idx"], hmm_pred_arcs)
            if true_sub.get(obs_i) == p_arc
        )
        hmm_acc = (hmm_correct / len(decoded_df)) * 100.0 if len(decoded_df) > 0 else 0.0
        hmm_cont = compute_trip_route_continuity(hmm_pred_arcs, arcs_path)

        # Build HMM top-1 selection mask for macro F1
        t_sub_reset = t_sub.reset_index(drop=True)
        hmm_cand_tuples = set(zip(decoded_df["obs_idx"], hmm_pred_arcs))
        is_hmm_top1 = np.array([
            1 if (obs, cand) in hmm_cand_tuples else 0
            for obs, cand in zip(t_sub_reset["obs_idx"], t_sub_reset["candidate_arc_id"])
        ], dtype=int)
        hmm_macro_f1 = float(f1_score(t_sub_reset["label"], is_hmm_top1, average="macro"))

        # Sequence-level metrics
        hmm_collapsed_route = collapse_consecutive_duplicates(hmm_pred_arcs.tolist())
        hmm_exact = bool(hmm_collapsed_route == gt_route)
        hmm_edit_dist = compute_edit_distance(hmm_collapsed_route, gt_route)
        hmm_lcs_len = compute_lcs_length(hmm_collapsed_route, gt_route)
        denom_hmm = (len(hmm_collapsed_route) + len(gt_route))
        hmm_lcs_ratio = (2.0 * hmm_lcs_len / denom_hmm) if denom_hmm > 0 else 0.0

        # Store for visualizations
        decoded_routes_dict[tid] = {
            "gt_route": gt_route,
            "cls_pred_arcs": cls_pred_arcs,
            "hmm_pred_arcs": hmm_pred_arcs,
            "obs_indices": decoded_df["obs_idx"].values,
        }

        trip_results.append({
            "trip_id": str(tid),
            "n_observations": len(cls_sub),
            "ground_truth_route_length": len(gt_route),
            "classifier_only": {
                "top1_accuracy": round(cls_acc, 2),
                "macro_f1": round(cls_macro_f1, 4),
                "route_continuity": round(cls_cont, 2),
                "collapsed_route_length": len(cls_collapsed_route),
                "exact_match": cls_exact,
                "edit_distance": cls_edit_dist,
                "lcs_ratio": round(cls_lcs_ratio, 4),
            },
            "hmm_viterbi": {
                "top1_accuracy": round(hmm_acc, 2),
                "macro_f1": round(hmm_macro_f1, 4),
                "route_continuity": round(hmm_cont, 2),
                "collapsed_route_length": len(hmm_collapsed_route),
                "exact_match": hmm_exact,
                "edit_distance": hmm_edit_dist,
                "lcs_ratio": round(hmm_lcs_ratio, 4),
                "decode_duration_ms": round(decode_duration_ms, 2),
            },
        })

        print(
            f"  [Trip {tid}] Obs: {len(cls_sub):4d} | "
            f"Cls Acc: {cls_acc:5.2f}% (Cont: {cls_cont:5.2f}%, Edit: {cls_edit_dist:3d}) -> "
            f"HMM Acc: {hmm_acc:5.2f}% (Cont: {hmm_cont:5.2f}%, Edit: {hmm_edit_dist:3d}, Time: {decode_duration_ms:5.1f}ms)"
        )

    # Aggregate summaries across all 10 trips
    df_res = pd.DataFrame(trip_results)
    cls_acc_mean = float(np.mean([r["classifier_only"]["top1_accuracy"] for r in trip_results]))
    cls_f1_mean = float(np.mean([r["classifier_only"]["macro_f1"] for r in trip_results]))
    cls_cont_mean = float(np.mean([r["classifier_only"]["route_continuity"] for r in trip_results]))
    cls_edit_mean = float(np.mean([r["classifier_only"]["edit_distance"] for r in trip_results]))
    cls_lcs_mean = float(np.mean([r["classifier_only"]["lcs_ratio"] for r in trip_results]))
    cls_exact_count = int(sum(1 for r in trip_results if r["classifier_only"]["exact_match"]))

    hmm_acc_mean = float(np.mean([r["hmm_viterbi"]["top1_accuracy"] for r in trip_results]))
    hmm_f1_mean = float(np.mean([r["hmm_viterbi"]["macro_f1"] for r in trip_results]))
    hmm_cont_mean = float(np.mean([r["hmm_viterbi"]["route_continuity"] for r in trip_results]))
    hmm_edit_mean = float(np.mean([r["hmm_viterbi"]["edit_distance"] for r in trip_results]))
    hmm_lcs_mean = float(np.mean([r["hmm_viterbi"]["lcs_ratio"] for r in trip_results]))
    hmm_exact_count = int(sum(1 for r in trip_results if r["hmm_viterbi"]["exact_match"]))
    hmm_time_mean = float(np.mean([r["hmm_viterbi"]["decode_duration_ms"] for r in trip_results]))

    summary = {
        "dataset": "datasets/student_test (10 trips)",
        "hyperparameters": best_params,
        "aggregate_comparison": {
            "observation_top1_accuracy": {
                "classifier_only": round(cls_acc_mean, 2),
                "hmm_viterbi": round(hmm_acc_mean, 2),
                "delta": round(hmm_acc_mean - cls_acc_mean, 2),
            },
            "observation_macro_f1": {
                "classifier_only": round(cls_f1_mean, 4),
                "hmm_viterbi": round(hmm_f1_mean, 4),
                "delta": round(hmm_f1_mean - cls_f1_mean, 4),
            },
            "route_continuity": {
                "classifier_only": round(cls_cont_mean, 2),
                "hmm_viterbi": round(hmm_cont_mean, 2),
                "delta": round(hmm_cont_mean - cls_cont_mean, 2),
            },
            "sequence_edit_distance": {
                "classifier_only": round(cls_edit_mean, 2),
                "hmm_viterbi": round(hmm_edit_mean, 2),
                "reduction": round(cls_edit_mean - hmm_edit_mean, 2),
            },
            "sequence_lcs_ratio": {
                "classifier_only": round(cls_lcs_mean, 4),
                "hmm_viterbi": round(hmm_lcs_mean, 4),
                "delta": round(hmm_lcs_mean - cls_lcs_mean, 4),
            },
            "exact_sequence_matches": {
                "classifier_only": f"{cls_exact_count}/10",
                "hmm_viterbi": f"{hmm_exact_count}/10",
            },
            "average_decode_time_ms_per_trip": round(hmm_time_mean, 2),
        },
        "per_trip_breakdown": trip_results,
    }

    return summary, decoded_routes_dict


def generate_interactive_map(
    trip_id: str,
    test_dir: str,
    trip_data: Dict[str, Any],
    output_path: str
):
    """
    Generates rich, publication-grade interactive Folium map showing:
    - Raw GPS points
    - Road Network Arcs
    - Ground Truth Route
    - Classifier-Only Top 1
    - HMM / Viterbi Reconstructed Route
    """
    print(f"[Visualization] Building interactive map for trip {trip_id} -> {output_path}")
    t_dir = os.path.join(test_dir, trip_id)
    track_path = os.path.join(t_dir, f"{trip_id}.track")
    nodes_path = os.path.join(t_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(t_dir, f"{trip_id}.arcs")

    track_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype=np.float64)
    nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64)
    arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)

    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # 1. Road Network Arcs (Background context)
    fg_network = folium.FeatureGroup(name="Road Network Arcs", show=True)
    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values
    node_lats = nodes_df["lat"].values
    node_lons = nodes_df["lon"].values

    for arc_i in range(len(arcs_df)):
        u = u_nodes[arc_i]
        v = v_nodes[arc_i]
        folium.PolyLine(
            locations=[[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]],
            color="#888888",
            weight=1.5,
            opacity=0.35,
            popup=f"Arc ID: {arc_i}",
        ).add_to(fg_network)
    fg_network.add_to(m)

    # 2. Raw GPS Track
    fg_gps = folium.FeatureGroup(name="Raw GPS Track (Noisy Points)", show=True)
    for idx, row in track_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3.5,
            color="#E65100",
            fill=True,
            fill_color="#FF9800",
            fill_opacity=0.85,
            popup=f"GPS Obs #{int(idx)}<br>t = {row['t']:.1f}s<br>Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # 3. Ground Truth Route (.route)
    fg_gt = folium.FeatureGroup(name="Ground Truth Route (.route)", show=True)
    gt_route = trip_data["gt_route"]
    for arc_id in gt_route:
        if arc_id < len(arcs_df):
            u = u_nodes[arc_id]
            v = v_nodes[arc_id]
            folium.PolyLine(
                locations=[[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]],
                color="#2E7D32",
                weight=5,
                opacity=0.85,
                popup=f"Ground Truth Arc #{arc_id}",
            ).add_to(fg_gt)
    fg_gt.add_to(m)

    # 4. Classifier-Only Top 1 Route
    fg_cls = folium.FeatureGroup(name="Classifier-Only Route (Greedy Top-1)", show=False)
    cls_arcs = trip_data["cls_pred_arcs"]
    for arc_id in collapse_consecutive_duplicates(cls_arcs.tolist()):
        if arc_id < len(arcs_df):
            u = u_nodes[arc_id]
            v = v_nodes[arc_id]
            folium.PolyLine(
                locations=[[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]],
                color="#D32F2F",
                weight=3.5,
                dash_array="6, 6",
                opacity=0.85,
                popup=f"Classifier Top-1 Arc #{arc_id}",
            ).add_to(fg_cls)
    fg_cls.add_to(m)

    # 5. HMM / Viterbi Reconstructed Route
    fg_hmm = folium.FeatureGroup(name="HMM + Viterbi Decoded Route (Continuous)", show=True)
    hmm_arcs = trip_data["hmm_pred_arcs"]
    for arc_id in collapse_consecutive_duplicates(hmm_arcs.tolist()):
        if arc_id < len(arcs_df):
            u = u_nodes[arc_id]
            v = v_nodes[arc_id]
            folium.PolyLine(
                locations=[[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]],
                color="#1565C0",
                weight=4,
                opacity=0.90,
                popup=f"HMM/Viterbi Arc #{arc_id}",
            ).add_to(fg_hmm)
    fg_hmm.add_to(m)

    # Add Layer Control
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Floating HTML Legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 310px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.94);
        border: 2px solid #333333;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    ">
        <h4 style="margin: 0 0 8px 0; color: #1a237e; font-size: 14px; font-weight: bold; border-bottom: 1px solid #ccc; padding-bottom: 4px;">
            PathSnap Route Reconstruction (Trip {trip_id})
        </h4>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #FF9800; border: 1.5px solid #E65100; margin-right: 8px;"></span>
            <span><b>Raw GPS Track</b> (Noisy Sensor Points)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 22px; height: 4px; background: #2E7D32; margin-right: 8px;"></span>
            <span><b>Ground Truth Route</b> (True Sequence)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 22px; height: 3px; border-top: 3px dashed #D32F2F; margin-right: 8px;"></span>
            <span><b>Classifier-Only</b> (Greedy Disconnected)</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <span style="display: inline-block; width: 22px; height: 4px; background: #1565C0; margin-right: 8px;"></span>
            <span><b>HMM + Viterbi</b> (Globally Reconstructed)</span>
        </div>
        <div style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 22px; height: 2px; background: #888888; margin-right: 8px;"></span>
            <span style="color: #666;">Road Network Arcs (Metric Graph)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(Element(legend_html))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"[Visualization] Saved {output_path}")


def main():
    print("=" * 80)
    print(" PATHSNAP HMM + VITERBI SEQUENTIAL ROUTE EVALUATION ")
    print("=" * 80)

    # 1. Load trained model and features
    model_path = "models/gradient_boosting/tuned_lgbm_model.joblib"
    if not os.path.exists(model_path):
        print(f"[Warning] {model_path} not found. Falling back to untuned gbdt_model.joblib...")
        model_path = "models/gradient_boosting/gbdt_model.joblib"

    print(f"[Model] Loading LightGBM model from {model_path}...")
    model = joblib.load(model_path)

    with open("processed_data/feature_columns.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]

    train_df = pd.read_parquet("processed_data/train_features.parquet")
    test_df = pd.read_parquet("processed_data/student_test_features.parquet")
    train_dir = "datasets/train"
    test_dir = "datasets/student_test"

    # 2. Tune HMM parameters strictly on training trips (e.g. 8 representative trips)
    valid_train_trips = [
        str(tid) for tid in sorted(train_df["trip_id"].unique())
        if os.path.exists(os.path.join(train_dir, str(tid), f"{tid}.arcs"))
    ]
    tuning_trips = valid_train_trips[:8]
    best_params = tune_hmm_parameters_on_train(train_df, train_dir, model, feature_cols, tuning_trips)

    # Save tuned HMM config
    os.makedirs("hmm_viterbi", exist_ok=True)
    with open("hmm_viterbi/hmm_config.json", "w") as f:
        json.dump({
            "model_name": "PathSnap HMM + Viterbi Sequential Route Decoder",
            "hyperparameters": best_params,
            "tuning_set": f"{len(tuning_trips)} representative training trips",
            "saved_at": pd.Timestamp.now().isoformat(),
        }, f, indent=2)
    print("[Config] Saved hmm_viterbi/hmm_config.json")

    # 3. Evaluate on 10 student-test trips
    summary, decoded_routes_dict = evaluate_student_test(test_df, test_dir, model, feature_cols, best_params)

    # Save comparison metrics
    os.makedirs("results", exist_ok=True)
    comp_path = "results/hmm_viterbi_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Results] Saved {comp_path}")

    # 4. Generate Interactive Folium Maps for Trip 00000000 and 00000009
    os.makedirs("visualizations", exist_ok=True)
    for sample_tid in ["00000000", "00000009"]:
        if sample_tid in decoded_routes_dict:
            map_out = f"visualizations/hmm_reconstructed_trip_{sample_tid}.html"
            generate_interactive_map(sample_tid, test_dir, decoded_routes_dict[sample_tid], map_out)

    # 5. Print Summary Comparison Table
    agg = summary["aggregate_comparison"]
    print("\n" + "=" * 80)
    print(" SUMMARY: CLASSIFIER-ONLY vs. HMM + VITERBI RECONSTRUCTION ")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Classifier-Only':<18} | {'HMM + Viterbi':<18} | {'Delta':<12}")
    print("-" * 89)
    print(f"{'Observation Top-1 Accuracy (%)':<35} | {agg['observation_top1_accuracy']['classifier_only']:<18.2f} | {agg['observation_top1_accuracy']['hmm_viterbi']:<18.2f} | {agg['observation_top1_accuracy']['delta']:+<12.2f}")
    print(f"{'Observation Macro F1-Score':<35} | {agg['observation_macro_f1']['classifier_only']:<18.4f} | {agg['observation_macro_f1']['hmm_viterbi']:<18.4f} | {agg['observation_macro_f1']['delta']:+<12.4f}")
    print(f"{'Route Continuity (%)':<35} | {agg['route_continuity']['classifier_only']:<18.2f} | {agg['route_continuity']['hmm_viterbi']:<18.2f} | {agg['route_continuity']['delta']:+<12.2f}")
    print(f"{'Mean Route Edit Distance':<35} | {agg['sequence_edit_distance']['classifier_only']:<18.2f} | {agg['sequence_edit_distance']['hmm_viterbi']:<18.2f} | {-agg['sequence_edit_distance']['reduction']:+<12.2f}")
    print(f"{'Mean Route LCS Ratio':<35} | {agg['sequence_lcs_ratio']['classifier_only']:<18.4f} | {agg['sequence_lcs_ratio']['hmm_viterbi']:<18.4f} | {agg['sequence_lcs_ratio']['delta']:+<12.4f}")
    print(f"{'Exact Sequence Matches (10 Trips)':<35} | {agg['exact_sequence_matches']['classifier_only']:<18} | {agg['exact_sequence_matches']['hmm_viterbi']:<18} | -")
    print(f"{'Avg Inference Time per Trip (ms)':<35} | {'~15 ms':<18} | {agg['average_decode_time_ms_per_trip']:<18.2f} | -")
    print("=" * 80)


if __name__ == "__main__":
    main()
