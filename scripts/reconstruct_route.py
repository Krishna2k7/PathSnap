"""
reconstruct_route.py - Full-route reconstruction and evaluation script.
Compares Classifier-Only predictions vs. Classifier + HMM / Viterbi decoding
against ground-truth reference routes.
"""

from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.parser import load_trip_data
from src.preprocessing.projection import LocalMetricProjector
from src.preprocessing.feature_engine import extract_trip_candidate_features
from src.models.gradient_boosting import PathSnapGradientBoosting
from src.hmm.transition import NetworkTransitionModel
from src.hmm.viterbi import ViterbiDecoder, compute_path_continuity


def reconstruct_trip(trip_id: str, partition: str = "student_test", model: PathSnapGradientBoosting = None):
    trip_dir = PROJECT_ROOT / "datasets" / partition / trip_id
    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    trip_data = load_trip_data(trip_dir)
    track_df = trip_data["track"]
    nodes_df = trip_data["nodes"]
    arcs_df = trip_data["arcs"]
    route = trip_data.get("route")

    # Local metric projection for HMM Euclidean step distances
    lon0 = float(track_df["lon"].mean())
    lat0 = float(track_df["lat"].mean())
    projector = LocalMetricProjector(lon0, lat0)
    t_x, t_y = projector.project(track_df["lon"].values, track_df["lat"].values)
    n_x, n_y = projector.project(nodes_df["lon"].values, nodes_df["lat"].values)
    gps_coords_m = np.column_stack([t_x, t_y])

    # 1. Feature extraction
    t0 = time.time()
    feat_df = extract_trip_candidate_features(trip_data)
    feat_time = time.time() - t0

    # 2. Model inference
    if model is None:
        model_path = PROJECT_ROOT / "models" / "gradient_boosting.joblib"
        model = PathSnapGradientBoosting.load(model_path)

    t0 = time.time()
    cand_scores = model.predict_candidate_scores(feat_df)
    feat_df["score"] = cand_scores
    model_time = time.time() - t0

    # 3. Method A: Classifier-Only (Top-1)
    t0 = time.time()
    classifier_only_arcs = []
    observations_candidates = []

    for obs_id, group in feat_df.groupby("obs_id"):
        # Top-1 candidate
        best_arc = int(group.loc[group["score"].idxmax()]["cand_arc_id"])
        classifier_only_arcs.append(best_arc)

        # Structure candidates for HMM
        # Softmax / normalize scores across candidates for observation
        scores = group["score"].values
        arc_ids = group["cand_arc_id"].values.astype(int)
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / (np.sum(exp_s) + 1e-7)

        cands = [{"arc_id": int(a), "prob": float(p)} for a, p in zip(arc_ids, probs)]
        observations_candidates.append(cands)

    cls_time = time.time() - t0

    # 4. Method B: Classifier + HMM / Viterbi Decoding
    t0 = time.time()
    transition_model = NetworkTransitionModel(n_x, n_y, arcs_df, beta=15.0)
    viterbi_decoder = ViterbiDecoder(transition_model, emission_weight=1.5)
    hmm_decoded_arcs = viterbi_decoder.decode(observations_candidates, gps_coords_m)
    hmm_time = time.time() - t0

    # 5. Route Continuity evaluation
    continuity_cls = compute_path_continuity(classifier_only_arcs, arcs_df)
    continuity_hmm = compute_path_continuity(hmm_decoded_arcs, arcs_df)

    # 6. Pointwise accuracy against ground-truth route
    cls_acc = None
    hmm_acc = None
    if "is_true_arc" in feat_df.columns and feat_df["is_true_arc"].sum() > 0:
        true_dict = {}
        for _, row in feat_df[feat_df["is_true_arc"] == 1].iterrows():
            true_dict[int(row["obs_id"])] = int(row["cand_arc_id"])

        T = len(classifier_only_arcs)
        cls_correct = sum(1 for t, a in enumerate(classifier_only_arcs) if true_dict.get(t) == a)
        hmm_correct = sum(1 for t, a in enumerate(hmm_decoded_arcs) if true_dict.get(t) == a)
        cls_acc = (cls_correct / T) * 100.0
        hmm_acc = (hmm_correct / T) * 100.0

    result = {
        "trip_id": trip_id,
        "num_observations": len(track_df),
        "num_candidates": len(feat_df),
        "feature_time_s": feat_time,
        "model_infer_time_s": model_time,
        "viterbi_time_s": hmm_time,
        "classifier_only_continuity_pct": continuity_cls,
        "hmm_continuity_pct": continuity_hmm,
        "classifier_only_accuracy_pct": cls_acc,
        "hmm_accuracy_pct": hmm_acc,
        "classifier_only_path": classifier_only_arcs[:10],
        "hmm_decoded_path": hmm_decoded_arcs[:10],
        "full_classifier_path": classifier_only_arcs,
        "full_hmm_path": hmm_decoded_arcs,
    }

    print(f"\n=== Route Reconstruction Results: Trip {trip_id} ({partition}) ===")
    print(f"  GPS Observations: {len(track_df)} | Total Candidates: {len(feat_df)}")
    print(f"  Feature Time: {feat_time:.2f}s | Model Infer Time: {model_time:.3f}s | Viterbi Time: {hmm_time:.2f}s")
    if cls_acc is not None:
        print(f"  Pointwise Accuracy: Classifier-Only = {cls_acc:.2f}%  -->  Classifier + HMM = {hmm_acc:.2f}%")
    print(f"  Route Continuity:   Classifier-Only = {continuity_cls:.2f}%  -->  Classifier + HMM = {continuity_hmm:.2f}%")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-id", type=str, default="00000000")
    parser.add_argument("--partition", type=str, default="student_test")
    args = parser.parse_args()

    reconstruct_trip(args.trip_id, args.partition)
