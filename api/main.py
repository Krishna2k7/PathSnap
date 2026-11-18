"""
main.py - FastAPI Backend Service for PathSnap (Laptop 1 Server).
Exposes endpoints for trip discovery, distributed route reconstruction inference,
and server-side evaluation against reference ground truth.
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.parser import load_trip_data
from src.preprocessing.projection import LocalMetricProjector
from src.preprocessing.feature_engine import extract_trip_candidate_features
from src.models.random_forest import PathSnapRandomForest
from src.models.gradient_boosting import PathSnapGradientBoosting
from src.hmm.transition import NetworkTransitionModel
from src.hmm.viterbi import ViterbiDecoder, compute_path_continuity

# Initialize loaded model caches
MODELS_CACHE = {}


def get_model(model_type: str = "random_forest"):
    """Loads and caches trained model instances."""
    if model_type not in MODELS_CACHE:
        if model_type == "random_forest":
            model_path = PROJECT_ROOT / "models" / "random_forest.joblib"
            MODELS_CACHE[model_type] = PathSnapRandomForest.load(model_path)
        elif model_type == "gradient_boosting":
            model_path = PROJECT_ROOT / "models" / "gradient_boosting.joblib"
            MODELS_CACHE[model_type] = PathSnapGradientBoosting.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return MODELS_CACHE[model_type]


class ReconstructionResponse(BaseModel):
    trip_id: str
    model_name: str
    num_observations: int
    num_candidates: int
    inference_time_ms: float
    route_continuity_pct: float
    pointwise_accuracy_pct: Optional[float]
    predicted_path_length: int
    predicted_coordinates: List[List[float]]  # [[lat, lon], ...]


def execute_pipeline(trip_id: str, model_type: str = "random_forest") -> Dict[str, object]:
    """Executes the full map-matching and reconstruction pipeline for a given trip."""
    # Check student_test or train
    trip_dir = PROJECT_ROOT / "datasets" / "student_test" / trip_id
    if not trip_dir.exists():
        trip_dir = PROJECT_ROOT / "datasets" / "train" / trip_id
    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip {trip_id} not found in datasets.")

    t_start = time.time()
    trip_data = load_trip_data(trip_dir)
    track_df = trip_data["track"]
    nodes_df = trip_data["nodes"]
    arcs_df = trip_data["arcs"]

    lon0 = float(track_df["lon"].mean())
    lat0 = float(track_df["lat"].mean())
    projector = LocalMetricProjector(lon0, lat0)
    t_x, t_y = projector.project(track_df["lon"].values, track_df["lat"].values)
    n_x, n_y = projector.project(nodes_df["lon"].values, nodes_df["lat"].values)
    gps_coords_m = [t_x, t_y]

    feat_df = extract_trip_candidate_features(trip_data)

    model = get_model(model_type)
    scores = model.predict_candidate_scores(feat_df)
    feat_df["score"] = scores

    observations_candidates = []
    for obs_id, group in feat_df.groupby("obs_id"):
        g_scores = group["score"].values
        g_arcs = group["cand_arc_id"].values.astype(int)
        exp_s = [float(s) for s in g_scores]
        max_s = max(exp_s)
        probs = [math.exp(s - max_s) for s in exp_s]
        sum_p = sum(probs) + 1e-7
        cands = [{"arc_id": int(a), "prob": float(p / sum_p)} for a, p in zip(g_arcs, probs)]
        observations_candidates.append(cands)

    transition_model = NetworkTransitionModel(n_x, n_y, arcs_df, beta=15.0)
    decoder = ViterbiDecoder(transition_model, emission_weight=1.5)
    decoded_path = decoder.decode(observations_candidates, [t_x, t_y])

    continuity = compute_path_continuity(decoded_path, arcs_df)
    total_time_ms = (time.time() - t_start) * 1000.0

    # Server-side accuracy calculation against ground truth if present
    acc = None
    if "is_true_arc" in feat_df.columns and feat_df["is_true_arc"].sum() > 0:
        true_dict = {
            int(r["obs_id"]): int(r["cand_arc_id"])
            for _, r in feat_df[feat_df["is_true_arc"] == 1].iterrows()
        }
        correct = sum(1 for t, a in enumerate(decoded_path) if true_dict.get(t) == a)
        acc = (correct / len(decoded_path)) * 100.0

    # Build decoded geographic coordinates [[lat, lon], ...]
    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values
    node_lats = nodes_df["lat"].values
    node_lons = nodes_df["lon"].values

    coords = []
    for arc_id in decoded_path:
        u, v = u_nodes[arc_id], v_nodes[arc_id]
        coords.append([float(node_lats[u]), float(node_lons[u])])
    if decoded_path:
        last_v = v_nodes[decoded_path[-1]]
        coords.append([float(node_lats[last_v]), float(node_lons[last_v])])

    return {
        "trip_id": trip_id,
        "model_name": model_type,
        "num_observations": len(track_df),
        "num_candidates": len(feat_df),
        "inference_time_ms": round(total_time_ms, 2),
        "route_continuity_pct": round(continuity, 2),
        "pointwise_accuracy_pct": round(acc, 2) if acc is not None else None,
        "predicted_path_length": len(decoded_path),
        "predicted_coordinates": coords,
    }
