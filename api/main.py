"""
PathSnap FastAPI Production Backend Service (Laptop 1 Server)
=============================================================
Live inference executing our actual trained ML + HMM/Viterbi pipeline:
1. Spatial candidate generation & 21-feature extraction: features.pipeline.transform_single_trip
2. Tuned LightGBM classifier: models/gradient_boosting/tuned_lgbm_model.joblib
3. Sequential HMM/Viterbi route decoder: hmm_viterbi.decoder.ViterbiDecoder using hmm_viterbi/hmm_config.json

Strict Two-Laptop & Data Leakage Rules:
- Dataset Discovery: Evaluator test directory checked first (datasets/evaluator_test),
  fallback to datasets/student_test, or PATHSNAP_EVAL_DIR.
- Zero Ground-Truth Leakage: .route data is NEVER transmitted over the wire.
  If .route exists on the server, evaluation metrics (Accuracy, F1, Continuity, Edit Distance, LCS, Latency)
  are computed strictly server-side.
- Terminal Logging: Visible request logging with trip ID, point counts, inference latency, and metrics.
- LAN CORS & Host Binding: Bound to 0.0.0.0:8000 with permissive CORS.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from sklearn.metrics import f1_score, log_loss, brier_score_loss

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.pipeline import transform_single_trip, match_ground_truth_route
from hmm_viterbi.decoder import ViterbiDecoder
from hmm_viterbi.evaluate_hmm import (
    compute_edit_distance,
    compute_lcs_length,
    collapse_consecutive_duplicates,
    compute_trip_route_continuity,
)

# -----------------------------------------------------------------------------
# 1. Dataset Discovery & Path Resolution
# -----------------------------------------------------------------------------
def get_evaluation_base_dir() -> Path:
    """
    Discovers the active evaluation directory according to priority rules:
    1. PATHSNAP_EVAL_DIR environment variable (if explicitly set).
    2. datasets/evaluator_test (if mounted during final evaluation).
    3. datasets/student_test (default for local and pre-demo testing).
    4. datasets/train (fallback).
    """
    env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    eval_test = PROJECT_ROOT / "datasets" / "evaluator_test"
    if eval_test.exists():
        subdirs = [d for d in eval_test.iterdir() if d.is_dir()]
        if len(subdirs) > 0:
            return eval_test

    student_test = PROJECT_ROOT / "datasets" / "student_test"
    if student_test.exists():
        return student_test

    return PROJECT_ROOT / "datasets" / "train"


def find_trip_directory(trip_id: str) -> Optional[Path]:
    """Finds the directory containing raw trip data (.track, .nodes, .arcs)."""
    base_dir = get_evaluation_base_dir()
    direct = base_dir / str(trip_id)
    if direct.exists():
        return direct

    # Search in order of preference
    for search_root in [
        PROJECT_ROOT / "datasets" / "evaluator_test",
        PROJECT_ROOT / "datasets" / "student_test",
        PROJECT_ROOT / "datasets" / "train",
    ]:
        cand = search_root / str(trip_id)
        if cand.exists():
            return cand

    return None


# -----------------------------------------------------------------------------
# 2. Model & Pipeline Artifact Preloading
# -----------------------------------------------------------------------------
print("=" * 80)
print(" [PATHSNAP BACKEND] INITIALIZING PRODUCTION MACHINE LEARNING PIPELINE ")
print("=" * 80)

# Load feature metadata & scaler
scaler_path = PROJECT_ROOT / "processed_data" / "scaler.joblib"
feature_cols_path = PROJECT_ROOT / "processed_data" / "feature_columns.json"

if not scaler_path.exists() or not feature_cols_path.exists():
    raise FileNotFoundError(
        "Preprocessing artifacts not found in processed_data/. "
        "Ensure processed_data/scaler.joblib and feature_columns.json exist."
    )

SCALER = joblib.load(scaler_path)
with open(feature_cols_path, "r") as f:
    FEATURE_COLS = json.load(f)["feature_columns"]
print(f"[+] Loaded StandardScaler and {len(FEATURE_COLS)} feature column definitions.")

# Load Tuned LightGBM Model
TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting" / "tuned_lgbm_model.joblib"
if not TUNED_MODEL_PATH.exists():
    print("[!] Tuned LightGBM model not found. Falling back to untuned gbdt_model.joblib...")
    TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting" / "gbdt_model.joblib"

MODEL = joblib.load(TUNED_MODEL_PATH)
print(f"[+] Loaded Winning LightGBM Classifier from {TUNED_MODEL_PATH.name}")

# Load HMM / Viterbi Configuration
HMM_CONFIG_PATH = PROJECT_ROOT / "hmm_viterbi" / "hmm_config.json"
if HMM_CONFIG_PATH.exists():
    with open(HMM_CONFIG_PATH, "r") as f:
        hmm_meta = json.load(f)
        HMM_PARAMS = hmm_meta.get("hyperparameters", {})
    print(f"[+] Loaded Tuned HMM Parameters from hmm_viterbi/hmm_config.json: {HMM_PARAMS}")
else:
    HMM_PARAMS = {
        "sigma": 8.0,
        "beta": 30.0,
        "alpha": 0.70,
        "hop_penalty": 2.0,
        "self_transition_log_prob": 0.70,
    }
    print(f"[!] Using default HMM Parameters: {HMM_PARAMS}")

DECODER = ViterbiDecoder(**HMM_PARAMS)

eval_dir_initial = get_evaluation_base_dir()
print(f"[+] Active Evaluation Directory: {eval_dir_initial} ({eval_dir_initial.name})")
print("=" * 80)


# -----------------------------------------------------------------------------
# 3. FastAPI Application & Middleware
# -----------------------------------------------------------------------------
app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching & Sequential Route Reconstruction Service (Laptop 1 Server)",
    version="2.0.0",
)

# Enable CORS for Laptop 2 LAN connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# 4. Request / Response Schemas
# -----------------------------------------------------------------------------
class RouteGeometry(BaseModel):
    gps_coordinates: List[List[float]]  # [[lat, lon], ...]
    predicted_coordinates: List[List[float]]  # [[lat, lon], ...]


class EvaluationMetrics(BaseModel):
    route_continuity_pct: float
    observation_top1_accuracy_pct: Optional[float] = None
    hmm_route_accuracy_pct: Optional[float] = None
    candidate_macro_f1: Optional[float] = None
    candidate_miss_rate_pct: Optional[float] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    sequence_edit_distance: Optional[int] = None
    sequence_lcs_ratio: Optional[float] = None


class ReconstructionResponse(BaseModel):
    trip_id: str
    model_name: str
    num_observations: int
    num_candidates: int
    inference_time_ms: float
    metrics: EvaluationMetrics
    geometry: RouteGeometry
    predicted_arcs: List[int]


# -----------------------------------------------------------------------------
# 5. API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    """Serves the evaluator single-page web UI directly to any connected laptop."""
    react_dist = PROJECT_ROOT / "frontend" / "dist" / "index.html"
    if react_dist.exists():
        return FileResponse(react_dist, media_type="text/html")

    ui_path = PROJECT_ROOT / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path, media_type="text/html")

    return HTMLResponse("<h2>PathSnap UI not found. Please ensure ui/index.html exists.</h2>", status_code=404)


@app.get("/health")
def health_check():
    """Confirms backend service health and model readiness for Laptop 2."""
    eval_dir = get_evaluation_base_dir()
    return {
        "status": "healthy",
        "service": "PathSnap-API",
        "model_name": "Tuned LightGBM + HMM/Viterbi",
        "active_evaluation_dir": str(eval_dir.name),
        "feature_count": len(FEATURE_COLS),
        "timestamp": pd.Timestamp.now().isoformat(),
    }


@app.get("/trips")
def list_trips():
    """
    Lists all available evaluation trips from the active evaluation directory.
    Evaluator test directory is checked first, followed by student test.
    """
    eval_dir = get_evaluation_base_dir()
    if not eval_dir.exists():
        return {"trips": [], "active_evaluation_dir": str(eval_dir)}

    trips = sorted([d.name for d in eval_dir.iterdir() if d.is_dir()])
    return {
        "trips": trips,
        "count": len(trips),
        "active_evaluation_dir": str(eval_dir.name),
    }


@app.post("/reconstruct/{trip_id}", response_model=ReconstructionResponse)
@app.post("/api/v1/reconstruct/{trip_id}", response_model=ReconstructionResponse)
def reconstruct_route(
    trip_id: str,
    model_type: str = Query("gradient_boosting", description="Model architecture identifier"),
):
    """
    Executes live map-matching and sequential route reconstruction:
    1. Transforms trip with features.pipeline.transform_single_trip (NO .route access).
    2. Computes ML candidate emission scores using tuned LightGBM.
    3. Decodes globally optimal sequence via Viterbi dynamic programming.
    4. Computes server-side verification metrics if server-side .route reference exists.
    5. Returns predicted route coordinates and metrics. ZERO .route data is leaked.
    """
    t_start = time.time()
    trip_dir = find_trip_directory(trip_id)

    if trip_dir is None:
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in active evaluation datasets.")

    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    if not track_path.exists() or not nodes_path.exists() or not arcs_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Incomplete trip files for '{trip_id}'. Required: .track, .nodes, .arcs"
        )

    # 1. Feature Engineering & Preprocessing (Zero .route leakage)
    try:
        cand_df = transform_single_trip(
            str(trip_dir),
            scaler=SCALER,
            feature_cols=FEATURE_COLS,
            top_k=7,
            search_radius=80.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    if len(cand_df) == 0:
        raise HTTPException(status_code=400, detail="No spatial candidates could be generated for trajectory.")

    # 2. Supervised Model Scoring
    X = cand_df[FEATURE_COLS].values
    pred_probs = MODEL.predict_proba(X)[:, 1]
    cand_df["pred_prob"] = pred_probs

    # 3. HMM + Viterbi Sequential Route Decoding
    graph_data = DECODER.load_road_graph(str(arcs_path), str(nodes_path))
    decoded_arcs, decoded_df = DECODER.decode(cand_df, graph_data)

    # 4. Physical Route Continuity
    continuity_pct = compute_trip_route_continuity(decoded_arcs, str(arcs_path))

    # 5. Server-Side Reference Evaluation (STRICTLY SERVER-SIDE)
    # Ground truth reference is evaluated here if present on disk, but never exposed to client.
    metrics_dict = {
        "route_continuity_pct": round(continuity_pct, 2),
        "observation_top1_accuracy_pct": None,
        "hmm_route_accuracy_pct": None,
        "candidate_macro_f1": None,
        "candidate_miss_rate_pct": None,
        "log_loss": None,
        "brier_score": None,
        "sequence_edit_distance": None,
        "sequence_lcs_ratio": None,
    }

    gt_route = None
    if route_path.exists():
        try:
            gt_route = pd.read_csv(route_path, header=None, names=["arc_id"], dtype=np.int64)["arc_id"].tolist()
            
            # Read raw track and nodes for projection
            track_df_raw = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype=np.float64)
            nodes_df_raw = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64)
            arcs_df_raw = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)

            from data_processing.spatial_utils import get_transformer, project_coords
            mean_lon = float(track_df_raw["lon"].mean())
            mean_lat = float(track_df_raw["lat"].mean())
            transformer, _ = get_transformer(mean_lon, mean_lat)

            gx, gy = project_coords(track_df_raw["lon"].values, track_df_raw["lat"].values, transformer)
            nx, ny = project_coords(nodes_df_raw["lon"].values, nodes_df_raw["lat"].values, transformer)

            # Match ground truth arc per observation
            gt_arcs_per_obs = match_ground_truth_route(
                gx, gy, nx, ny,
                arcs_df_raw["u"].values,
                arcs_df_raw["v"].values,
                np.array(gt_route, dtype=np.int64)
            )

            # Observation Accuracy & Macro F1
            if len(decoded_arcs) == len(gt_arcs_per_obs):
                correct_count = int(np.sum(decoded_arcs == gt_arcs_per_obs))
                obs_acc = (correct_count / len(gt_arcs_per_obs)) * 100.0
                metrics_dict["observation_top1_accuracy_pct"] = round(obs_acc, 2)
                metrics_dict["hmm_route_accuracy_pct"] = round(obs_acc, 2)

            # Candidate Coverage / Miss Rate
            unique_cand_pairs = set(zip(cand_df["obs_idx"], cand_df["candidate_arc_id"]))
            missed_cands = sum(1 for t, true_a in enumerate(gt_arcs_per_obs) if (t, true_a) not in unique_cand_pairs)
            metrics_dict["candidate_miss_rate_pct"] = round((missed_cands / max(len(gt_arcs_per_obs), 1)) * 100.0, 2)

            # Sequence-Level Route Metrics
            collapsed_pred = collapse_consecutive_duplicates(decoded_arcs.tolist())
            edit_dist = compute_edit_distance(collapsed_pred, gt_route)
            lcs_len = compute_lcs_length(collapsed_pred, gt_route)
            denom = len(collapsed_pred) + len(gt_route)
            lcs_ratio = (2.0 * lcs_len / denom) if denom > 0 else 0.0

            metrics_dict["sequence_edit_distance"] = int(edit_dist)
            metrics_dict["sequence_lcs_ratio"] = round(float(lcs_ratio), 4)

            # Macro F1 on candidate labels
            gt_obs_cand_set = set(enumerate(gt_arcs_per_obs))
            cand_labels = np.array([
                1 if (obs, cand) in gt_obs_cand_set else 0
                for obs, cand in zip(cand_df["obs_idx"], cand_df["candidate_arc_id"])
            ], dtype=int)
            cand_preds = (pred_probs >= 0.5).astype(int)
            metrics_dict["candidate_macro_f1"] = round(float(f1_score(cand_labels, cand_preds, average="macro", zero_division=0)), 4)
            metrics_dict["log_loss"] = round(float(log_loss(cand_labels, pred_probs)), 4)
            metrics_dict["brier_score"] = round(float(brier_score_loss(cand_labels, pred_probs)), 4)

        except Exception as eval_err:
            print(f"[!] Server-side metric calculation warning: {eval_err}")

    # 6. Build Predicted Geometry (Coordinates for Leaflet Map)
    nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64)
    arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
    track_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype=np.float64)

    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values
    node_lats = nodes_df["lat"].values
    node_lons = nodes_df["lon"].values

    collapsed_arcs = collapse_consecutive_duplicates(decoded_arcs.tolist())
    pred_coords = []
    for arc_id in collapsed_arcs:
        if arc_id < len(arcs_df):
            u, v = u_nodes[arc_id], v_nodes[arc_id]
            pred_coords.append([round(float(node_lats[u]), 6), round(float(node_lons[u]), 6)])
    if collapsed_arcs and collapsed_arcs[-1] < len(arcs_df):
        last_v = v_nodes[collapsed_arcs[-1]]
        pred_coords.append([round(float(node_lats[last_v]), 6), round(float(node_lons[last_v]), 6)])

    gps_coords = [
        [round(float(lat), 6), round(float(lon), 6)]
        for lat, lon in zip(track_df["lat"], track_df["lon"])
    ]

    total_time_ms = (time.time() - t_start) * 1000.0

    # 7. VISIBLE TERMINAL REQUEST LOGGING (stdout for Two-Laptop Evaluator Demo)
    acc_display = f"{metrics_dict['observation_top1_accuracy_pct']}%" if metrics_dict['observation_top1_accuracy_pct'] is not None else "N/A"
    edit_display = str(metrics_dict['sequence_edit_distance']) if metrics_dict['sequence_edit_distance'] is not None else "N/A"
    lcs_display = f"{metrics_dict['sequence_lcs_ratio']:.4f}" if metrics_dict['sequence_lcs_ratio'] is not None else "N/A"

    print("\n" + "=" * 80)
    print(f"[API RECONSTRUCT] Trip: {trip_id} | Partition: {trip_dir.parent.name}")
    print(f"  -> GPS Observations: {len(track_df):,} | Candidates: {len(cand_df):,} | Latency: {total_time_ms:.1f} ms")
    print(f"  -> Route Continuity: {continuity_pct:.2f}% | Top-1 Accuracy: {acc_display}")
    print(f"  -> Sequence Edit Dist: {edit_display} | LCS Ratio: {lcs_display}")
    print("=" * 80 + "\n", flush=True)

    return {
        "trip_id": str(trip_id),
        "model_name": "Tuned LightGBM + HMM/Viterbi",
        "num_observations": len(track_df),
        "num_candidates": len(cand_df),
        "inference_time_ms": round(total_time_ms, 1),
        "metrics": metrics_dict,
        "geometry": {
            "gps_coordinates": gps_coords,
            "predicted_coordinates": pred_coords,
        },
        "predicted_arcs": decoded_arcs.tolist(),
    }


# -----------------------------------------------------------------------------
# 6. Direct Entrypoint Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("PATHSNAP_HOST", "0.0.0.0")
    port = int(os.environ.get("PATHSNAP_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
