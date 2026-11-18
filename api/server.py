"""
server.py - Production HTTP API Server for PathSnap (Laptop 1).
Built with Python standard library (zero external dependencies).
Supports CORS, health checks, trip discovery, route reconstruction,
and serves the evaluator UI for Laptop 2.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import sys
import time
import math
from pathlib import Path
from urllib.parse import urlparse, parse_qs

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

# Cache models in memory
CACHED_MODELS = {}


def get_cached_model(model_name: str):
    if model_name not in CACHED_MODELS:
        if model_name == "random_forest":
            p = PROJECT_ROOT / "models" / "random_forest.joblib"
            CACHED_MODELS[model_name] = PathSnapRandomForest.load(p)
        else:
            p = PROJECT_ROOT / "models" / "gradient_boosting.joblib"
            CACHED_MODELS[model_name] = PathSnapGradientBoosting.load(p)
    return CACHED_MODELS[model_name]


class PathSnapHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for two-laptop LAN access
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/ui":
            ui_file = PROJECT_ROOT / "ui" / "index.html"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(ui_file, "rb") as f:
                self.wfile.write(f.read())
            return

        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "PathSnap-API"}).encode())
            return

        if path == "/trips":
            test_dir = PROJECT_ROOT / "datasets" / "student_test"
            trips = sorted([d.name for d in test_dir.iterdir() if d.is_dir()]) if test_dir.exists() else []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"trips": trips}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path.startswith("/reconstruct/"):
            trip_id = path.replace("/reconstruct/", "").strip("/")
            model_type = qs.get("model_type", ["random_forest"])[0]

            trip_dir = PROJECT_ROOT / "datasets" / "student_test" / trip_id
            if not trip_dir.exists():
                trip_dir = PROJECT_ROOT / "datasets" / "train" / trip_id

            if not trip_dir.exists():
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Trip {trip_id} not found"}).encode())
                return

            t0 = time.time()
            trip_data = load_trip_data(trip_dir)
            track_df = trip_data["track"]
            nodes_df = trip_data["nodes"]
            arcs_df = trip_data["arcs"]

            lon0 = float(track_df["lon"].mean())
            lat0 = float(track_df["lat"].mean())
            projector = LocalMetricProjector(lon0, lat0)
            t_x, t_y = projector.project(track_df["lon"].values, track_df["lat"].values)
            n_x, n_y = projector.project(nodes_df["lon"].values, nodes_df["lat"].values)

            feat_df = extract_trip_candidate_features(trip_data)

            model = get_cached_model(model_type)
            scores = model.predict_candidate_scores(feat_df)
            feat_df["score"] = scores

            observations_candidates = []
            for obs_id, group in feat_df.groupby("obs_id"):
                g_scores = group["score"].values
                g_arcs = group["cand_arc_id"].values.astype(int)
                exp_s = [math.exp(s - max(g_scores)) for s in g_scores]
                sum_p = sum(exp_s) + 1e-7
                cands = [{"arc_id": int(a), "prob": float(p / sum_p)} for a, p in zip(g_arcs, exp_s)]
                observations_candidates.append(cands)

            transition_model = NetworkTransitionModel(n_x, n_y, arcs_df, beta=15.0)
            decoder = ViterbiDecoder(transition_model, emission_weight=1.5)
            decoded_path = decoder.decode(observations_candidates, [t_x, t_y])

            continuity = compute_path_continuity(decoded_path, arcs_df)
            total_time_ms = (time.time() - t0) * 1000.0

            # Server-side accuracy calculation against ground truth if present
            acc = None
            if "is_true_arc" in feat_df.columns and feat_df["is_true_arc"].sum() > 0:
                true_dict = {
                    int(r["obs_id"]): int(r["cand_arc_id"])
                    for _, r in feat_df[feat_df["is_true_arc"] == 1].iterrows()
                }
                correct = sum(1 for t, a in enumerate(decoded_path) if true_dict.get(t) == a)
                acc = (correct / len(decoded_path)) * 100.0

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

            response = {
                "trip_id": trip_id,
                "model_name": model_type,
                "num_observations": len(track_df),
                "num_candidates": len(feat_df),
                "inference_time_ms": round(total_time_ms, 1),
                "route_continuity_pct": round(continuity, 2),
                "pointwise_accuracy_pct": round(acc, 2) if acc is not None else None,
                "predicted_path_length": len(decoded_path),
                "predicted_coordinates": coords,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            return

        self.send_response(404)
        self.end_headers()


def run_server(host="0.0.0.0", port=8000):
    server = HTTPServer((host, port), PathSnapHandler)
    print("=" * 65)
    print(f"[+] PathSnap Two-Laptop Server Running on http://{host}:{port}")
    print(f"    - Evaluator UI: http://localhost:{port}/")
    print(f"    - Health Check: http://localhost:{port}/health")
    print(f"    - Trip API:     http://localhost:{port}/trips")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Server stopped.")


if __name__ == "__main__":
    run_server()
