"""
test_fastapi_client.py - In-Memory ASGI TestClient Verification
================================================================
Uses FastAPI's official TestClient to execute end-to-end HTTP requests
against /health, /trips, and /reconstruct/00000000 in-memory.
"""

import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[Init] Importing FastAPI app and TestClient...", flush=True)
from fastapi.testclient import TestClient
from api.main import app

def test_api():
    print("=" * 80, flush=True)
    print(" PATHSNAP FASTAPI END-TO-END IN-MEMORY VERIFICATION ", flush=True)
    print("=" * 80, flush=True)

    client = TestClient(app)

    # 1. Test GET /health
    print("\n[1/4] Testing GET /health...", flush=True)
    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Expected 200, got {r_health.status_code}"
    health_data = r_health.json()
    print(f"  -> HTTP {r_health.status_code}: {health_data}", flush=True)
    assert health_data["status"] == "healthy"
    assert "active_evaluation_dir" in health_data

    # 2. Test GET /trips
    print("\n[2/4] Testing GET /trips...", flush=True)
    r_trips = client.get("/trips")
    assert r_trips.status_code == 200, f"Expected 200, got {r_trips.status_code}"
    trips_data = r_trips.json()
    print(f"  -> HTTP {r_trips.status_code}: Found {trips_data['count']} trips in '{trips_data['active_evaluation_dir']}':", flush=True)
    print(f"     {trips_data['trips']}", flush=True)
    assert "00000000" in trips_data["trips"]

    # 3. Test POST /reconstruct/00000000
    print("\n[3/4] Testing POST /reconstruct/00000000 (Live Pipeline Execution)...", flush=True)
    t0 = time.time()
    r_recon = client.post("/reconstruct/00000000")
    t_roundtrip = (time.time() - t0) * 1000.0

    assert r_recon.status_code == 200, f"Expected 200, got {r_recon.status_code}: {r_recon.text}"
    recon_data = r_recon.json()
    print(f"  -> HTTP {r_recon.status_code} OK (Client Roundtrip: {t_roundtrip:.1f} ms)", flush=True)

    # 4. Strict Validation of Payload & Zero-Leakage Compliance
    print("\n[4/4] Validating Payload Architecture & Academic Integrity...", flush=True)
    print(f"  - Trip ID:              {recon_data['trip_id']}")
    print(f"  - Model Architecture:   {recon_data['model_name']}")
    print(f"  - GPS Observations:     {recon_data['num_observations']}")
    print(f"  - Spatial Candidates:   {recon_data['num_candidates']}")
    print(f"  - Inference Latency:    {recon_data['inference_time_ms']} ms")

    m = recon_data["metrics"]
    print("\n  - Server-Side Computed Metrics:")
    print(f"    * Route Continuity:       {m['route_continuity_pct']}%")
    print(f"    * Observation Top-1 Acc:  {m['observation_top1_accuracy_pct']}%")
    print(f"    * HMM Route Accuracy:     {m['hmm_route_accuracy_pct']}%")
    print(f"    * Sequence Edit Distance: {m['sequence_edit_distance']}")
    print(f"    * Sequence LCS Ratio:     {m['sequence_lcs_ratio']}")
    print(f"    * Candidate Macro F1:     {m['candidate_macro_f1']}")
    print(f"    * Candidate Miss Rate:    {m['candidate_miss_rate_pct']}%")
    print(f"    * Log Loss / Brier Score: {m['log_loss']} / {m['brier_score']}")

    g = recon_data["geometry"]
    print(f"\n  - Reconstructed Geometry:")
    print(f"    * GPS Coordinates:        {len(g['gps_coordinates'])} points")
    print(f"    * Predicted Route Coords: {len(g['predicted_coordinates'])} vertices")
    print(f"    * Predicted Arcs Array:   {len(recon_data['predicted_arcs'])} items")

    # Strict Data-Leakage Assertions
    payload_dump = json.dumps(recon_data)
    forbidden_terms = [
        "ground_truth_route",
        "ground_truth_coordinates",
        "true_arc",
        "is_true_arc",
        "gt_route",
    ]
    for term in forbidden_terms:
        assert term not in payload_dump, f"LEAKAGE ALERT: '{term}' exposed in API response!"

    print("\n" + "=" * 80)
    print(" ALL ENDPOINTS & PIPELINES VERIFIED SUCCESSFULLY WITH ZERO LEAKAGE! ")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    test_api()
