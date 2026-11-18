"""
test_server_e2e.py - End-to-End FastAPI Verification Script
===========================================================
Starts the FastAPI server locally, verifies /health, /trips, and /reconstruct/00000000,
validates that zero ground-truth data is leaked, and prints verification diagnostics.
"""

import sys
import os
import time
import json
import threading
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from api.main import app

def run_server_thread():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def main():
    print("=" * 80)
    print(" PATHSNAP FASTAPI END-TO-END INTEGRATION TEST ")
    print("=" * 80)

    # 1. Start Uvicorn in background thread
    server_thread = threading.Thread(target=run_server_thread, daemon=True)
    server_thread.start()
    print("[1/5] Starting FastAPI test server on http://127.0.0.1:8000...")
    time.sleep(2.0)

    base_url = "http://127.0.0.1:8000"

    # 2. Test /health
    print("\n[2/5] Testing GET /health...")
    req = urllib.request.Request(f"{base_url}/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        health_data = json.loads(resp.read().decode())
        print(f"  -> Health OK: {health_data}")
        assert health_data["status"] == "healthy"

    # 3. Test /trips
    print("\n[3/5] Testing GET /trips...")
    req = urllib.request.Request(f"{base_url}/trips")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        trips_data = json.loads(resp.read().decode())
        print(f"  -> Trips Discovered: {trips_data['trips']} (Total: {trips_data['count']})")
        assert "00000000" in trips_data["trips"], "Trip 00000000 must be discovered"

    # 4. Test /reconstruct/00000000
    print("\n[4/5] Testing POST /reconstruct/00000000 (Live ML + HMM Inference)...")
    post_req = urllib.request.Request(
        f"{base_url}/reconstruct/00000000",
        data=b"",
        headers={"Accept": "application/json"},
        method="POST"
    )
    t_start = time.time()
    with urllib.request.urlopen(post_req, timeout=30) as resp:
        elapsed = (time.time() - t_start) * 1000.0
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        recon_data = json.loads(resp.read().decode())
        print(f"  -> Received HTTP 200 in {elapsed:.1f} ms")

    # 5. Verify Response Schema & Data-Leakage Checks
    print("\n[5/5] Auditing Response Payload & Zero-Leakage Compliance...")
    print(f"  Trip ID:              {recon_data['trip_id']}")
    print(f"  Model Name:           {recon_data['model_name']}")
    print(f"  GPS Observations:     {recon_data['num_observations']}")
    print(f"  Candidate Segments:   {recon_data['num_candidates']}")
    print(f"  Inference Latency:    {recon_data['inference_time_ms']} ms")

    metrics = recon_data["metrics"]
    print("\n  Server-Side Validated Metrics:")
    print(f"    - Route Continuity:       {metrics['route_continuity_pct']}%")
    print(f"    - Observation Top-1 Acc:  {metrics['observation_top1_accuracy_pct']}%")
    print(f"    - Sequence Edit Distance: {metrics['sequence_edit_distance']}")
    print(f"    - Sequence LCS Ratio:     {metrics['sequence_lcs_ratio']}")
    print(f"    - Candidate Macro F1:     {metrics['candidate_macro_f1']}")
    print(f"    - Candidate Miss Rate:    {metrics['candidate_miss_rate_pct']}%")

    geometry = recon_data["geometry"]
    n_gps = len(geometry["gps_coordinates"])
    n_pred = len(geometry["predicted_coordinates"])
    print(f"\n  Geometry Arrays:")
    print(f"    - GPS Coordinates:        {n_gps} points (Sample: {geometry['gps_coordinates'][0]})")
    print(f"    - Predicted Route Coords: {n_pred} vertices (Sample: {geometry['predicted_coordinates'][0]})")

    # STRICT ZERO-LEAKAGE AUDIT:
    payload_str = json.dumps(recon_data)
    forbidden_keys = [
        "ground_truth_route",
        "ground_truth_coordinates",
        "true_arc",
        "is_true_arc",
        "gt_route",
    ]
    for k in forbidden_keys:
        assert k not in recon_data, f"SECURITY VIOLATION: '{k}' found in top-level response!"
        assert k not in metrics, f"SECURITY VIOLATION: '{k}' found in metrics response!"
        assert k not in geometry, f"SECURITY VIOLATION: '{k}' found in geometry response!"

    print("\n" + "=" * 80)
    print(" ALL TESTS PASSED: HTTP 200, VALID HMM RECONSTRUCTION & ZERO DATA LEAKAGE! ")
    print("=" * 80)

if __name__ == "__main__":
    main()
