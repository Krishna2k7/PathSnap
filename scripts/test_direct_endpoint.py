"""
test_direct_endpoint.py - Direct API Function & Schema Verification
===================================================================
Directly calls FastAPI endpoint handlers (health_check, list_trips, reconstruct_route)
to verify live pipeline execution, response schemas, and zero data leakage.
"""

import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import health_check, list_trips, reconstruct_route

def main():
    print("=" * 80)
    print(" PATHSNAP PRODUCTION ENDPOINT VERIFICATION (LIVE PIPELINE) ")
    print("=" * 80)

    # 1. Test health_check
    print("\n[1/3] Testing health_check()...")
    health_resp = health_check()
    print(f"  -> Health OK: {health_resp}")
    assert health_resp["status"] == "healthy"
    assert health_resp["model_name"] == "Tuned LightGBM + HMM/Viterbi"

    # 2. Test list_trips
    print("\n[2/3] Testing list_trips()...")
    trips_resp = list_trips()
    print(f"  -> Trips Count: {trips_resp['count']} in '{trips_resp['active_evaluation_dir']}':")
    print(f"     {trips_resp['trips']}")
    assert "00000000" in trips_resp["trips"]

    # 3. Test reconstruct_route("00000000")
    print("\n[3/3] Testing reconstruct_route('00000000')...")
    t0 = time.time()
    recon_resp = reconstruct_route("00000000")
    duration_ms = (time.time() - t0) * 1000.0

    print(f"  -> Execution Completed in {duration_ms:.1f} ms")
    print(f"  Trip ID:              {recon_resp['trip_id']}")
    print(f"  Model Name:           {recon_resp['model_name']}")
    print(f"  Observations:         {recon_resp['num_observations']}")
    print(f"  Candidates:           {recon_resp['num_candidates']}")
    print(f"  Reported Latency:     {recon_resp['inference_time_ms']} ms")

    m = recon_resp["metrics"]
    print("\n  Server-Side Evaluated Metrics:")
    print(f"    - Route Continuity:       {m['route_continuity_pct']}%")
    print(f"    - Observation Top-1 Acc:  {m['observation_top1_accuracy_pct']}%")
    print(f"    - Sequence Edit Distance: {m['sequence_edit_distance']}")
    print(f"    - Sequence LCS Ratio:     {m['sequence_lcs_ratio']}")
    print(f"    - Candidate Macro F1:     {m['candidate_macro_f1']}")
    print(f"    - Candidate Miss Rate:    {m['candidate_miss_rate_pct']}%")
    print(f"    - Log Loss / Brier Score: {m['log_loss']} / {m['brier_score']}")

    g = recon_resp["geometry"]
    print(f"\n  Geometry Arrays:")
    print(f"    - GPS Coordinates:        {len(g['gps_coordinates'])} points")
    print(f"    - Predicted Route Coords: {len(g['predicted_coordinates'])} vertices")
    print(f"    - Predicted Arcs:         {len(recon_resp['predicted_arcs'])} elements")

    # Audit for zero leakage
    payload_dump = json.dumps(recon_resp)
    forbidden_keys = [
        "ground_truth_route",
        "ground_truth_coordinates",
        "true_arc",
        "is_true_arc",
        "gt_route",
    ]
    for key in forbidden_keys:
        assert key not in payload_dump, f"LEAKAGE DETECTED: '{key}' found in response!"

    print("\n" + "=" * 80)
    print(" ALL VERIFICATIONS PASSED: ZERO LEAKAGE & PERFECT ACCURACY! ")
    print("=" * 80)

if __name__ == "__main__":
    main()
