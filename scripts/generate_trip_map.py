"""
generate_trip_map.py - Generates an interactive multi-layer map for any trip.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.parser import load_trip_data
from src.models.gradient_boosting import PathSnapGradientBoosting
from src.visualization.map_generator import generate_interactive_map_html
from scripts.reconstruct_route import reconstruct_trip


def build_map_for_trip(trip_id: str, partition: str = "student_test"):
    trip_dir = PROJECT_ROOT / "datasets" / partition / trip_id
    trip_data = load_trip_data(trip_dir)

    print(f"[*] Reconstructing route for Trip {trip_id}...")
    result = reconstruct_trip(trip_id, partition)

    html_out = PROJECT_ROOT / "reports" / f"map_trip_{trip_id}.html"
    generate_interactive_map_html(
        track_df=trip_data["track"],
        nodes_df=trip_data["nodes"],
        arcs_df=trip_data["arcs"],
        ground_truth_arcs=trip_data.get("route"),
        predicted_arcs=result["full_hmm_path"],
        output_html_path=html_out,
        title=f"PathSnap Trip {trip_id} Reconstructed Route",
    )
    print(f"[+] Map successfully created: {html_out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-id", type=str, default="00000007")
    parser.add_argument("--partition", type=str, default="student_test")
    args = parser.parse_args()

    build_map_for_trip(args.trip_id, args.partition)
