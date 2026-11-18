"""
build_features.py - Batch feature extraction pipeline for PathSnap.
Processes all trips in datasets/train and datasets/student_test,
and saves standardized feature matrices to data/processed/.
"""

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import pandas as pd

from src.preprocessing.parser import load_trip_data
from src.preprocessing.feature_engine import extract_trip_candidate_features


def process_dataset(partition: str, output_path: Path, max_trips: int = None):
    dataset_dir = Path("datasets") / partition
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {dataset_dir}")

    trip_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if max_trips:
        trip_dirs = trip_dirs[:max_trips]

    print(f"[*] Processing partition '{partition}': {len(trip_dirs)} trips...")
    dfs = []
    start_time = time.time()

    for idx, tdir in enumerate(trip_dirs, start=1):
        t0 = time.time()
        try:
            trip_data = load_trip_data(tdir)
            feat_df = extract_trip_candidate_features(trip_data)
            dfs.append(feat_df)
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(trip_dirs)}] Trip {tdir.name}: {len(feat_df)} candidate rows ({elapsed:.2f}s)")
        except Exception as e:
            print(f"  [!] Error processing trip {tdir.name}: {e}")

    if not dfs:
        print("[!] No features extracted.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False, compression="gzip" if str(output_path).endswith(".gz") else None)

    total_time = time.time() - start_time
    print(f"[+] Finished '{partition}': {len(combined_df)} total records saved to {output_path} ({total_time:.2f}s)")
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features for PathSnap trips.")
    parser.add_argument("--partition", type=str, default="train", choices=["train", "student_test"])
    parser.add_argument("--max-trips", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    default_out = f"data/processed/{args.partition}_features.csv.gz"
    out_path = Path(args.out or default_out)
    process_dataset(args.partition, out_path, max_trips=args.max_trips)
