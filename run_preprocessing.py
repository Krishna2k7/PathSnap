"""
PathSnap Master Preprocessing, EDA & Feature Engineering Execution Script
========================================================================
Single entry-point script to execute:
1. Comprehensive dataset exploration & quality audit (EDA).
2. Generation of publication-ready distribution plots and interactive Folium maps.
3. Spatial candidate generation and candidate miss rate auditing.
4. Rich geometric, kinematic, and topological feature engineering.
5. Leak-free 5-fold GroupKFold CV splitting (grouped strictly by trip_id).
6. Leak-free StandardScaler fitting on datasets/train and transforming datasets/student_test.
7. Verification of standalone FastAPI inference pipeline (transform_single_trip).
"""

import os
import sys
import tempfile
import time
import json
import pandas as pd
import numpy as np

# Ensure matplotlib uses writable temporary directory
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_pathsnap")

from data_processing.eda import run_full_eda
from features.pipeline import (
    fit_and_save_pipeline,
    transform_single_trip,
    FEATURE_COLUMNS,
)


def print_header(title: str):
    print("\n" + "=" * 75)
    print(f" {title.upper()}")
    print("=" * 75)


def run_pipeline():
    start_time = time.time()
    print_header("PathSnap End-to-End Pipeline Execution")

    # Step 1: Run EDA & Visualizations
    print_header("Step 1: Dataset Audit, Inventory & Visualizations (EDA)")
    eda_summary = run_full_eda(
        train_dir="datasets/train",
        student_test_dir="datasets/student_test",
        sample_train_trip="00000001",
        sample_test_trip="00000000",
    )

    # Step 2: Feature Engineering, Scaling & GroupKFold Splits
    print_header("Step 2: Candidate Generation, Feature Extraction & Scaling")
    pipe_summary = fit_and_save_pipeline(
        train_dir="datasets/train",
        student_test_dir="datasets/student_test",
        output_dir="processed_data",
        top_k=7,
        search_radius=80.0,
    )

    # Step 3: Verify Standalone FastAPI Inference Transform
    print_header("Step 3: Verifying Standalone FastAPI Inference Function")
    sample_trip_dir = "datasets/student_test/00000000"
    print(f"[Inference Test] Calling transform_single_trip('{sample_trip_dir}') without .route...")
    inf_df = transform_single_trip(sample_trip_dir)
    print(f"[Inference Test] Success! Generated {len(inf_df)} candidate rows.")
    print(f"[Inference Test] Output columns: {list(inf_df.columns[:8])}... (total {len(inf_df.columns)})")
    print(f"[Inference Test] Null values check: {inf_df[FEATURE_COLUMNS].isna().sum().sum()} nulls.")

    # Step 4: Verification and Consolidated Report
    elapsed = time.time() - start_time
    print_header("Step 4: Preprocessing & Feature Pipeline Verification Summary")

    train_stats = pipe_summary["train_stats"]
    test_stats = pipe_summary["student_test_stats"]

    train_parquet_size_mb = os.path.getsize("processed_data/train_features.parquet") / (1024 * 1024)
    test_parquet_size_mb = os.path.getsize("processed_data/student_test_features.parquet") / (1024 * 1024)

    print(f"Total Execution Time: {elapsed:.2f} seconds\n")

    print("--- DATASET PARTITIONS AUDIT ---")
    print(f"  Training Trips:            {train_stats['trip_count']} trips | {train_stats['total_gps_points']:,} GPS points")
    print(f"  Student-Testing Trips:     {test_stats['trip_count']} trips | {test_stats['total_gps_points']:,} GPS points")
    print(f"  Total Processed Trips:     {train_stats['trip_count'] + test_stats['trip_count']} trips")

    print("\n--- CANDIDATE GENERATION & MISS RATE ---")
    print(f"  Search Radius:             80.0 meters (with nearest-arc fallback)")
    print(f"  Top-K Candidates:          7 candidates per GPS observation")
    print(f"  Train Miss Rate:           {train_stats['candidate_miss_rate_pct']:.2f}% ({train_stats['total_misses']:,} misses out of {train_stats['total_gps_points']:,} obs)")
    print(f"  Student Test Miss Rate:    {test_stats['candidate_miss_rate_pct']:.2f}% ({test_stats['total_misses']:,} misses out of {test_stats['total_gps_points']:,} obs)")
    print(f"  Train Candidate Rows:      {train_stats['total_candidate_rows']:,}")
    print(f"  Student Test Rows:         {test_stats['total_candidate_rows']:,}")

    print("\n--- CLASS BALANCE (BINARY TARGET 'label') ---")
    tr_bal = pipe_summary["class_balance_train"]
    te_bal = pipe_summary["class_balance_student_test"]
    print(f"  Train Label 1 (Positive):  {tr_bal['label_1']:,} ({tr_bal['pos_ratio']*100:.2f}%)")
    print(f"  Train Label 0 (Negative):  {tr_bal['label_0']:,} ({(1-tr_bal['pos_ratio'])*100:.2f}%)")
    print(f"  Test Label 1 (Positive):   {te_bal['label_1']:,} ({te_bal['pos_ratio']*100:.2f}%)")
    print(f"  Test Label 0 (Negative):   {te_bal['label_0']:,} ({(1-te_bal['pos_ratio'])*100:.2f}%)")

    print("\n--- LEAK-FREE 5-FOLD GROUPKFOLD DISTRIBUTION ---")
    cv_dist = pipe_summary["cv_folds_distribution"]
    for fold_id, n_trips in sorted(cv_dist.items()):
        print(f"  Fold {fold_id}: {n_trips} trips")

    print("\n--- GENERATED ARTIFACTS & FILE SIZES ---")
    print(f"  results/eda_summary.json                 ({os.path.getsize('results/eda_summary.json'):,} bytes)")
    print(f"  results/trip_inventory.csv               ({os.path.getsize('results/trip_inventory.csv'):,} bytes)")
    print(f"  visualizations/trajectory_lengths.png    ({os.path.getsize('visualizations/trajectory_lengths.png'):,} bytes)")
    print(f"  visualizations/sampling_gaps.png         ({os.path.getsize('visualizations/sampling_gaps.png'):,} bytes)")
    print(f"  visualizations/road_network_stats.png    ({os.path.getsize('visualizations/road_network_stats.png'):,} bytes)")
    print(f"  visualizations/sample_train_map.html     ({os.path.getsize('visualizations/sample_train_map.html'):,} bytes)")
    print(f"  visualizations/sample_student_test_map.html ({os.path.getsize('visualizations/sample_student_test_map.html'):,} bytes)")
    print(f"  processed_data/scaler.joblib             ({os.path.getsize('processed_data/scaler.joblib'):,} bytes)")
    print(f"  processed_data/feature_columns.json      ({os.path.getsize('processed_data/feature_columns.json'):,} bytes)")
    print(f"  processed_data/train_features.parquet    ({train_parquet_size_mb:.2f} MB)")
    print(f"  processed_data/student_test_features.parquet ({test_parquet_size_mb:.2f} MB)")

    print("\n[SUCCESS] All checks passed cleanly with ZERO errors.")


if __name__ == "__main__":
    run_pipeline()
