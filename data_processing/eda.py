"""
PathSnap Exploratory Data Analysis (EDA) Module
===============================================
Comprehensive audit and exploratory analysis of the PathSnap dataset partitions:
- datasets/train (80 trips)
- datasets/student_test (10 trips)

Performs:
1. File inventory, row counts, missing/NaN/infinite value detection.
2. Temporal consistency: monotonicity, sampling intervals (nominal 1Hz vs irregular gaps).
3. Spatial properties: geographical bounding boxes, UTM projection zones, coordinate extents.
4. Road-network graph topology: node/arc counts, degree distributions, connected components.
5. Reference route properties: path length, segment counts, topological route continuity.
6. Exports structured outputs:
   - results/eda_summary.json
   - results/trip_inventory.csv
   - visualizations/*.png plots
   - visualizations/sample_train_map.html & sample_student_test_map.html (interactive Folium)
"""

import os
import json
import glob
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
import folium
from branca.element import Element, Figure

from data_processing.spatial_utils import (
    get_utm_epsg,
    get_transformer,
    project_coords,
    project_points_to_segments,
)


def load_raw_trip_data(trip_dir: str) -> Dict[str, Any]:
    """
    Loads raw .track, .nodes, .arcs, and optional .route files from a trip directory.
    """
    trip_id = os.path.basename(trip_dir)
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    data = {
        "trip_id": trip_id,
        "trip_dir": trip_dir,
        "track": None,
        "nodes": None,
        "arcs": None,
        "route": None,
        "files_present": {
            "track": os.path.exists(track_path),
            "nodes": os.path.exists(nodes_path),
            "arcs": os.path.exists(arcs_path),
            "route": os.path.exists(route_path),
        },
        "file_sizes_bytes": {},
    }

    if data["files_present"]["track"]:
        data["file_sizes_bytes"]["track"] = os.path.getsize(track_path)
        data["track"] = pd.read_csv(
            track_path, sep=r"\s+", header=None, names=["lon", "lat", "t"], dtype=np.float64
        )

    if data["files_present"]["nodes"]:
        data["file_sizes_bytes"]["nodes"] = os.path.getsize(nodes_path)
        data["nodes"] = pd.read_csv(
            nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64
        )

    if data["files_present"]["arcs"]:
        data["file_sizes_bytes"]["arcs"] = os.path.getsize(arcs_path)
        data["arcs"] = pd.read_csv(
            arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64
        )

    if data["files_present"]["route"]:
        data["file_sizes_bytes"]["route"] = os.path.getsize(route_path)
        data["route"] = pd.read_csv(
            route_path, header=None, names=["arc_id"], dtype=np.int64
        )["arc_id"].values

    return data


def analyze_single_trip(trip_data: Dict[str, Any], partition: str) -> Dict[str, Any]:
    """
    Computes summary audit metrics for one trip.
    """
    trip_id = trip_data["trip_id"]
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    record: Dict[str, Any] = {
        "trip_id": trip_id,
        "partition": partition,
        "files_all_present": all(trip_data["files_present"].values()),
        "track_file_size_kb": round(trip_data["file_sizes_bytes"].get("track", 0) / 1024, 2),
        "nodes_file_size_kb": round(trip_data["file_sizes_bytes"].get("nodes", 0) / 1024, 2),
        "arcs_file_size_kb": round(trip_data["file_sizes_bytes"].get("arcs", 0) / 1024, 2),
        "route_file_size_kb": round(trip_data["file_sizes_bytes"].get("route", 0) / 1024, 2),
        "n_gps_points": 0,
        "n_nodes": 0,
        "n_arcs": 0,
        "n_route_arcs": 0,
        "nan_count_gps": 0,
        "nan_count_nodes": 0,
        "nan_count_arcs": 0,
        "t_start": None,
        "t_end": None,
        "duration_sec": None,
        "dt_mean": None,
        "dt_median": None,
        "dt_max": None,
        "pct_1s_gaps": None,
        "pct_gt1s_gaps": None,
        "pct_gt5s_gaps": None,
        "is_time_monotonic": True,
        "min_lon": None,
        "max_lon": None,
        "min_lat": None,
        "max_lat": None,
        "span_lon_deg": None,
        "span_lat_deg": None,
        "utm_epsg": None,
        "trajectory_dist_km": None,
        "avg_speed_kmh": None,
        "graph_components": None,
        "isolated_nodes_count": None,
        "mean_in_degree": None,
        "mean_out_degree": None,
        "route_connected_pct": None,
        "route_total_len_km": None,
    }

    # 1. GPS Track Analysis
    if track is not None and len(track) > 0:
        record["n_gps_points"] = len(track)
        record["nan_count_gps"] = int(track.isna().sum().sum())

        t_vals = track["t"].values
        record["t_start"] = float(t_vals[0])
        record["t_end"] = float(t_vals[-1])
        record["duration_sec"] = float(t_vals[-1] - t_vals[0])

        diff_t = np.diff(t_vals)
        record["is_time_monotonic"] = bool(np.all(diff_t >= 0))
        if len(diff_t) > 0:
            record["dt_mean"] = float(np.mean(diff_t))
            record["dt_median"] = float(np.median(diff_t))
            record["dt_max"] = float(np.max(diff_t))
            record["pct_1s_gaps"] = float(np.mean(np.isclose(diff_t, 1.0, atol=0.05)) * 100.0)
            record["pct_gt1s_gaps"] = float(np.mean(diff_t > 1.05) * 100.0)
            record["pct_gt5s_gaps"] = float(np.mean(diff_t > 5.0) * 100.0)

        record["min_lon"] = float(track["lon"].min())
        record["max_lon"] = float(track["lon"].max())
        record["min_lat"] = float(track["lat"].min())
        record["max_lat"] = float(track["lat"].max())
        record["span_lon_deg"] = float(record["max_lon"] - record["min_lon"])
        record["span_lat_deg"] = float(record["max_lat"] - record["min_lat"])

        mean_lon = float(track["lon"].mean())
        mean_lat = float(track["lat"].mean())
        epsg = get_utm_epsg(mean_lon, mean_lat)
        record["utm_epsg"] = epsg

        transformer, _ = get_transformer(mean_lon, mean_lat)
        gx, gy = project_coords(track["lon"].values, track["lat"].values, transformer)
        gps_dx = np.diff(gx)
        gps_dy = np.diff(gy)
        point_step_dists = np.sqrt(gps_dx ** 2 + gps_dy ** 2)
        total_dist_m = float(np.sum(point_step_dists))
        record["trajectory_dist_km"] = round(total_dist_m / 1000.0, 3)
        if record["duration_sec"] and record["duration_sec"] > 0:
            record["avg_speed_kmh"] = round((total_dist_m / record["duration_sec"]) * 3.6, 2)

    # 2. Road Network Analysis
    if nodes is not None and arcs is not None:
        record["n_nodes"] = len(nodes)
        record["n_arcs"] = len(arcs)
        record["nan_count_nodes"] = int(nodes.isna().sum().sum())
        record["nan_count_arcs"] = int(arcs.isna().sum().sum())

        u_nodes = arcs["u"].values
        v_nodes = arcs["v"].values
        n_v = len(nodes)

        # In-degree and out-degree
        out_counts = np.bincount(u_nodes, minlength=n_v)
        in_counts = np.bincount(v_nodes, minlength=n_v)
        total_degrees = in_counts + out_counts
        record["isolated_nodes_count"] = int(np.sum(total_degrees == 0))
        record["mean_in_degree"] = round(float(np.mean(in_counts)), 3)
        record["mean_out_degree"] = round(float(np.mean(out_counts)), 3)

        # Graph connected components (directed & undirected)
        adj_data = np.ones(len(arcs), dtype=bool)
        adj_mat = csr_matrix((adj_data, (u_nodes, v_nodes)), shape=(n_v, n_v))
        n_comp, _ = csgraph.connected_components(adj_mat, directed=False)
        record["graph_components"] = int(n_comp)

    # 3. Reference Route Analysis
    if route is not None and len(route) > 0 and arcs is not None and nodes is not None:
        record["n_route_arcs"] = len(route)
        route_u = arcs.iloc[route]["u"].values
        route_v = arcs.iloc[route]["v"].values

        # Check consecutive arc connectivity
        if len(route) > 1:
            consecutive_matches = np.sum(route_v[:-1] == route_u[1:])
            record["route_connected_pct"] = round(float(consecutive_matches / (len(route) - 1) * 100.0), 2)
        else:
            record["route_connected_pct"] = 100.0

        # Calculate physical length of ground-truth route
        if record["utm_epsg"] is not None:
            transformer, _ = get_transformer(mean_lon, mean_lat)
            nx, ny = project_coords(nodes["lon"].values, nodes["lat"].values, transformer)
            p1x, p1y = nx[route_u], ny[route_u]
            p2x, p2y = nx[route_v], ny[route_v]
            arc_lengths = np.sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)
            record["route_total_len_km"] = round(float(np.sum(arc_lengths)) / 1000.0, 3)

    return record


def run_dataset_audit(
    train_dir: str = "datasets/train",
    student_test_dir: str = "datasets/student_test"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes full audit over both train and student_test partitions.
    """
    train_trip_dirs = sorted(glob.glob(os.path.join(train_dir, "*")))
    student_test_trip_dirs = sorted(glob.glob(os.path.join(student_test_dir, "*")))

    records = []
    print(f"[EDA] Auditing train partition ({len(train_trip_dirs)} trips)...")
    for td in train_trip_dirs:
        trip_data = load_raw_trip_data(td)
        rec = analyze_single_trip(trip_data, "train")
        records.append(rec)

    print(f"[EDA] Auditing student_test partition ({len(student_test_trip_dirs)} trips)...")
    for td in student_test_trip_dirs:
        trip_data = load_raw_trip_data(td)
        rec = analyze_single_trip(trip_data, "student_test")
        records.append(rec)

    inventory_df = pd.DataFrame(records)

    # Compute partition-level aggregate statistics
    def partition_summary(part_df: pd.DataFrame, name: str) -> Dict[str, Any]:
        return {
            "partition": name,
            "trip_count": len(part_df),
            "all_files_present_pct": float(part_df["files_all_present"].mean() * 100.0),
            "total_gps_points": int(part_df["n_gps_points"].sum()),
            "gps_points_per_trip": {
                "min": int(part_df["n_gps_points"].min()),
                "max": int(part_df["n_gps_points"].max()),
                "mean": round(float(part_df["n_gps_points"].mean()), 1),
                "median": int(part_df["n_gps_points"].median()),
                "std": round(float(part_df["n_gps_points"].std()), 1),
            },
            "nan_or_invalid_values": {
                "gps_nans": int(part_df["nan_count_gps"].sum()),
                "nodes_nans": int(part_df["nan_count_nodes"].sum()),
                "arcs_nans": int(part_df["nan_count_arcs"].sum()),
            },
            "temporal_consistency": {
                "strictly_monotonic_trips": int(part_df["is_time_monotonic"].sum()),
                "mean_pct_exactly_1s": round(float(part_df["pct_1s_gaps"].dropna().mean()), 2),
                "mean_pct_irregular_gt1s": round(float(part_df["pct_gt1s_gaps"].dropna().mean()), 2),
                "trips_with_gaps_gt5s": int((part_df["pct_gt5s_gaps"] > 0).sum()),
                "max_time_gap_observed_sec": float(part_df["dt_max"].max()),
            },
            "road_network": {
                "nodes_per_trip_mean": int(part_df["n_nodes"].mean()),
                "arcs_per_trip_mean": int(part_df["n_arcs"].mean()),
                "mean_graph_components": round(float(part_df["graph_components"].mean()), 1),
                "mean_isolated_nodes": round(float(part_df["isolated_nodes_count"].mean()), 1),
            },
            "reference_route": {
                "mean_route_arcs": round(float(part_df["n_route_arcs"].mean()), 1),
                "mean_route_len_km": round(float(part_df["route_total_len_km"].dropna().mean()), 2),
                "mean_route_connected_pct": round(float(part_df["route_connected_pct"].dropna().mean()), 2),
            },
            "spatial_extent": {
                "min_lon_global": float(part_df["min_lon"].min()),
                "max_lon_global": float(part_df["max_lon"].max()),
                "min_lat_global": float(part_df["min_lat"].min()),
                "max_lat_global": float(part_df["max_lat"].max()),
                "utm_zones_used": [int(x) for x in sorted(part_df["utm_epsg"].unique().tolist())],
            },
        }

    train_subset = inventory_df[inventory_df["partition"] == "train"]
    test_subset = inventory_df[inventory_df["partition"] == "student_test"]

    eda_summary = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "train": partition_summary(train_subset, "train"),
        "student_test": partition_summary(test_subset, "student_test"),
        "total_dataset_trips": len(inventory_df),
        "total_dataset_gps_observations": int(inventory_df["n_gps_points"].sum()),
    }

    # Save outputs
    os.makedirs("results", exist_ok=True)
    inventory_df.to_csv("results/trip_inventory.csv", index=False)
    with open("results/eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)

    print("[EDA] Successfully saved results/trip_inventory.csv and results/eda_summary.json")
    return inventory_df, eda_summary


def generate_eda_plots(inventory_df: pd.DataFrame, output_dir: str = "visualizations"):
    """
    Generates publication-quality EDA plots analyzing dataset distributions.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    train_df = inventory_df[inventory_df["partition"] == "train"]
    test_df = inventory_df[inventory_df["partition"] == "student_test"]

    # 1. Trajectory Lengths & Route Sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(train_df["n_gps_points"], bins=20, alpha=0.7, color="#1f77b4", label="Train (80 trips)", edgecolor="black")
    axes[0].hist(test_df["n_gps_points"], bins=10, alpha=0.7, color="#ff7f0e", label="Student Test (10 trips)", edgecolor="black")
    axes[0].set_title("GPS Observation Count per Trip Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of GPS Observations", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend()

    axes[1].hist(train_df["n_route_arcs"], bins=20, alpha=0.7, color="#2ca02c", label="Train Arcs", edgecolor="black")
    axes[1].hist(test_df["n_route_arcs"], bins=10, alpha=0.7, color="#d62728", label="Student Test Arcs", edgecolor="black")
    axes[1].set_title("Ground Truth Route Arc Count Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Route Road Segments", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_lengths.png"), dpi=200)
    plt.close(fig)

    # 2. Sampling Gap Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot(
        [train_df["pct_1s_gaps"].dropna(), test_df["pct_1s_gaps"].dropna()],
        patch_artist=True,
        boxprops=dict(facecolor="#aec7e8", color="#1f77b4"),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[0].set_xticks([1, 2], ["Train", "Student Test"])
    axes[0].set_title("Percentage of 1.0s Sampling Steps per Trip", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    axes[0].set_ylim(80, 101)

    axes[1].hist(train_df["dt_max"], bins=20, color="#d62728", alpha=0.75, edgecolor="black")
    axes[1].set_title("Maximum GPS Gap per Trip (Outlier / Dropout Analysis)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Max Time Gap (seconds)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "sampling_gaps.png"), dpi=200)
    plt.close(fig)

    # 3. Road-Network Size and Topological Structure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(train_df["n_nodes"], train_df["n_arcs"], alpha=0.8, color="#1f77b4", edgecolors="none", s=50, label="Train")
    axes[0].scatter(test_df["n_nodes"], test_df["n_arcs"], alpha=0.9, color="#d62728", marker="^", s=70, label="Student Test")
    axes[0].set_title("Road Network Scale: Nodes vs Arcs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Node Count", fontsize=11)
    axes[0].set_ylabel("Arc Count", fontsize=11)
    axes[0].legend()

    axes[1].hist(train_df["graph_components"], bins=15, alpha=0.7, color="#9467bd", edgecolor="black", label="Components")
    axes[1].set_title("Disconnected Graph Components in Subgraphs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Connected Components Count", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "road_network_stats.png"), dpi=200)
    plt.close(fig)

    print(f"[EDA] Generated EDA distribution plots in {output_dir}/")


def create_interactive_folium_map(
    trip_dir: str,
    output_path: str,
    max_network_arcs: int = 4000
) -> folium.Map:
    """
    Builds a rich, interactive Folium map visualization of a single trip with:
    - Layer 1: Raw GPS Observations (blue circle markers with index & timestamp tooltips)
    - Layer 2: Road Network Arcs (subdued slate gray lines of local street network)
    - Layer 3: Ground-truth .route (bold vibrant green path traversed by the vehicle)
    - Layer 4: Trip Start and Finish markers (green marker for start, red marker for end)
    - Tile layers: OpenStreetMap & CartoDB Positron
    - Interactive LayerControl and Color Legend
    """
    trip_data = load_raw_trip_data(trip_dir)
    trip_id = trip_data["trip_id"]
    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]

    center_lat = float(track["lat"].mean())
    center_lon = float(track["lon"].mean())

    f_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(f_map)

    # Trajectory bounding box with small buffer to filter nearby road network
    lat_pad = (track["lat"].max() - track["lat"].min()) * 0.15 + 0.005
    lon_pad = (track["lon"].max() - track["lon"].min()) * 0.15 + 0.005
    min_lat, max_lat = track["lat"].min() - lat_pad, track["lat"].max() + lat_pad
    min_lon, max_lon = track["lon"].min() - lon_pad, track["lon"].max() + lon_pad

    # 1. Road Network Arcs Layer
    net_fg = folium.FeatureGroup(name="Road Network (Local Arcs)", show=True)
    if nodes is not None and arcs is not None:
        node_lons = nodes["lon"].values
        node_lats = nodes["lat"].values
        u = arcs["u"].values
        v = arcs["v"].values

        # Filter arcs within bounding box
        in_bbox = (
            (node_lons[u] >= min_lon) & (node_lons[u] <= max_lon) &
            (node_lats[u] >= min_lat) & (node_lats[u] <= max_lat)
        )
        filtered_indices = np.where(in_bbox)[0]
        if len(filtered_indices) > max_network_arcs:
            filtered_indices = np.random.choice(filtered_indices, max_network_arcs, replace=False)

        lines_coords = []
        for a_idx in filtered_indices:
            u_i, v_i = u[a_idx], v[a_idx]
            lines_coords.append([
                [node_lats[u_i], node_lons[u_i]],
                [node_lats[v_i], node_lons[v_i]],
            ])

        folium.PolyLine(
            locations=lines_coords,
            color="#7f7f7f",
            weight=1.5,
            opacity=0.45,
            tooltip=f"Road Network Arcs ({len(lines_coords)} in view)",
        ).add_to(net_fg)
    net_fg.add_to(f_map)

    # 2. Ground-truth .route Layer
    if route is not None and len(route) > 0 and nodes is not None and arcs is not None:
        route_fg = folium.FeatureGroup(name="Ground-Truth Route (.route)", show=True)
        r_u = arcs.iloc[route]["u"].values
        r_v = arcs.iloc[route]["v"].values
        route_lines = []
        for u_i, v_i in zip(r_u, r_v):
            route_lines.append([
                [node_lats[u_i], node_lons[u_i]],
                [node_lats[v_i], node_lons[v_i]],
            ])
        folium.PolyLine(
            locations=route_lines,
            color="#00aa00",
            weight=5.0,
            opacity=0.85,
            tooltip=f"Ground Truth Route ({len(route)} arcs)",
        ).add_to(route_fg)
        route_fg.add_to(f_map)

    # 3. Raw GPS Trajectory Layer
    gps_fg = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    # Connect GPS points with thin dashed blue line
    gps_coords = list(zip(track["lat"].values, track["lon"].values))
    folium.PolyLine(
        locations=gps_coords,
        color="#0066cc",
        weight=2.0,
        dash_array="5, 5",
        opacity=0.6,
    ).add_to(gps_fg)

    # Plot sample GPS observation dots (subsample if trip is massive to keep browser fast)
    step = 1 if len(track) < 1500 else int(np.ceil(len(track) / 1200))
    for i in range(0, len(track), step):
        lat_i = float(track["lat"].iloc[i])
        lon_i = float(track["lon"].iloc[i])
        t_i = float(track["t"].iloc[i])
        folium.CircleMarker(
            location=[lat_i, lon_i],
            radius=3.5,
            color="#0044aa",
            fill=True,
            fill_color="#3399ff",
            fill_opacity=0.9,
            weight=1,
            tooltip=f"GPS #{i} | t={t_i:.1f}s | ({lat_i:.5f}, {lon_i:.5f})",
        ).add_to(gps_fg)
    gps_fg.add_to(f_map)

    # 4. Start & End Markers
    markers_fg = folium.FeatureGroup(name="Start & Destination Markers", show=True)
    start_lat, start_lon = float(track["lat"].iloc[0]), float(track["lon"].iloc[0])
    end_lat, end_lon = float(track["lat"].iloc[-1]), float(track["lon"].iloc[-1])
    folium.Marker(
        location=[start_lat, start_lon],
        popup=f"Trip {trip_id} Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(markers_fg)
    folium.Marker(
        location=[end_lat, end_lon],
        popup=f"Trip {trip_id} End",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
    ).add_to(markers_fg)
    markers_fg.add_to(f_map)

    # 5. LayerControl
    folium.LayerControl(collapsed=False).add_to(f_map)

    # 6. Embedded Custom HTML Legend
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 260px; height: 160px; 
        background-color: white; border:2px solid #888; border-radius: 6px;
        z-index: 9999; font-size: 13px; font-family: sans-serif;
        padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <b>PathSnap Map: Trip {trip_id}</b><br>
        <i class="fa fa-circle" style="color:#0044aa; margin-top: 8px;"></i>&nbsp; Raw GPS Points (.track)<br>
        <span style="color:#0066cc; font-weight:bold;">---</span>&nbsp; GPS Trajectory Polyline<br>
        <span style="color:#00aa00; font-weight:bold;">━━━</span>&nbsp; Ground-Truth Route (.route)<br>
        <span style="color:#7f7f7f; font-weight:bold;">━━━</span>&nbsp; Road Network Arcs (.arcs)<br>
        <i class="fa fa-play" style="color:green;"></i>&nbsp; Start Point &nbsp;&nbsp;
        <i class="fa fa-flag" style="color:red;"></i>&nbsp; End Point
    </div>
    """
    f_map.get_root().html.add_child(Element(legend_html))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f_map.save(output_path)
    print(f"[EDA] Interactive Folium map saved to {output_path}")
    return f_map


def run_full_eda(
    train_dir: str = "datasets/train",
    student_test_dir: str = "datasets/student_test",
    sample_train_trip: str = "00000001",
    sample_test_trip: str = "00000000"
) -> Dict[str, Any]:
    """
    Main entry point for EDA pipeline.
    """
    print("=" * 60)
    print("      PATHSNAP EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 60)
    inventory_df, eda_summary = run_dataset_audit(train_dir, student_test_dir)
    generate_eda_plots(inventory_df, "visualizations")

    # Generate Folium maps
    train_sample_path = os.path.join(train_dir, sample_train_trip)
    test_sample_path = os.path.join(student_test_dir, sample_test_trip)

    create_interactive_folium_map(train_sample_path, "visualizations/sample_train_map.html")
    create_interactive_folium_map(test_sample_path, "visualizations/sample_student_test_map.html")

    print("[EDA] Audit completed successfully.")
    return eda_summary


if __name__ == "__main__":
    run_full_eda()
