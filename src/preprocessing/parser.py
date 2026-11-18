"""
parser.py - Fast parsing and loading of raw PathSnap trip files.
Handles .track, .nodes, .arcs, and .route files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def load_trip_data(trip_dir: Path) -> Dict[str, object]:
    """
    Loads all data components for a specific trip directory.
    
    Args:
        trip_dir: Path to directory containing <trip_id>.track, .nodes, .arcs, [.route]
        
    Returns:
        dict containing:
            'trip_id': str
            'track': pd.DataFrame [lon, lat, time]
            'nodes': pd.DataFrame [lon, lat]
            'arcs': pd.DataFrame [u, v]
            'route': Optional[List[int]] (None if not provided)
    """
    trip_id = trip_dir.name
    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"
    
    if not track_file.exists():
        raise FileNotFoundError(f"Missing track file: {track_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")
    if not arcs_file.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_file}")
        
    # .track: longitude, latitude, timestamp
    track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "time"], dtype=np.float64)
    
    # .nodes: longitude, latitude (row index = node_id)
    nodes_df = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64)
    
    # .arcs: from_node, to_node (row index = arc_id)
    arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int32)
    
    # .route: ground truth sequence of arc_ids
    route = None
    if route_file.exists():
        with open(route_file, "r") as f:
            route = [int(line.strip()) for line in f if line.strip()]
            
    return {
        "trip_id": trip_id,
        "track": track_df,
        "nodes": nodes_df,
        "arcs": arcs_df,
        "route": route,
    }
