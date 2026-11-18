"""
Shared preprocessing package for PathSnap.
Exposes data parsers, metric projection, candidate generators, and feature engines.
"""

from src.preprocessing.parser import load_trip_data
from src.preprocessing.projection import LocalMetricProjector, haversine_distance, point_to_segment_metrics
from src.preprocessing.candidate_gen import CandidateGenerator
from src.preprocessing.labeler import assign_observation_ground_truth
from src.preprocessing.feature_engine import extract_trip_candidate_features

__all__ = [
    "load_trip_data",
    "LocalMetricProjector",
    "haversine_distance",
    "point_to_segment_metrics",
    "CandidateGenerator",
    "assign_observation_ground_truth",
    "extract_trip_candidate_features",
]
