"""
projection.py - Geodesic calculations and local Cartesian metric projections.
Converts WGS84 angular coordinates (lon, lat) to metric coordinates (x, y) in meters,
and provides geometric primitives for point-to-segment distance and heading calculation.
"""

import math
import numpy as np
from typing import Tuple

EARTH_RADIUS = 6371000.0  # meters


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two angular points in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c


class LocalMetricProjector:
    """
    Projects WGS84 coordinates to a 2D Euclidean tangent plane in meters
    centered around the trip's centroid (lon0, lat0).
    Appropriate for localized vehicle trajectories (< 100 km).
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.cos_lat0 = math.cos(math.radians(self.lat0))
        self.m_per_deg_lat = (math.pi / 180.0) * EARTH_RADIUS
        self.m_per_deg_lon = (math.pi / 180.0) * EARTH_RADIUS * self.cos_lat0

    def project(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts lon, lat arrays to local metric x, y coordinates in meters."""
        x = (lon - self.lon0) * self.m_per_deg_lon
        y = (lat - self.lat0) * self.m_per_deg_lat
        return x, y

    def unproject(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts local metric x, y coordinates back to WGS84 lon, lat."""
        lon = self.lon0 + (x / self.m_per_deg_lon)
        lat = self.lat0 + (y / self.m_per_deg_lat)
        return lon, lat


def compute_heading(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Computes azimuth heading angle in radians from point (x1, y1) to (x2, y2).
    Range: [-pi, pi], 0 pointing along the positive x-axis (East), pi/2 North.
    """
    return math.atan2(y2 - y1, x2 - x1)


def angular_difference(theta1: float, theta2: float) -> float:
    """Computes the minimal unsigned absolute difference between two angles in [0, pi]."""
    diff = abs(theta1 - theta2) % (2 * math.pi)
    return 2 * math.pi - diff if diff > math.pi else diff


def point_to_segment_metrics(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """
    Calculates geometric metrics between a point P and directed line segment A -> B.
    
    Returns:
        (perp_dist, proj_ratio, seg_length, seg_heading)
        - perp_dist: Shortest Euclidean distance from P to the line segment in meters.
        - proj_ratio: Scalar projection along the segment, clamped to [0.0, 1.0].
        - seg_length: Metric length of the segment in meters.
        - seg_heading: Azimuth angle of directed segment from A to B.
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_length_sq = dx * dx + dy * dy
    seg_length = math.sqrt(seg_length_sq)
    seg_heading = math.atan2(dy, dx)

    if seg_length_sq == 0.0:
        dist = math.hypot(px - x1, py - y1)
        return dist, 0.0, 0.0, 0.0

    # Orthogonal projection factor: t = dot(AP, AB) / |AB|^2
    t = ((px - x1) * dx + (py - y1) * dy) / seg_length_sq
    t_clamped = max(0.0, min(1.0, t))

    # Nearest point on segment
    nearest_x = x1 + t_clamped * dx
    nearest_y = y1 + t_clamped * dy

    perp_dist = math.hypot(px - nearest_x, py - nearest_y)
    return perp_dist, t_clamped, seg_length, seg_heading
