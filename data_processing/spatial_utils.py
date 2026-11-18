"""
PathSnap Spatial Utilities Module
=================================
Provides robust, vectorized spatial calculations for converting geographical coordinates
(longitude, latitude in EPSG:4326) into local metric projected coordinates (appropriate
UTM zone via pyproj), computing Euclidean distances in meters, bearings, angular differences,
and point-to-segment perpendicular projections.

Key design principles:
- All physical distances, speeds, and offsets are computed in meters using UTM projections.
- Supports trips across different UTM zones worldwide via dynamic zone computation.
- Vectorized numpy operations for microsecond latency during inference.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
from pyproj import CRS, Transformer


def get_utm_epsg(lon: float, lat: float) -> int:
    """
    Computes the appropriate WGS 84 / UTM zone EPSG code for a given longitude and latitude.
    
    Formula:
        zone = floor((longitude + 180) / 6) + 1
        Northern hemisphere: EPSG = 32600 + zone
        Southern hemisphere: EPSG = 32700 + zone

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.

    Returns
    -------
    int
        EPSG code for the local UTM projection.
    """
    zone = int((lon + 180.0) // 6.0) + 1
    # Clamping zone between 1 and 60
    zone = max(1, min(60, zone))
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def get_transformer(lon: float, lat: float) -> Tuple[Transformer, int]:
    """
    Initializes a pyproj Transformer from EPSG:4326 (lon, lat) to the appropriate local UTM projection.

    Parameters
    ----------
    lon : float
        Representative longitude (e.g. mean or median longitude of trip).
    lat : float
        Representative latitude (e.g. mean or median latitude of trip).

    Returns
    -------
    Tuple[Transformer, int]
        Configured Transformer (with always_xy=True) and the selected EPSG code.
    """
    epsg = get_utm_epsg(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return transformer, epsg


def project_coords(
    lons: Union[np.ndarray, list, float],
    lats: Union[np.ndarray, list, float],
    transformer: Transformer
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts angular coordinates (lon, lat) to metric coordinates (x, y) in meters.

    Parameters
    ----------
    lons : array-like or float
        Longitude(s) in degrees.
    lats : array-like or float
        Latitude(s) in degrees.
    transformer : Transformer
        Configured pyproj Transformer with always_xy=True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Metric UTM x (easting) and y (northing) coordinates in meters.
    """
    x, y = transformer.transform(np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def compute_bearing(
    x1: Union[np.ndarray, float],
    y1: Union[np.ndarray, float],
    x2: Union[np.ndarray, float],
    y2: Union[np.ndarray, float]
) -> np.ndarray:
    """
    Computes azimuth / bearing in degrees [0, 360) clockwise from True North.

    Parameters
    ----------
    x1, y1 : metric coordinates of starting point(s).
    x2, y2 : metric coordinates of target point(s).

    Returns
    -------
    np.ndarray
        Bearing angle(s) in degrees in range [0, 360).
    """
    dx = np.asarray(x2, dtype=np.float64) - np.asarray(x1, dtype=np.float64)
    dy = np.asarray(y2, dtype=np.float64) - np.asarray(y1, dtype=np.float64)
    # arctan2(dx, dy) yields angle from North (y-axis) towards East (x-axis)
    bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
    return bearing


def compute_angle_difference(
    angle1: Union[np.ndarray, float],
    angle2: Union[np.ndarray, float]
) -> np.ndarray:
    """
    Computes the shortest angular difference in degrees [0, 180] between two bearings.

    Parameters
    ----------
    angle1, angle2 : array-like or float
        Bearings in degrees [0, 360).

    Returns
    -------
    np.ndarray
        Absolute angular difference in degrees in range [0, 180].
    """
    a1 = np.asarray(angle1, dtype=np.float64)
    a2 = np.asarray(angle2, dtype=np.float64)
    diff = np.abs(a1 - a2) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


def project_points_to_segments(
    px: np.ndarray,
    py: np.ndarray,
    p1x: np.ndarray,
    p1y: np.ndarray,
    p2x: np.ndarray,
    p2y: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Vectorized orthogonal projection of 2D points onto directed line segments.
    
    Given point P=(px, py) and segment S from P1=(p1x, p1y) to P2=(p2x, p2y):
        V = P2 - P1
        W = P - P1
        t = (W . V) / |V|^2
        t_clamped = clamp(t, 0.0, 1.0)
        Projection = P1 + t_clamped * V
        Perpendicular distance = |P - Projection|

    Parameters
    ----------
    px, py : Coordinates of query points (meters).
    p1x, p1y : Start coordinates of candidate segments (from_node, meters).
    p2x, p2y : End coordinates of candidate segments (to_node, meters).

    Returns
    -------
    Dict[str, np.ndarray] containing:
        - 'perp_dist': Perpendicular Euclidean distance in meters.
        - 'longitudinal_offset': Distance along segment from P1 to projection (meters).
        - 'proj_offset_ratio': Clamped projection parameter t in [0, 1].
        - 'seg_len': Segment length in meters.
        - 'arc_bearing': Bearing of directed segment (P1 -> P2) in degrees [0, 360).
        - 'proj_x': UTM Easting of projection point (meters).
        - 'proj_y': UTM Northing of projection point (meters).
        - 'dist_to_p1': Euclidean distance to start node (meters).
        - 'dist_to_p2': Euclidean distance to end node (meters).
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    p1x = np.asarray(p1x, dtype=np.float64)
    p1y = np.asarray(p1y, dtype=np.float64)
    p2x = np.asarray(p2x, dtype=np.float64)
    p2y = np.asarray(p2y, dtype=np.float64)

    dx = p2x - p1x
    dy = p2y - p1y
    seg_len_sq = dx * dx + dy * dy
    # Guard against zero-length segments
    safe_len_sq = np.where(seg_len_sq == 0.0, 1e-9, seg_len_sq)
    seg_len = np.sqrt(seg_len_sq)

    # Parametric projection factor
    t = ((px - p1x) * dx + (py - p1y) * dy) / safe_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    # Closest point coordinates on segment
    proj_x = p1x + t_clamped * dx
    proj_y = p1y + t_clamped * dy

    # Perpendicular distance in meters
    perp_dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    longitudinal_offset = t_clamped * seg_len

    # Distance to endpoints
    dist_to_p1 = np.sqrt((px - p1x) ** 2 + (py - p1y) ** 2)
    dist_to_p2 = np.sqrt((px - p2x) ** 2 + (py - p2y) ** 2)

    # Arc heading
    arc_bearing = compute_bearing(p1x, p1y, p2x, p2y)

    return {
        "perp_dist": perp_dist,
        "longitudinal_offset": longitudinal_offset,
        "proj_offset_ratio": t_clamped,
        "seg_len": seg_len,
        "arc_bearing": arc_bearing,
        "proj_x": proj_x,
        "proj_y": proj_y,
        "dist_to_p1": dist_to_p1,
        "dist_to_p2": dist_to_p2,
    }
