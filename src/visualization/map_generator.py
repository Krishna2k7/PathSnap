"""
map_generator.py - Multi-layer interactive map visualizer for PathSnap.
Generates interactive Leaflet/Folium maps visualizing:
1. Raw GPS observations (blue circle markers)
2. Road network geometry (gray lines)
3. Reference ground-truth route (green line)
4. Model-reconstructed route (red/orange line)
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd


def generate_interactive_map_html(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    ground_truth_arcs: Optional[List[int]] = None,
    predicted_arcs: Optional[List[int]] = None,
    output_html_path: Optional[Path] = None,
    title: str = "PathSnap Route Reconstruction",
) -> str:
    """
    Renders an interactive HTML map using Leaflet.js with full layer control.
    """
    center_lat = float(track_df["lat"].mean())
    center_lon = float(track_df["lon"].mean())

    # Raw GPS points: [[lat, lon], ...]
    gps_points = track_df[["lat", "lon"]].values.tolist()

    # Network arcs within trajectory bounding box (+ margin)
    lat_min, lat_max = track_df["lat"].min() - 0.01, track_df["lat"].max() + 0.01
    lon_min, lon_max = track_df["lon"].min() - 0.01, track_df["lon"].max() + 0.01

    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values
    node_lats = nodes_df["lat"].values
    node_lons = nodes_df["lon"].values

    # Extract road network segments in view
    network_lines = []
    # Subsample if too dense to ensure fast browser rendering
    for arc_id in range(0, len(arcs_df), 2 if len(arcs_df) > 50000 else 1):
        u = u_nodes[arc_id]
        v = v_nodes[arc_id]
        lat1, lon1 = node_lats[u], node_lons[u]
        lat2, lon2 = node_lats[v], node_lons[v]
        if lat_min <= lat1 <= lat_max and lon_min <= lon1 <= lon_max:
            network_lines.append([[lat1, lon1], [lat2, lon2]])

    # Build reference route line
    ref_lines = []
    if ground_truth_arcs is not None:
        for arc_id in ground_truth_arcs:
            if arc_id < len(arcs_df):
                u = u_nodes[arc_id]
                v = v_nodes[arc_id]
                ref_lines.append([[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]])

    # Build predicted route line
    pred_lines = []
    if predicted_arcs is not None:
        for arc_id in predicted_arcs:
            if arc_id < len(arcs_df):
                u = u_nodes[arc_id]
                v = v_nodes[arc_id]
                pred_lines.append([[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        .legend {{
            background: white;
            padding: 12px 16px;
            line-height: 22px;
            color: #333;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            border-radius: 6px;
            font-size: 13px;
        }}
        .legend i {{
            width: 18px;
            height: 4px;
            display: inline-block;
            margin-right: 8px;
            vertical-align: middle;
        }}
        .legend .circle {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 14);

        // OpenStreetMap basemap
        var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        // Layer 1: Road Network
        var networkGroup = L.layerGroup();
        var networkData = {json.dumps(network_lines[:5000])};
        networkData.forEach(function(line) {{
            L.polyline(line, {{ color: '#888888', weight: 1.5, opacity: 0.5 }}).addTo(networkGroup);
        }});
        networkGroup.addTo(map);

        // Layer 2: Raw GPS Track
        var gpsGroup = L.layerGroup();
        var gpsData = {json.dumps(gps_points)};
        gpsData.forEach(function(pt, idx) {{
            L.circleMarker(pt, {{
                radius: 3,
                color: '#1f77b4',
                fillColor: '#1f77b4',
                fillOpacity: 0.7,
                weight: 1
            }}).bindPopup("GPS Fix #" + idx + "<br>Lat: " + pt[0] + "<br>Lon: " + pt[1]).addTo(gpsGroup);
        }});
        gpsGroup.addTo(map);

        // Layer 3: Reference Ground-Truth Route
        var refGroup = L.layerGroup();
        var refData = {json.dumps(ref_lines)};
        refData.forEach(function(line) {{
            L.polyline(line, {{ color: '#2ca02c', weight: 5, opacity: 0.85 }}).addTo(refGroup);
        }});
        refGroup.addTo(map);

        // Layer 4: Reconstructed Predicted Route
        var predGroup = L.layerGroup();
        var predData = {json.dumps(pred_lines)};
        predData.forEach(function(line) {{
            L.polyline(line, {{ color: '#d62728', weight: 3.5, dashArray: '6, 6', opacity: 0.95 }}).addTo(predGroup);
        }});
        predGroup.addTo(map);

        // Layer Controls
        var overlays = {{
            "Road Network (Arcs)": networkGroup,
            "Raw GPS Observations": gpsGroup,
            "Reference Route (Ground Truth)": refGroup,
            "Reconstructed Route (Predicted)": predGroup
        }};
        L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

        // Custom Legend
        var legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function(map) {{
            var div = L.DomUtil.create('div', 'legend');
            div.innerHTML = '<strong>' + '{title}' + '</strong><br>' +
                '<span class="circle" style="background: #1f77b4;"></span> Raw GPS Fixes<br>' +
                '<i style="background: #888888;"></i> Road Network<br>' +
                '<i style="background: #2ca02c; height: 5px;"></i> Reference Route (Truth)<br>' +
                '<i style="background: #d62728; border-top: 2px dashed #d62728; height: 0;"></i> Predicted Route';
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>
"""
    if output_html_path is not None:
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[+] Saved interactive map to {output_html_path}")

    return html_content
