import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Eye, EyeOff, MapPin, Navigation } from 'lucide-react';

export default function RouteMap({ gpsCoordinates, predictedCoordinates, selectedTrip, loading }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const gpsLayerGroupRef = useRef(null);
  const routeLayerGroupRef = useRef(null);

  const [showGps, setShowGps] = useState(true);
  const [showPredicted, setShowPredicted] = useState(true);

  // Initialize Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        zoomControl: false,
        attributionControl: false,
      }).setView([43.44, 77.19], 13);

      L.control.zoom({ position: 'bottomright' }).addTo(map);

      // CartoDB Dark Matter tiles for clean dark theme
      L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
      }).addTo(map);

      gpsLayerGroupRef.current = L.layerGroup().addTo(map);
      routeLayerGroupRef.current = L.layerGroup().addTo(map);

      mapInstanceRef.current = map;
    }

    return () => {
      // Keep map alive across re-renders
    };
  }, []);

  // Update Geometry
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const gpsLayer = gpsLayerGroupRef.current;
    const routeLayer = routeLayerGroupRef.current;

    gpsLayer.clearLayers();
    routeLayer.clearLayers();

    const boundsPoints = [];

    // Add GPS Observation Points (Blue)
    if (gpsCoordinates && gpsCoordinates.length > 0) {
      gpsCoordinates.forEach((pt, idx) => {
        const marker = L.circleMarker(pt, {
          radius: 3.5,
          color: '#38bdf8',
          fillColor: '#0284c7',
          fillOpacity: 0.8,
          weight: 1,
        }).bindTooltip(`Fix #${idx + 1}<br>Lat: ${pt[0]}<br>Lon: ${pt[1]}`, {
          className: 'bg-slate-900 text-white text-xs border border-slate-700 px-2 py-1 rounded shadow',
        });
        marker.addTo(gpsLayer);
        boundsPoints.push(pt);
      });
    }

    // Add Predicted Route (Red)
    if (predictedCoordinates && predictedCoordinates.length > 0) {
      const polyline = L.polyline(predictedCoordinates, {
        color: '#ef4444',
        weight: 4.5,
        opacity: 0.9,
        lineJoin: 'round',
      }).addTo(routeLayer);

      predictedCoordinates.forEach((pt) => boundsPoints.push(pt));

      // Start & End markers
      const startPt = predictedCoordinates[0];
      const endPt = predictedCoordinates[predictedCoordinates.length - 1];

      L.circleMarker(startPt, {
        radius: 6,
        color: '#22c55e',
        fillColor: '#16a34a',
        fillOpacity: 1,
        weight: 2,
      }).bindTooltip('Reconstruction Origin', { permanent: false }).addTo(routeLayer);

      L.circleMarker(endPt, {
        radius: 6,
        color: '#e11d48',
        fillColor: '#be123c',
        fillOpacity: 1,
        weight: 2,
      }).bindTooltip('Reconstruction Destination', { permanent: false }).addTo(routeLayer);
    }

    if (boundsPoints.length > 0) {
      map.fitBounds(L.latLngBounds(boundsPoints), {
        padding: [40, 40],
        maxZoom: 16,
      });
    }
  }, [gpsCoordinates, predictedCoordinates]);

  // Toggle Visibility
  useEffect(() => {
    if (!gpsLayerGroupRef.current || !routeLayerGroupRef.current || !mapInstanceRef.current) return;
    const map = mapInstanceRef.current;

    if (showGps) {
      if (!map.hasLayer(gpsLayerGroupRef.current)) map.addLayer(gpsLayerGroupRef.current);
    } else {
      if (map.hasLayer(gpsLayerGroupRef.current)) map.removeLayer(gpsLayerGroupRef.current);
    }

    if (showPredicted) {
      if (!map.hasLayer(routeLayerGroupRef.current)) map.addLayer(routeLayerGroupRef.current);
    } else {
      if (map.hasLayer(routeLayerGroupRef.current)) map.removeLayer(routeLayerGroupRef.current);
    }
  }, [showGps, showPredicted]);

  return (
    <div className="relative w-full h-full min-h-[500px] bg-slate-950 flex flex-col">
      {/* Map Element */}
      <div ref={mapContainerRef} className="w-full h-full flex-1" />

      {/* Floating Map Legend & Layer Controls */}
      <div className="absolute top-4 right-4 z-[400] bg-slate-900/90 backdrop-blur border border-slate-800 rounded-xl p-3 shadow-xl space-y-2 text-xs">
        <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 block border-b border-slate-800 pb-1">
          Map Layers & Legends
        </span>

        {/* GPS Layer Toggle */}
        <button
          onClick={() => setShowGps(!showGps)}
          className={`w-full flex items-center justify-between gap-3 px-2 py-1.5 rounded transition ${
            showGps ? 'bg-slate-800/80 text-white' : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-sky-400 border border-sky-300"></span>
            <span>Raw GPS Observations</span>
          </div>
          {showGps ? <Eye className="w-3.5 h-3.5 text-sky-400" /> : <EyeOff className="w-3.5 h-3.5" />}
        </button>

        {/* Predicted Route Toggle */}
        <button
          onClick={() => setShowPredicted(!showPredicted)}
          className={`w-full flex items-center justify-between gap-3 px-2 py-1.5 rounded transition ${
            showPredicted ? 'bg-slate-800/80 text-white' : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          <div className="flex items-center gap-2">
            <span className="w-3.5 h-1 rounded bg-rose-500"></span>
            <span>Predicted Route (HMM)</span>
          </div>
          {showPredicted ? <Eye className="w-3.5 h-3.5 text-rose-400" /> : <EyeOff className="w-3.5 h-3.5" />}
        </button>

        <div className="pt-1 border-t border-slate-800/80 text-[10px] text-slate-500">
          Trip: <span className="font-mono text-slate-300 font-medium">{selectedTrip || 'None'}</span>
        </div>
      </div>

      {/* Empty State Overlay */}
      {!loading && (!gpsCoordinates || gpsCoordinates.length === 0) && (
        <div className="absolute inset-0 z-[300] bg-slate-950/60 backdrop-blur-sm flex items-center justify-center pointer-events-none">
          <div className="text-center p-6 rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl max-w-sm pointer-events-auto">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-400 flex items-center justify-center mx-auto mb-3 border border-indigo-500/20">
              <Navigation className="w-6 h-6" />
            </div>
            <h4 className="text-sm font-bold text-white mb-1">No Trajectory Rendered</h4>
            <p className="text-xs text-slate-400 leading-relaxed">
              Select an evaluation trip on the left and click <b>Execute Reconstruction</b> to run map-matching on Laptop 1.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
