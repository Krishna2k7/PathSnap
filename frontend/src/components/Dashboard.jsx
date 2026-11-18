import React from 'react';
import TripSelector from './TripSelector';
import MetricsPanel from './MetricsPanel';
import ResultsPanel from './ResultsPanel';
import StatusIndicator from './StatusIndicator';
import RouteMap from './RouteMap';

export default function Dashboard({
  trips,
  selectedTrip,
  onSelectTrip,
  onRunReconstruction,
  loading,
  backendConnected,
  backendUrl,
  result,
  error,
}) {
  return (
    <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
      {/* Sidebar Controls & Metrics */}
      <aside className="w-full lg:w-96 bg-slate-950 border-r border-slate-800/80 p-4 space-y-4 overflow-y-auto custom-scrollbar shrink-0 flex flex-col justify-between">
        <div className="space-y-4">
          {/* Target Host Indicator */}
          <StatusIndicator backendConnected={backendConnected} backendUrl={backendUrl} />

          {/* Trip Selector & Execution Button */}
          <TripSelector
            trips={trips}
            selectedTrip={selectedTrip}
            onSelectTrip={onSelectTrip}
            onRunReconstruction={onRunReconstruction}
            loading={loading}
            backendConnected={backendConnected}
          />

          {/* Error Banner or Completion Notice */}
          <ResultsPanel result={result} error={error} />

          {/* Metrics Dashboard */}
          <MetricsPanel
            metrics={result?.metrics}
            numObservations={result?.num_observations}
            numCandidates={result?.num_candidates}
            inferenceTime={result?.inference_time_ms}
          />
        </div>

        {/* Security & Academic Integrity Badge */}
        <div className="text-[10px] text-slate-500 border-t border-slate-800/80 pt-3 leading-relaxed">
          <span className="font-semibold text-slate-400 block mb-0.5">🔒 Data Integrity Guaranteed:</span>
          Ground-truth reference trajectories and labels remain securely server-side on Laptop 1.
          Evaluations are computed dynamically using authorized test splits.
        </div>
      </aside>

      {/* Main Map View */}
      <main className="flex-1 relative flex flex-col min-h-[500px]">
        <RouteMap
          gpsCoordinates={result?.geometry?.gps_coordinates}
          predictedCoordinates={result?.geometry?.predicted_coordinates}
          selectedTrip={selectedTrip}
          loading={loading}
        />
      </main>
    </div>
  );
}
