import React from 'react';
import { Play, Loader2, Navigation, Layers } from 'lucide-react';

export default function TripSelector({
  trips,
  selectedTrip,
  onSelectTrip,
  onRunReconstruction,
  loading,
  backendConnected,
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-4">
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <Navigation className="w-3.5 h-3.5 text-indigo-400" />
            Evaluation Trajectory
          </label>
          <span className="text-[10px] text-slate-400 font-medium px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
            {trips.length} Available
          </span>
        </div>
        <p className="text-[11px] text-slate-400 mb-2">
          Select an authorized test trajectory from the Laptop 1 registry:
        </p>

        <select
          value={selectedTrip}
          onChange={(e) => onSelectTrip(e.target.value)}
          disabled={loading || trips.length === 0}
          className="w-full bg-slate-800 border border-slate-700 hover:border-slate-600 focus:border-indigo-500 rounded-lg px-3 py-2.5 text-sm text-white font-mono focus:ring-2 focus:ring-indigo-500/20 outline-none transition disabled:opacity-50"
        >
          {trips.map((id) => (
            <option key={id} value={id}>
              Trip {id} (Development Test)
            </option>
          ))}
        </select>
      </div>

      <div className="pt-1">
        <button
          onClick={onRunReconstruction}
          disabled={loading || !backendConnected || !selectedTrip}
          className="w-full py-3 px-4 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:from-slate-800 disabled:to-slate-800 disabled:text-slate-500 text-white font-medium rounded-lg text-sm shadow-lg shadow-indigo-600/20 disabled:shadow-none transition-all flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Running Reconstruction...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Execute Reconstruction</span>
            </>
          )}
        </button>
      </div>

      <div className="text-[10px] text-slate-500 leading-relaxed border-t border-slate-800/80 pt-2.5 flex items-center gap-1.5">
        <Layers className="w-3 h-3 text-slate-500 shrink-0" />
        <span>Executes metric projection, candidate scoring, and Viterbi decoding server-side.</span>
      </div>
    </div>
  );
}
