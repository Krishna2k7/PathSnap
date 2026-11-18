import React from 'react';
import { CheckCircle2, AlertTriangle, Hash, Clock, Cpu } from 'lucide-react';

export default function ResultsPanel({ result, error }) {
  if (error) {
    return (
      <div className="bg-rose-950/30 border border-rose-800/50 rounded-xl p-3.5 flex items-start gap-2.5 text-xs text-rose-300">
        <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold block text-rose-200">Reconstruction Failed</span>
          <span className="text-[11px] leading-relaxed">{error}</span>
        </div>
      </div>
    );
  }

  if (!result) return null;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-2 text-xs">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="font-medium text-slate-300 flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
          Reconstruction Complete
        </span>
        <span className="font-mono text-[11px] text-slate-400">
          Trip {result.trip_id}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400">
        <div>
          <span>Model Architecture:</span>
          <div className="font-medium text-slate-200 truncate">{result.model_name}</div>
        </div>
        <div>
          <span>Evaluated Candidates:</span>
          <div className="font-mono font-medium text-slate-200">{result.num_candidates} arcs</div>
        </div>
      </div>
    </div>
  );
}
