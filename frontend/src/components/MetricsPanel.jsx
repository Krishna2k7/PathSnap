import React from 'react';
import { Gauge, CheckCircle, Crosshair, Zap, ShieldCheck, BarChart2 } from 'lucide-react';

export default function MetricsPanel({ metrics, numObservations, numCandidates, inferenceTime }) {
  const m = metrics || {};

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-3.5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
          <BarChart2 className="w-3.5 h-3.5 text-indigo-400" />
          Evaluation Metrics
        </h3>
        <span className="text-[10px] text-emerald-400 font-medium px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20">
          Server Validated
        </span>
      </div>

      {/* Row 1: Key Accuracies */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-slate-800/80 border border-slate-700/70 rounded-lg p-3">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
            <span>Route Continuity</span>
            <span className="text-[9px] uppercase px-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
              HMM
            </span>
          </div>
          <div className="text-xl font-bold text-emerald-400 font-mono">
            {m.route_continuity_pct != null ? `${m.route_continuity_pct}%` : '--'}
          </div>
          <p className="text-[10px] text-slate-500 mt-0.5">Topological arc connectivity</p>
        </div>

        <div className="bg-slate-800/80 border border-slate-700/70 rounded-lg p-3">
          <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
            <span>Top-1 Accuracy</span>
            <span className="text-[9px] uppercase px-1 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
              Classifier
            </span>
          </div>
          <div className="text-xl font-bold text-indigo-400 font-mono">
            {m.observation_top1_accuracy_pct != null ? `${m.observation_top1_accuracy_pct}%` : '--'}
          </div>
          <p className="text-[10px] text-slate-500 mt-0.5">Pointwise candidate ranking</p>
        </div>
      </div>

      {/* Row 2: Secondary Accuracies */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-slate-800/80 border border-slate-700/70 rounded-lg p-2.5">
          <span className="text-[11px] text-slate-400 block mb-0.5">HMM Path Accuracy</span>
          <div className="text-base font-bold text-slate-100 font-mono">
            {m.hmm_route_accuracy_pct != null ? `${m.hmm_route_accuracy_pct}%` : '--'}
          </div>
          <p className="text-[10px] text-slate-500">Viterbi decoded ground-truth</p>
        </div>

        <div className="bg-slate-800/80 border border-slate-700/70 rounded-lg p-2.5">
          <span className="text-[11px] text-slate-400 block mb-0.5">Candidate Macro F1</span>
          <div className="text-base font-bold text-violet-300 font-mono">
            {m.candidate_macro_f1 != null ? m.candidate_macro_f1 : '--'}
          </div>
          <p className="text-[10px] text-slate-500">Imbalance-adjusted score</p>
        </div>
      </div>

      {/* Row 3: Probabilistic Calibration & Candidate Coverage */}
      <div className="grid grid-cols-2 gap-2.5">
        <div className="bg-slate-800/80 border border-slate-700/70 rounded-lg p-2.5">
          <span className="text-[11px] text-slate-400 block mb-0.5">Candidate Miss Rate</span>
          <div className="text-base font-bold text-amber-400 font-mono">
            {m.candidate_miss_rate_pct != null ? `${m.candidate_miss_rate_pct}%` : '--'}
          </div>
          <p className="text-[10px] text-slate-500">KDTree spatial coverage</p>
        </div>

        <div className="bg-slate-800/80 border border-slate-700/70 rounded-lg p-2.5">
          <span className="text-[11px] text-slate-400 block mb-0.5">Log Loss / Brier</span>
          <div className="text-sm font-semibold text-slate-300 font-mono mt-0.5">
            {m.log_loss ?? '--'} / {m.brier_score ?? '--'}
          </div>
          <p className="text-[10px] text-slate-500">Emission calibration</p>
        </div>
      </div>

      {/* Row 4: Latency & Volume */}
      <div className="bg-slate-800/50 border border-slate-800 rounded-lg p-2.5 flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-slate-400">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span>Inference Latency:</span>
          <span className="font-semibold text-white font-mono">{inferenceTime ? `${inferenceTime} ms` : '--'}</span>
        </div>
        <div className="text-slate-400 text-[11px]">
          {numObservations ? `${numObservations} GPS fixes` : ''}
        </div>
      </div>
    </div>
  );
}
