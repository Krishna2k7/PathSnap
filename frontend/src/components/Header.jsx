import React from 'react';
import { Activity, Server, Cpu, RefreshCw, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Header({ backendConnected, backendUrl, checkingHealth, onRefreshHealth }) {
  return (
    <header className="bg-slate-900/90 backdrop-blur border-b border-slate-800 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-black text-white shadow-lg shadow-indigo-500/25 text-lg tracking-wider">
          PS
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-white">PathSnap</h1>
            <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Evaluator v1.0
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Distributed Map-Matching & Route Reconstruction • Client Node
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Selected Model Tag */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/80 text-xs">
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span className="text-slate-400">Selected Model:</span>
          <span className="font-semibold text-white">HistGradientBoosting</span>
        </div>

        {/* Backend Connection Status */}
        <div className="flex items-center gap-2">
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
              backendConnected
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/25'
            }`}
          >
            {backendConnected ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span>Laptop 1 Online</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-rose-400" />
                <span>Backend Offline</span>
              </>
            )}
          </div>

          <button
            onClick={onRefreshHealth}
            disabled={checkingHealth}
            title="Ping Backend (Laptop 1)"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${checkingHealth ? 'animate-spin text-indigo-400' : ''}`} />
          </button>
        </div>
      </div>
    </header>
  );
}
