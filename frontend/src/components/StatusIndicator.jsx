import React from 'react';
import { Wifi, WifiOff, Globe } from 'lucide-react';

export default function StatusIndicator({ backendConnected, backendUrl }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400 font-medium flex items-center gap-1.5">
          <Globe className="w-3.5 h-3.5 text-indigo-400" />
          Target API Host (Laptop 1)
        </span>
        <span className="font-mono text-slate-300 text-[11px] truncate max-w-[150px]">
          {backendUrl}
        </span>
      </div>

      <div className="flex items-center justify-between pt-1 border-t border-slate-800/80 text-xs">
        <span className="text-slate-400">Communication Mode</span>
        <span className="font-medium text-slate-200">Local LAN (REST)</span>
      </div>
    </div>
  );
}
