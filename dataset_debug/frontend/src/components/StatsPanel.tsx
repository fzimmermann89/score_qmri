import React from 'react';
import { ImageStats } from '../types';

export const StatsPanel: React.FC<{ stats: ImageStats }> = ({ stats }) => {
  const fmt = (v: number) => {
    if (v === undefined) return '-';
    return Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(2) : v.toFixed(4);
  };
  const fmtPct = (v: number) => (v * 100).toFixed(1) + '%';

  if (stats.components) {
    const { real, imag, abs } = stats.components;
    
    const Row = ({ label, k, isPct }: { label: string, k: keyof ImageStats, isPct?: boolean }) => {
      const r = real[k] as number;
      const i = imag[k] as number;
      const a = abs[k] as number;
      
      return (
         <div className="grid grid-cols-[40px_1fr_1fr_1fr] text-sm border-b border-gray-800 last:border-0 py-1 gap-1">
            <span className="text-gray-400 uppercase tracking-wider text-[10px] self-center">{label}</span>
            <span className="font-mono text-gray-100 text-[10px] text-right truncate" title={r?.toString()}>
              {isPct ? fmtPct(r) : fmt(r)}
            </span>
            <span className="font-mono text-gray-100 text-[10px] text-right truncate" title={i?.toString()}>
              {isPct ? fmtPct(i) : fmt(i)}
            </span>
            <span className="font-mono text-gray-100 text-[10px] text-right truncate" title={a?.toString()}>
              {isPct ? fmtPct(a) : fmt(a)}
            </span>
         </div>
      );
    };

    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
         <div className="grid grid-cols-[40px_1fr_1fr_1fr] text-[10px] font-bold text-gray-400 uppercase mb-2 gap-1 border-b border-gray-800 pb-1">
            <span>Stat</span>
            <span className="text-right text-blue-400">Re</span>
            <span className="text-right text-green-400">Im</span>
            <span className="text-right text-yellow-400">Abs</span>
         </div>
         <div className="flex flex-col gap-0">
           <Row label="Min" k="min" />
           <Row label="Max" k="max" />
           <Row label="99%" k="p99" />
           <Row label="99.9" k="p99_9" />
           <Row label="Mean" k="mean" />
           <Row label="Std" k="std" />
           <Row label="Sig%" k="signal_fraction" isPct />
         </div>
      </div>
    );
  }

  const StatItem = ({ label, value }: { label: string, value: string }) => (
    <div className="flex justify-between items-center text-sm border-b border-gray-800 last:border-0 py-1.5">
      <span className="text-gray-400 uppercase tracking-wider text-[11px]">{label}</span>
      <span className="font-mono text-gray-100 text-sm">{value}</span>
    </div>
  );

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
      <h3 className="text-sm font-bold text-gray-300 uppercase mb-2">Image Statistics</h3>
      <div className="flex flex-col gap-0">
        <StatItem label="Min" value={fmt(stats.min)} />
        <StatItem label="Max" value={fmt(stats.max)} />
        <StatItem label="99%" value={fmt(stats.p99)} />
        <StatItem label="99.9%" value={fmt(stats.p99_9)} />
        <StatItem label="Mean" value={fmt(stats.mean)} />
        <StatItem label="Std Dev" value={fmt(stats.std)} />
        <StatItem label="Variance" value={stats.variance.toExponential(2)} />
        <StatItem label="Signal %" value={fmtPct(stats.signal_fraction)} />
      </div>
    </div>
  );
};
