import React from 'react';
import { ImageStats } from '../types';

export const StatsPanel: React.FC<{ stats: ImageStats }> = ({ stats }) => {
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
        <StatItem label="Min" value={stats.min.toFixed(4)} />
        <StatItem label="Max" value={stats.max.toFixed(4)} />
        <StatItem label="Mean" value={stats.mean.toFixed(4)} />
        <StatItem label="Std Dev" value={stats.std.toFixed(4)} />
        <StatItem label="Variance" value={stats.variance.toExponential(2)} />
        <StatItem label="Signal %" value={(stats.signal_fraction * 100).toFixed(1) + '%'} />
      </div>
    </div>
  );
};
