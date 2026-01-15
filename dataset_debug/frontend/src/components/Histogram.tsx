import React from 'react';
import { HistogramResponse } from '../types';

interface Props {
  data: HistogramResponse | null;
  loading: boolean;
  onClose: () => void;
}

export const Histogram: React.FC<Props> = ({ data, loading, onClose }) => {
  if (loading) {
    return <div className="h-32 flex items-center justify-center text-xs text-gray-500 animate-pulse">Loading histogram...</div>;
  }
  if (!data) return null;

  const counts = data.counts;
  const sorted = [...counts].sort((a, b) => a - b);
  const percentileIndex = Math.max(0, Math.floor(sorted.length * 0.95) - 1);
  const displayMax = Math.max(1, sorted[percentileIndex]);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-2 relative mt-1">
      <button 
        onClick={onClose}
        className="absolute top-2 right-2 text-gray-600 hover:text-white"
      >✕</button>
      <h3 className="text-xs font-bold text-gray-400 uppercase mb-1">Histogram</h3>
      
      <div className="h-24 flex items-end gap-[1px] w-full">
        {counts.map((count, i) => (
          <div 
            key={i}
            className="flex-1 bg-blue-500/60 hover:bg-blue-400 transition-colors"
            style={{ height: `${Math.min(100, (count / displayMax) * 100)}%` }}
            title={`Range: ${data.bins[i].toFixed(2)} - ${data.bins[i + 1]?.toFixed(2)}\nCount: ${count}`}
          />
        ))}
      </div>
      
      <div className="flex justify-between text-[11px] text-gray-400 mt-1 font-mono">
        <span>{data.bins[0].toFixed(2)}</span>
        <span>{data.bins[data.bins.length - 1].toFixed(2)}</span>
      </div>
      
      <div className="flex gap-4 mt-1 justify-center text-[11px] font-mono text-gray-300">
        <span>μ: {data.stats.mean.toFixed(2)}</span>
        <span>σ: {data.stats.std.toFixed(2)}</span>
      </div>
    </div>
  );
};
