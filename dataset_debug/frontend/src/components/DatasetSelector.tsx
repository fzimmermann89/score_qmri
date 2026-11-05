import React from 'react';
import { DatasetInfo } from '../types';

interface Props {
  datasets: DatasetInfo[];
  selectedIdx: number;
  onSelect: (idx: number) => void;
}

export const DatasetSelector: React.FC<Props> = ({ datasets, selectedIdx, onSelect }) => {
  return (
    <div className="flex flex-col gap-1 w-full max-w-sm">
      <label className="text-sm font-bold text-gray-400 uppercase tracking-wider">
        Active Dataset
      </label>
      <select
        value={selectedIdx}
        onChange={(e) => onSelect(Number(e.target.value))}
        className="bg-gray-800 border border-gray-700 text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full px-3 py-2"
      >
        {datasets.map((ds) => (
          <option key={ds.dataset_idx} value={ds.dataset_idx}>
            [{ds.dataset_idx}] {ds.h5_filename} ({ds.total_samples} samples) - {ds.kind}
          </option>
        ))}
      </select>
      <div className="text-sm text-gray-500 px-1">
        Type: {datasets.find(ds => ds.dataset_idx === selectedIdx)?.config?.type} | Mode: {datasets.find(ds => ds.dataset_idx === selectedIdx)?.config?.mode || 'N/A'}
      </div>
    </div>
  );
};
