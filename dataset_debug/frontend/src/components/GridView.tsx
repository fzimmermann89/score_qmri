import React, { useState, useEffect, useCallback } from 'react';
import { DatasetInfo, BatchSampleResponse, HistogramResponse, ImageStats, DisplayMode } from '../types';
import { api } from '../api';

interface Props {
  dataset: DatasetInfo;
  pageStart: number;
  pageSize: number;
  isRandomMode: boolean;
  randomRequestKey: number;
  onPageStartChange: (start: number) => void;
  onRandomModeChange: (enabled: boolean) => void;
  onSelectSample: (index: number) => void;
  augment: boolean;
  seed: number;
  displayMode: DisplayMode;
  onUpdateGlobalStats: (stats: ImageStats, histogram: HistogramResponse) => void;
  onLoadingChange: (loading: boolean) => void;
}

export const GridView: React.FC<Props> = ({
  dataset,
  pageStart,
  pageSize,
  isRandomMode,
  randomRequestKey,
  onPageStartChange,
  onRandomModeChange,
  onSelectSample,
  augment,
  seed,
  displayMode,
  onUpdateGlobalStats,
  onLoadingChange,
}) => {
  const [samples, setSamples] = useState<BatchSampleResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [randomIndices, setRandomIndices] = useState<number[]>([]);
  const [globalRange, setGlobalRange] = useState<[number, number] | null>(null);

  const setLoadingState = useCallback((value: boolean) => {
    setLoading(value);
    onLoadingChange(value);
  }, [onLoadingChange]);

  const fetchBatch = useCallback(async (indices: number[]) => {
    setLoadingState(true);
    try {
      const data = await api.getBatchData(dataset.dataset_idx, {
        indices,
        augment,
        seed,
        display_mode: displayMode,
      });

      setSamples(data.samples);
      setGlobalRange(data.global_range);
      onUpdateGlobalStats(data.global_stats, data.global_histogram);
    } catch (e) {
      console.error('Batch fetch failed', e);
    } finally {
      setLoadingState(false);
    }
  }, [dataset.dataset_idx, augment, seed, displayMode, onUpdateGlobalStats, setLoadingState]);

  const refreshRandom = useCallback(() => {
    const count = Math.min(pageSize, dataset.total_samples);
    const indices = new Set<number>();
    while (indices.size < count) {
      indices.add(Math.floor(Math.random() * dataset.total_samples));
    }
    setRandomIndices(Array.from(indices));
    onRandomModeChange(true);
  }, [dataset.total_samples, pageSize, onRandomModeChange]);

  useEffect(() => {
    if (isRandomMode && randomIndices.length === 0) {
      refreshRandom();
    }
  }, [isRandomMode, randomIndices.length, refreshRandom]);

  useEffect(() => {
    if (!isRandomMode) return;
    if (randomRequestKey === 0) return;
    refreshRandom();
  }, [randomRequestKey, isRandomMode, refreshRandom]);

  const handleSequential = () => {
    onRandomModeChange(false);
    onPageStartChange(Math.floor(pageStart / pageSize) * pageSize);
  };

  const handlePrev = () => {
    if (isRandomMode) {
      refreshRandom();
      return;
    }
    onPageStartChange(Math.max(0, pageStart - pageSize));
  };

  const handleNext = () => {
    if (isRandomMode) {
      refreshRandom();
      return;
    }
    if (pageStart + pageSize < dataset.total_samples) {
      onPageStartChange(pageStart + pageSize);
    }
  };

  useEffect(() => {
    if (isRandomMode) {
      if (randomIndices.length > 0) {
        fetchBatch(randomIndices);
      }
      return;
    }

    const end = Math.min(pageStart + pageSize, dataset.total_samples);
    const indices = [];
    for (let i = pageStart; i < end; i++) {
      indices.push(i);
    }
    if (indices.length > 0) {
      fetchBatch(indices);
    }
  }, [pageStart, pageSize, dataset.total_samples, isRandomMode, randomIndices, fetchBatch]);

  useEffect(() => {
    if (!isRandomMode) {
      setRandomIndices([]);
    }
  }, [isRandomMode]);

  const colorbar = globalRange ? (
    <div className="w-8 bg-gray-950 border border-gray-800 rounded flex flex-col items-center py-2">
      <span className="text-[9px] text-gray-400 font-mono">{globalRange[1].toFixed(2)}</span>
      <div
        className="flex-1 w-2 rounded-sm my-1"
        style={{ background: 'linear-gradient(to top, #000, #fff)' }}
      />
      <span className="text-[9px] text-gray-400 font-mono">{globalRange[0].toFixed(2)}</span>
    </div>
  ) : null;

  return (
    <div className="flex flex-col h-full min-h-0 bg-gray-950 p-3 gap-3">
      <div className="flex justify-between items-center bg-gray-900 p-2 rounded-lg border border-gray-800 shrink-0">
        <div className="text-sm font-mono text-gray-400">
          {isRandomMode ? (
            <span className="text-blue-400">Random Selection ({samples.length})</span>
          ) : (
            <>
              Samples {pageStart} - {Math.min(pageStart + pageSize - 1, dataset.total_samples - 1)}
              <span className="text-gray-600 mx-2">/</span>
              {dataset.total_samples}
            </>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleSequential}
            className={`px-3 py-1 text-xs rounded border transition-colors ${!isRandomMode ? 'bg-blue-600 border-blue-500 text-white' : 'bg-gray-800 border-gray-700 text-gray-400'}`}
          >
            Sequential
          </button>
          <button
            onClick={refreshRandom}
            className={`px-3 py-1 text-xs rounded border transition-colors ${isRandomMode ? 'bg-blue-600 border-blue-500 text-white' : 'bg-gray-800 border-gray-700 text-gray-400'}`}
          >
            Random {pageSize}
          </button>
          <div className="w-px bg-gray-800 mx-2" />
          <button
            onClick={handlePrev}
            disabled={loading || (!isRandomMode && pageStart === 0)}
            className="glass-button disabled:opacity-30"
          >
            ← Prev
          </button>
          <button
            onClick={handleNext}
            disabled={loading || (!isRandomMode && pageStart + pageSize >= dataset.total_samples)}
            className="glass-button disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex gap-3">
        <div className="flex-1 min-h-0 grid grid-cols-4 auto-rows-fr gap-3 overflow-y-auto pr-1">
          {loading && samples.length === 0 ? (
            <div className="col-span-4 flex items-center justify-center text-gray-500 animate-pulse h-64">
              Loading grid...
            </div>
          ) : (
            samples.map((s) => {
              const aspectRatio = s.shape ? `${s.shape[1]} / ${s.shape[0]}` : '1 / 1';
              return (
                <button
                  key={s.index}
                  onClick={() => onSelectSample(s.index)}
                  className="relative bg-black rounded-lg border border-gray-800 overflow-hidden hover:border-blue-500 transition-colors text-left"
                  style={{ aspectRatio }}
                >
                  <div className="absolute top-1 left-1 z-10 bg-black/70 px-1.5 py-0.5 rounded text-[9px] font-mono text-gray-300 pointer-events-none">
                    #{s.index}
                  </div>
                  <img
                    src={s.image_left_base64}
                    className="w-full h-full object-contain"
                    style={{ imageRendering: 'pixelated' }}
                    draggable={false}
                  />
                </button>
              );
            })
          )}
        </div>
        {colorbar}
      </div>
    </div>
  );
};
