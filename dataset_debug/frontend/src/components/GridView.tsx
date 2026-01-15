import React, { useState, useEffect } from 'react';
import { DatasetInfo, SampleResponse } from '../types';
import { api } from '../api';

interface Props {
  dataset: DatasetInfo;
  startIndex: number;
  onSelectSample: (index: number) => void;
  augment: boolean;
  seed: number;
  displayMode: 'real_imag' | 'mag_phase';
}

const PAGE_SIZE = 9;

export const GridView: React.FC<Props> = ({ 
  dataset, startIndex, onSelectSample, augment, seed, displayMode 
}) => {
  const [samples, setSamples] = useState<(SampleResponse | null)[]>([]);
  const [loading, setLoading] = useState(false);
  const [pageStart, setPageStart] = useState(startIndex);

  useEffect(() => {
    setPageStart(Math.floor(startIndex / PAGE_SIZE) * PAGE_SIZE);
  }, [startIndex]);

  useEffect(() => {
    const fetchGrid = async () => {
      setLoading(true);
      const promises = [];
      for (let i = 0; i < PAGE_SIZE; i++) {
        const idx = pageStart + i;
        if (idx >= dataset.total_samples) {
          promises.push(Promise.resolve(null));
        } else {
          promises.push(api.getSample(dataset.dataset_idx, {
            index: idx,
            augment,
            seed,
            display_mode: displayMode
          }).catch(e => {
            console.error(`Failed to load sample ${idx}`, e);
            return null;
          }));
        }
      }
      
      const results = await Promise.all(promises);
      setSamples(results);
      setLoading(false);
    };

    fetchGrid();
  }, [dataset.dataset_idx, pageStart, augment, seed, displayMode]);

  const handlePrev = () => setPageStart(Math.max(0, pageStart - PAGE_SIZE));
  const handleNext = () => {
    if (pageStart + PAGE_SIZE < dataset.total_samples) {
      setPageStart(pageStart + PAGE_SIZE);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-950 p-4 gap-4">
      <div className="flex justify-between items-center bg-gray-900 p-2 rounded-lg border border-gray-800">
        <div className="text-sm font-mono text-gray-400">
          Samples {pageStart} - {Math.min(pageStart + PAGE_SIZE - 1, dataset.total_samples - 1)} 
          <span className="text-gray-600 mx-2">/</span> 
          {dataset.total_samples}
        </div>
        <div className="flex gap-2">
          <button 
            onClick={handlePrev} 
            disabled={pageStart === 0 || loading}
            className="glass-button disabled:opacity-30"
          >
            ← Prev Page
          </button>
          <button 
            onClick={handleNext} 
            disabled={pageStart + PAGE_SIZE >= dataset.total_samples || loading}
            className="glass-button disabled:opacity-30"
          >
            Next Page →
          </button>
        </div>
      </div>

      <div className="flex-grow grid grid-cols-3 gap-4 min-h-0 overflow-y-auto">
        {loading ? (
          <div className="col-span-3 flex items-center justify-center text-gray-500 animate-pulse">
            Loading grid...
          </div>
        ) : (
          samples.map((s, i) => {
            if (!s) return <div key={i} className="bg-gray-900/30 rounded-lg border border-gray-800/50" />;
            
            const img = s.image_left_base64;
            
            return (
              <button 
                key={s.index}
                onClick={() => onSelectSample(s.index)}
                className="relative bg-gray-900 rounded-lg border border-gray-800 overflow-hidden hover:border-blue-500 transition-colors group text-left"
              >
                <div className="absolute top-2 left-2 z-10 bg-black/70 px-2 py-0.5 rounded text-[10px] font-mono text-gray-300">
                  #{s.index}
                </div>
                
                <div className="w-full aspect-square flex items-center justify-center p-2">
                  <img src={img} className="max-w-full max-h-full object-contain pixelated" style={{ imageRendering: 'pixelated' }} />
                </div>
                
                <div className="bg-gray-950 px-2 py-1 text-[10px] text-gray-500 font-mono border-t border-gray-800 truncate">
                  {s.image_left_label}
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
};
