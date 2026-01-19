import { useState, useEffect, useCallback } from 'react';
import { api } from './api';
import { 
  DatasetInfo, SampleResponse, AnnotationField, AnnotationValues, HistogramResponse, DisplayMode, ImageStats 
} from './types';
import { useKeyboardNavigation } from './hooks/useKeyboardNavigation';
import { usePrefetch } from './hooks/usePrefetch';
import { useDatasets } from './hooks/useDatasets';
import { DatasetSelector } from './components/DatasetSelector';
import { ImageViewer } from './components/ImageViewer';
import { GridView } from './components/GridView';
import { NavigationControls } from './components/NavigationControls';
import { StatsPanel } from './components/StatsPanel';
import { ConfigPanel } from './components/ConfigPanel';
import { Histogram } from './components/Histogram';
import { AnnotationPanel } from './components/AnnotationPanel';
import { KeyboardHelp } from './components/KeyboardHelp';

function App() {
  const { datasets, mockMode, error: datasetsError } = useDatasets();
  const [selectedDsIdx, setSelectedDsIdx] = useState<number>(0);
  const selectedDataset = datasets.find(ds => ds.dataset_idx === selectedDsIdx) || null;
  
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [augment, setAugment] = useState(false);
  const [seed, setSeed] = useState(1234);
  const [displayMode, setDisplayMode] = useState<DisplayMode>('real_imag');
  const [showImage, setShowImage] = useState(true);
  const [showPenumbra, setShowPenumbra] = useState(true);
  const [showBackground, setShowBackground] = useState(true);
  const [showHistogram, setShowHistogram] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [viewMode, setViewMode] = useState<'single' | 'grid'>('single');
  const [intensity, setIntensity] = useState<{vmin?: number, vmax?: number}>({});

  const [histData, setHistData] = useState<HistogramResponse | null>(null);
  const [histLoading, setHistLoading] = useState(false);

  const [gridStats, setGridStats] = useState<ImageStats | null>(null);
  const [gridHistogram, setGridHistogram] = useState<HistogramResponse | null>(null);
  const [gridLoading, setGridLoading] = useState(false);
  const [gridPageStart, setGridPageStart] = useState(0);
  const [gridRandomMode, setGridRandomMode] = useState(false);
  const [gridRandomRequestKey, setGridRandomRequestKey] = useState(0);
  const gridPageSize = 8;
  const sampleIndex = sample?.index ?? 0;

  const { prefetchedSlice, prefetchedRandom, setPrefetchedSlice, setPrefetchedRandom } = usePrefetch({
    sample,
    selectedDsIdx,
    augment,
    seed,
    displayMode,
    intensity,
    loading,
    selectedDataset: !!selectedDataset
  });

  const [fields, setFields] = useState<AnnotationField[]>([]);
  const [annotations, setAnnotations] = useState<AnnotationValues>({});

  useEffect(() => {
    if (datasetsError) {
      setError(datasetsError);
    }
    if (datasets.length > 0 && !sample) {
      const firstIdx = datasets[0].dataset_idx;
      setSelectedDsIdx(firstIdx);
      loadSample(firstIdx, { index: 0, augment, seed, display_mode: displayMode });
    }
  }, [datasets, datasetsError]);

  useEffect(() => {
    loadAnnotationFields();
  }, []);

  useEffect(() => {
    if (viewMode !== 'grid' || gridRandomMode) return;
    const newStart = Math.floor(sampleIndex / gridPageSize) * gridPageSize;
    setGridPageStart((prev) => (prev === newStart ? prev : newStart));
  }, [viewMode, gridRandomMode, sampleIndex, gridPageSize]);

  useEffect(() => {
    if (viewMode === 'single') {
      setGridStats(null);
      setGridHistogram(null);
      setGridLoading(false);
    }
  }, [viewMode]);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (isFullscreen) {
        setIsFullscreen(false);
        return;
      }
      if (viewMode === 'single') {
        setViewMode('grid');
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isFullscreen, viewMode]);

  const loadAnnotationFields = async () => {
    try {
      const f = await api.getAnnotationFields();
      setFields(f);
    } catch (e) { console.error(e); }
  };

  const loadSample = useCallback(async (dsIdx: number, params: any, silent: boolean = false) => {
    if (!silent) setLoading(true);
    setError(null);
    try {
      const finalParams = { ...params, ...intensity };
      
      const [s, ann] = await Promise.all([
         api.getSample(dsIdx, finalParams),
         api.getDatasetAnnotations(dsIdx)
      ]);

      setSample(s);
      setAnnotations(ann);

      if (showHistogram) {
        fetchHistogram(dsIdx, { ...finalParams, display_mode: displayMode });
      }
    } catch (e: any) {
      setError(e.message || "Error loading sample");
    } finally {
      if (!silent) setLoading(false);
    }
  }, [showHistogram, intensity, displayMode]);

  const fetchHistogram = async (dsIdx: number, params: any) => {
    setHistLoading(true);
    try {
      const h = await api.getHistogram(dsIdx, params);
      setHistData(h);
    } catch (e) { console.error(e); }
    finally { setHistLoading(false); }
  };

  const handleDatasetSelect = (idx: number) => {
    setSelectedDsIdx(idx);
    setIntensity({});
    setGridRandomMode(false);
    setGridPageStart(0);
    loadSample(idx, { index: 0, augment, seed, display_mode: displayMode });
  };

  const handleNavigate = (type: 'sample' | 'coil' | 'slice' | 'subject', direction: 'next' | 'prev') => {
    if (!sample) return;
    
    const nav = sample.navigation;
    let nextParams = { 
      subject_idx: nav.current_subject_index,
      slice_idx: nav.current_slice_index,
      coil_idx: nav.coil_count ? (nav.current_coil_index ?? 0) : undefined,
      augment, seed, display_mode: displayMode,
      ...intensity
    };

    if (type === 'subject') {
      nextParams.subject_idx += (direction === 'next' ? 1 : -1);
      nextParams.slice_idx = 0;
      nextParams.coil_idx = nav.coil_count ? 0 : undefined;
    } else if (type === 'slice') {
       if (direction === 'next' && prefetchedSlice) {
          const pNav = prefetchedSlice.navigation;
          if (pNav.current_subject_index === nextParams.subject_idx &&
              pNav.current_slice_index === nextParams.slice_idx + 1) {
             setSample(prefetchedSlice);
             setPrefetchedSlice(null);
             return;
          } else {
             console.warn("Prefetch mismatch:", pNav.current_slice_index, "vs expected", nextParams.slice_idx + 1);
          }
       }
       nextParams.slice_idx += (direction === 'next' ? 1 : -1);
    } else if (type === 'coil') {
       if (!nav.coil_count) return;
       nextParams.coil_idx = (nextParams.coil_idx ?? 0) + (direction === 'next' ? 1 : -1);
    } else if (type === 'sample') {
       const delta = direction === 'next' ? 1 : -1;
       const newIdx = Math.max(0, Math.min(nav.total_samples - 1, nav.linear_index + delta));
       loadSample(selectedDsIdx, { index: newIdx, augment, seed, display_mode: displayMode }, true);
       return;
    }

    api.getSampleByHierarchy(selectedDsIdx, nextParams)
      .then(s => {
        setSample(s);
      })
      .catch(e => {
        console.warn("Navigation failed", e);
      });
  };

  const handleGridPrevPage = () => {
    if (!selectedDataset) return;
    if (gridRandomMode) {
      setGridRandomRequestKey((prev) => prev + 1);
      return;
    }
    setGridPageStart(Math.max(0, gridPageStart - gridPageSize));
  };

  const handleGridNextPage = () => {
    if (!selectedDataset) return;
    if (gridRandomMode) {
      setGridRandomRequestKey((prev) => prev + 1);
      return;
    }
    const nextStart = gridPageStart + gridPageSize;
    if (nextStart < selectedDataset.total_samples) {
      setGridPageStart(nextStart);
    }
  };

  const handleNextSampleKey = () => {
    if (viewMode === 'grid') {
      handleGridNextPage();
      return;
    }
    handleNavigate('sample', 'next');
  };

  const handlePrevSampleKey = () => {
    if (viewMode === 'grid') {
      handleGridPrevPage();
      return;
    }
    handleNavigate('sample', 'prev');
  };

  const handleGridStatsUpdate = useCallback((stats: ImageStats, histogram: HistogramResponse) => {
    setGridStats(stats);
    setGridHistogram(histogram);
  }, []);

  const handleGridLoadingChange = useCallback((value: boolean) => {
    setGridLoading(value);
  }, []);

  useKeyboardNavigation({
    onNextSample: handleNextSampleKey,
    onPrevSample: handlePrevSampleKey,
    onNextCoil: () => {
      if (viewMode === 'grid') return;
      handleNavigate('coil', 'next');
    },
    onPrevCoil: () => {
      if (viewMode === 'grid') return;
      handleNavigate('coil', 'prev');
    },
    onNextSlice: () => {
      if (viewMode === 'grid') return;
      handleNavigate('slice', 'next');
    },
    onPrevSlice: () => {
      if (viewMode === 'grid') return;
      handleNavigate('slice', 'prev');
    },
    onNextSubject: () => {
      if (viewMode === 'grid') return;
      handleNavigate('subject', 'next');
    },
    onPrevSubject: () => {
      if (viewMode === 'grid') return;
      handleNavigate('subject', 'prev');
    },
    onRandomSample: async () => {
      if (viewMode === 'grid') {
        setGridRandomMode(true);
        return;
      }
      if (prefetchedRandom) {
         setSample(prefetchedRandom);
         setPrefetchedRandom(null);
      } else {
        const idx = await api.getRandomIndex(selectedDsIdx);
        loadSample(selectedDsIdx, { index: idx, augment, seed, display_mode: displayMode });
      }
    },
    onJumpToIndex: () => {
      const idxStr = prompt("Jump to index:");
      if (idxStr && !isNaN(parseInt(idxStr))) {
        loadSample(selectedDsIdx, { index: parseInt(idxStr), augment, seed, display_mode: displayMode });
      }
    },
    onToggleAugment: () => {
      const newVal = !augment;
      setAugment(newVal);
      if (viewMode === 'single' && sample) {
        loadSample(selectedDsIdx, { index: sample.index, augment: newVal, seed, display_mode: displayMode });
      }
    },
    onNewAugmentSeed: () => {
      const newSeed = seed + 1;
      setSeed(newSeed);
      if (viewMode === 'single' && sample) {
        loadSample(selectedDsIdx, { index: sample.index, augment, seed: newSeed, display_mode: displayMode });
      }
    },
    onToggleMasks: () => { setShowPenumbra(p => !p); setShowBackground(b => !b); },
    onTogglePenumbra: () => setShowPenumbra(p => !p),
    onToggleBackground: () => setShowBackground(b => !b),
    onToggleHistogram: () => {
      const newVal = !showHistogram;
      setShowHistogram(newVal);
      if (viewMode === 'single' && newVal && sample) {
        fetchHistogram(selectedDsIdx, { index: sample.index, augment, seed, display_mode: displayMode });
      }
    },
    onToggleDisplayMode: () => {
      const newMode = displayMode === 'real_imag' ? 'mag_phase' : 'real_imag';
      setDisplayMode(newMode);
      if (viewMode === 'single' && sample) {
        loadSample(selectedDsIdx, { index: sample.index, augment, seed, display_mode: newMode });
      }
    },
    onToggleFullscreen: () => setIsFullscreen(f => !f),
    onToggleHelp: () => setShowHelp(h => !h),
    onExportAnnotations: () => window.open(api.exportAnnotationsUrl, '_blank'),
    onSaveAnnotations: () => api.setDatasetAnnotations(selectedDsIdx, annotations),
    onAddAnnotationField: () => { },
    onNextDataset: () => {
      const sortedDs = [...datasets].sort((a, b) => a.dataset_idx - b.dataset_idx);
      const currentIdx = sortedDs.findIndex(d => d.dataset_idx === selectedDsIdx);
      if (currentIdx < sortedDs.length - 1) {
        handleDatasetSelect(sortedDs[currentIdx + 1].dataset_idx);
      }
    },
    onPrevDataset: () => {
      const sortedDs = [...datasets].sort((a, b) => a.dataset_idx - b.dataset_idx);
      const currentIdx = sortedDs.findIndex(d => d.dataset_idx === selectedDsIdx);
      if (currentIdx > 0) {
        handleDatasetSelect(sortedDs[currentIdx - 1].dataset_idx);
      }
    }
  });

  const activeStats = viewMode === 'grid' ? gridStats : sample?.stats;
  const activeHistogram = viewMode === 'grid' ? gridHistogram : histData;
  const activeHistogramLoading = viewMode === 'grid' ? gridLoading : histLoading;

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-950 text-red-500 flex-col gap-4">
        <h2 className="text-xl font-bold">Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()} className="px-4 py-2 bg-gray-800 rounded">Reload</button>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col p-3 gap-2 overflow-hidden font-sans">
      <header className="flex justify-between items-center shrink-0">
        <div className="flex gap-3 items-center">
          <h1 className="text-xl font-black tracking-tight text-blue-500 flex items-center gap-2">
            <span>🔬</span> DATASET DEBUG
          </h1>
          {mockMode && (
            <div className="bg-amber-600/15 border border-amber-500/40 px-2 py-0.5 text-center rounded-md ml-4">
              <div className="inline-flex items-center gap-2 text-amber-400 text-[10px]">
                <span>⚠️</span>
                <span className="font-medium">Using mock data</span>
              </div>
            </div>
          )}
        </div>
        <div className="flex gap-2 items-center">
          <div className="flex bg-gray-900 rounded-lg p-1 border border-gray-800">
            <button 
              onClick={() => setViewMode('single')}
              className={`px-3 py-1 text-sm rounded transition-colors ${viewMode === 'single' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
            >
              Single
            </button>
            <button 
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1 text-sm rounded transition-colors ${viewMode === 'grid' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
            >
              Grid
            </button>
          </div>
          <button onClick={() => setShowHelp(true)} className="btn-secondary text-xs px-3 py-1.5">
            Shortcuts [?]
          </button>
        </div>
      </header>

      <div className="flex-grow grid grid-cols-[240px_minmax(0,1fr)_240px] gap-2 min-h-0">
        
        <div className="flex flex-col gap-2 overflow-y-auto pr-1 min-h-0">
          {datasets.length > 0 && (
            <DatasetSelector 
              datasets={datasets} 
              selectedIdx={selectedDsIdx} 
              onSelect={handleDatasetSelect} 
            />
          )}
          {sample && <ConfigPanel config={selectedDataset?.config} />}
          {activeStats && <StatsPanel stats={activeStats} />}
          
          <div className="bg-gray-900 border border-gray-800 rounded-lg p-3">
             <h3 className="text-sm font-bold text-gray-300 uppercase mb-2">View Settings</h3>
             <div className="space-y-2 text-sm">
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={augment} onChange={() => {
                    setAugment(!augment);
                    if(sample) loadSample(selectedDsIdx, { index: sample.index, augment: !augment, seed, display_mode: displayMode });
                  }} className="h-4 w-4" /> Augmentation (T)
                </label>
                <div className="flex items-center gap-2 pl-5">
                  <span className="text-gray-500">Seed:</span>
                  <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value))} className="w-20 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm" />
                   <button onClick={() => {
                     const newSeed = seed + 1;
                     setSeed(newSeed);
                     if(sample) loadSample(selectedDsIdx, { index: sample.index, augment, seed: newSeed, display_mode: displayMode });
                   }} className="text-blue-400 hover:text-blue-300 text-sm">New (Space)</button>
                 </div>

              </div>
           </div>

          <div className="bg-gray-900 border border-gray-800 rounded-lg p-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-gray-400 uppercase">Histogram</h3>
              <button 
                onClick={() => {
                  const n = !showHistogram;
                  setShowHistogram(n);
                  if (n && viewMode === 'single' && sample) {
                    fetchHistogram(selectedDsIdx, { index: sample.index, augment, seed, display_mode: displayMode });
                  }
                }}
                className={`text-[11px] px-2 py-1 rounded border ${showHistogram ? 'bg-blue-900 border-blue-700 text-white' : 'bg-gray-900 border-gray-700 text-gray-400'}`}
              >
                {showHistogram ? 'On' : 'Off'}
              </button>
            </div>
            {showHistogram && <Histogram data={activeHistogram} loading={activeHistogramLoading} onClose={() => setShowHistogram(false)} />}
          </div>
        </div>

        <div className="flex flex-col gap-3 min-w-0 min-h-0 relative">
          <div className="flex-grow min-h-0 bg-gray-900/80 rounded-xl relative overflow-hidden border border-gray-800 shadow-inner">
            {viewMode === 'grid' && selectedDataset ? (
               <GridView 
                  dataset={selectedDataset}
                  pageStart={gridPageStart}
                  pageSize={gridPageSize}
                  isRandomMode={gridRandomMode}
                  randomRequestKey={gridRandomRequestKey}
                  onPageStartChange={setGridPageStart}
                  onRandomModeChange={setGridRandomMode}
                  augment={augment}
                  seed={seed}
                  displayMode={displayMode}
                  onSelectSample={(idx) => {
                    setViewMode('single');
                    loadSample(selectedDsIdx, { index: idx, augment, seed, display_mode: displayMode });
                  }}
                  onUpdateGlobalStats={handleGridStatsUpdate}
                  onLoadingChange={handleGridLoadingChange}
               />
            ) : (
              <>
                {sample ? (
                  <ImageViewer 
                    sample={sample} 
                    showPenumbra={showPenumbra} 
                    showBackground={showBackground}
                    isFullscreen={isFullscreen}
                    onCloseFullscreen={() => setIsFullscreen(false)}
                     onIntensityChange={(vmin, vmax) => {
                        setIntensity({ vmin, vmax });
                        loadSample(selectedDsIdx, { index: sample.index, augment, seed, display_mode: displayMode, vmin, vmax });
                     }}
                     displayMode={displayMode}
                     onDisplayModeChange={(m) => {
                        setDisplayMode(m);
                        if(sample) loadSample(selectedDsIdx, { index: sample.index, augment, seed, display_mode: m });
                     }}
                     showImage={showImage}
                     onToggleImage={() => setShowImage(!showImage)}
                     onTogglePenumbra={() => setShowPenumbra(!showPenumbra)}
                     onToggleBackground={() => setShowBackground(!showBackground)}
                   />
                 ) : (
                   <div className="absolute inset-0 flex items-center justify-center text-gray-500">No Sample Loaded</div>
                 )}
                {loading && (
                  <div className="absolute inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-20">
                    <div className="animate-spin text-4xl">⏳</div>
                  </div>
                )}
              </>
            )}
          </div>

          {viewMode === 'single' && sample && (
             <>
                <NavigationControls 
                  nav={sample.navigation}
                  onNextSample={() => handleNavigate('sample', 'next')}
                  onPrevSample={() => handleNavigate('sample', 'prev')}
                  onNextCoil={() => handleNavigate('coil', 'next')}
                  onPrevCoil={() => handleNavigate('coil', 'prev')}
                  onNextSlice={() => handleNavigate('slice', 'next')}
                  onPrevSlice={() => handleNavigate('slice', 'prev')}
                  onNextSubject={() => handleNavigate('subject', 'next')}
                  onPrevSubject={() => handleNavigate('subject', 'prev')}
                  onRandom={async () => {
                    const idx = await api.getRandomIndex(selectedDsIdx);
                    loadSample(selectedDsIdx, { index: idx, augment, seed, display_mode: displayMode });
                  }}
                  onJumpToIndex={() => {
                    const idxStr = prompt("Jump to index:");
                    if (idxStr && !isNaN(parseInt(idxStr))) {
                      loadSample(selectedDsIdx, { index: parseInt(idxStr), augment, seed, display_mode: displayMode });
                    }
                  }}
                />
             </>
          )}
        </div>

        <div className="flex flex-col h-full overflow-y-auto min-h-0 gap-3 pr-1">
           <AnnotationPanel 
              datasetName={selectedDataset?.h5_filename || ''}
              fields={fields}
              values={annotations}
              onAddField={async (f) => {
                await api.addAnnotationField(f);
                loadAnnotationFields();
              }}
              onRemoveField={async (id) => {
                if(confirm('Delete this field definition?')) {
                  await api.removeAnnotationField(id);
                  loadAnnotationFields();
                }
              }}
              onUpdateValue={async (id, val) => {
                const newVals = { ...annotations, [id]: val };
                setAnnotations(newVals);
                await api.setDatasetAnnotations(selectedDsIdx, newVals);
              }}
              onExport={() => window.open(api.exportAnnotationsUrl)}
            />
        </div>
      </div>
      
      <KeyboardHelp isOpen={showHelp} onClose={() => setShowHelp(false)} />
    </div>
  );
}

export default App;
