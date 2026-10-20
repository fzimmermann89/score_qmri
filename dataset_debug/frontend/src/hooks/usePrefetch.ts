import { useState, useEffect } from 'react';
import { api } from '../api';
import { SampleResponse, DisplayMode } from '../types';

interface PrefetchParams {
  sample: SampleResponse | null;
  selectedDsIdx: number;
  augment: boolean;
  seed: number;
  displayMode: DisplayMode;
  intensity: { vmin?: number; vmax?: number };
  loading: boolean;
  selectedDataset: boolean;
}

export function usePrefetch({
  sample,
  selectedDsIdx,
  augment,
  seed,
  displayMode,
  intensity,
  loading,
  selectedDataset,
}: PrefetchParams) {
  const [prefetchedSlice, setPrefetchedSlice] = useState<SampleResponse | null>(null);
  const [prefetchedRandom, setPrefetchedRandom] = useState<SampleResponse | null>(null);

  useEffect(() => {
    if (!sample || loading || !selectedDataset) return;

    const prefetch = async () => {
      if (sample.navigation.can_next_slice) {
        const nav = sample.navigation;
        const nextSliceParams = {
          subject_idx: nav.current_subject_index,
          slice_idx: nav.current_slice_index + 1,
          coil_idx: nav.coil_count ? (nav.current_coil_index ?? 0) : undefined,
          augment,
          seed,
          display_mode: displayMode,
          ...intensity,
        };

        const isCached =
          prefetchedSlice &&
          prefetchedSlice.navigation.current_subject_index === nextSliceParams.subject_idx &&
          prefetchedSlice.navigation.current_slice_index === nextSliceParams.slice_idx &&
          prefetchedSlice.navigation.current_coil_index === nextSliceParams.coil_idx;

        if (!isCached) {
          try {
            const s = await api.getSampleByHierarchy(selectedDsIdx, nextSliceParams);
            setPrefetchedSlice(s);
          } catch (e) {
            console.warn('Prefetch slice failed', e);
          }
        }
      }

      if (!prefetchedRandom) {
        try {
          const idx = await api.getRandomIndex(selectedDsIdx);
          const s = await api.getSample(selectedDsIdx, {
            index: idx,
            augment,
            seed,
            display_mode: displayMode,
            ...intensity,
          });
          setPrefetchedRandom(s);
        } catch (e) {
          console.warn('Prefetch random failed', e);
        }
      }
    };

    const timer = setTimeout(prefetch, 200);
    return () => clearTimeout(timer);
  }, [sample, selectedDsIdx, augment, seed, displayMode, intensity, prefetchedSlice, prefetchedRandom, loading, selectedDataset]);

  return {
    prefetchedSlice,
    prefetchedRandom,
    setPrefetchedSlice,
    setPrefetchedRandom,
  };
}
