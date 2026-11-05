import { useState, useEffect, useCallback } from 'react';
import { api } from '../api';
import { DatasetInfo } from '../types';

export function useDatasets() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [mockMode, setMockMode] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      setMockMode(data.mock_mode || false);
    } catch (e) { console.error(e); }
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      const data = await api.getDatasets();
      setDatasets(data.datasets);
      setMockMode(data.mock_mode || false);
      return data.datasets;
    } catch (e) {
      console.error(e);
      setError("Failed to load datasets. Is backend running?");
      return [];
    }
  }, []);

  useEffect(() => {
    loadDatasets();
    loadStatus();
  }, [loadDatasets, loadStatus]);

  return { datasets, mockMode, error, loadDatasets };
}
