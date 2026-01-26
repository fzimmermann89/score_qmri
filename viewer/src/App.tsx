import { useState, useEffect, useCallback } from 'react'
import { Header } from './components/Header'
import { SampleGrid } from './components/SampleGrid'
import {
  fetchDatasets,
  fetchKinds,
  fetchSamplesDataset,
  fetchSamplesKind,
  reloadConfig,
} from './api'
import type { Dataset, KindInfo, SampleData, ViewerMode } from './types'

function App() {
  // State
  const [mode, setMode] = useState<ViewerMode>('dataset')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [datasetsByKind, setDatasetsByKind] = useState<Record<string, Dataset[]>>({})
  const [kinds, setKinds] = useState<KindInfo[]>([])
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null)
  const [selectedKind, setSelectedKind] = useState<string | null>(null)
  const [sampleData, setSampleData] = useState<SampleData | null>(null)
  const [showPenumbra, setShowPenumbra] = useState(false)
  const [showBackground, setShowBackground] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mockMode, setMockMode] = useState(false)

  // Load initial data
  useEffect(() => {
    async function init() {
      try {
        const [dsRes, kindRes] = await Promise.all([fetchDatasets(), fetchKinds()])
        setDatasets(dsRes.datasets)
        setDatasetsByKind(dsRes.by_kind)
        setKinds(kindRes.kinds)
        setMockMode(dsRes.mock_mode)

        // Set initial selection
        if (dsRes.datasets.length > 0) {
          setSelectedDataset(dsRes.datasets[0].dataset_idx)
        }
        if (kindRes.kinds.length > 0) {
          setSelectedKind(kindRes.kinds[0].name)
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load data')
      }
    }
    init()
  }, [])

  // Fetch samples
  const fetchSamples = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let data: SampleData
      if (mode === 'dataset' && selectedDataset !== null) {
        data = await fetchSamplesDataset(selectedDataset, 32)
      } else if (mode === 'kind' && selectedKind !== null) {
        data = await fetchSamplesKind(selectedKind, 32)
      } else {
        return
      }
      setSampleData(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch samples')
    } finally {
      setLoading(false)
    }
  }, [mode, selectedDataset, selectedKind])

  // Auto-fetch on selection change
  useEffect(() => {
    if ((mode === 'dataset' && selectedDataset !== null) || (mode === 'kind' && selectedKind !== null)) {
      fetchSamples()
    }
  }, [mode, selectedDataset, selectedKind, fetchSamples])

  // Spacebar handler
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault()
        fetchSamples()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [fetchSamples])

  // Reload handler
  const handleReload = useCallback(async () => {
    try {
      setLoading(true)
      await reloadConfig()
      const [dsRes, kindRes] = await Promise.all([fetchDatasets(), fetchKinds()])
      setDatasets(dsRes.datasets)
      setDatasetsByKind(dsRes.by_kind)
      setKinds(kindRes.kinds)
      setMockMode(dsRes.mock_mode)
      
      // Reset selection if needed
      if (selectedDataset !== null && !dsRes.datasets.some(d => d.dataset_idx === selectedDataset)) {
        setSelectedDataset(dsRes.datasets[0]?.dataset_idx ?? null)
      }
      if (selectedKind !== null && !kindRes.kinds.some(k => k.name === selectedKind)) {
        setSelectedKind(kindRes.kinds[0]?.name ?? null)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reload config')
    } finally {
      setLoading(false)
    }
  }, [selectedDataset, selectedKind])

  return (
    <>
      <Header
        mode={mode}
        onModeChange={setMode}
        datasets={datasets}
        datasetsByKind={datasetsByKind}
        kinds={kinds}
        selectedDataset={selectedDataset}
        onDatasetChange={setSelectedDataset}
        selectedKind={selectedKind}
        onKindChange={setSelectedKind}
        showPenumbra={showPenumbra}
        onPenumbraChange={setShowPenumbra}
        showBackground={showBackground}
        onBackgroundChange={setShowBackground}
        onReload={handleReload}
        mockMode={mockMode}
      />
      <SampleGrid
        sampleData={sampleData}
        showPenumbra={showPenumbra}
        showBackground={showBackground}
        showLabels={mode === 'kind'}
        loading={loading}
        error={error}
      />
    </>
  )
}

export default App
