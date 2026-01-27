import { useState, useEffect, useCallback } from 'react'
import { Header } from './components/Header'
import { SampleGrid } from './components/SampleGrid'
import { Toast, showToast } from './components/Toast'
import {
  fetchDatasets,
  fetchKinds,
  fetchSamplesDataset,
  fetchSamplesKind,
  fetchStatus,
  reloadConfig,
  warmDataset,
  warmKind,
} from './api'
import type { Dataset, KindInfo, SampleData, ViewerMode } from './types'

// Number of samples to fetch (3 rows x 7 cols)
const N_SAMPLES = 21

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
  const [initializing, setInitializing] = useState(true)

  // Load initial data with startup status polling
  useEffect(() => {
    let cancelled = false

    async function waitForReady() {
      // Poll status until ready (server prewarms datasets)
      for (let i = 0; i < 30; i++) {
        try {
        const status = await fetchStatus()
        if (status.ready) break
        } catch {
          // Server might not be ready yet
        }
        await new Promise((r) => setTimeout(r, 500))
        if (cancelled) return
      }
    }

    async function init() {
      try {
        await waitForReady()
        if (cancelled) return

        const [dsRes, kindRes, status] = await Promise.all([
          fetchDatasets(),
          fetchKinds(),
          fetchStatus(),
        ])
        if (cancelled) return

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

        showToast(`Loaded ${dsRes.datasets.length} datasets`, 'success')
        if (status.warming) {
          showToast('Warming dataloaders in background...', 'info')
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load data')
        showToast('Failed to connect to server', 'error')
      } finally {
        setInitializing(false)
      }
    }
    init()

    return () => {
      cancelled = true
    }
  }, [])

  // Fetch samples
  const fetchSamples = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let data: SampleData
      if (mode === 'dataset' && selectedDataset !== null) {
        try {
          await warmDataset(selectedDataset)
        } catch {
          // Ignore warm errors and still fetch samples
        }
        data = await fetchSamplesDataset(selectedDataset, N_SAMPLES)
      } else if (mode === 'kind' && selectedKind !== null) {
        try {
          await warmKind(selectedKind)
        } catch {
          // Ignore warm errors and still fetch samples
        }
        data = await fetchSamplesKind(selectedKind, N_SAMPLES)
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
    if (!initializing && ((mode === 'dataset' && selectedDataset !== null) || (mode === 'kind' && selectedKind !== null))) {
      fetchSamples()
    }
  }, [mode, selectedDataset, selectedKind, fetchSamples, initializing])

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
      const result = await reloadConfig()

      // Show toast with result
      if (result.status === 'unchanged') {
        showToast('Config unchanged', 'info')
      } else if (result.status === 'error') {
        showToast(result.message, 'error')
      } else {
        showToast(result.message, 'success')
      }

      // Refresh dataset list
      const [dsRes, kindRes] = await Promise.all([fetchDatasets(), fetchKinds()])
      setDatasets(dsRes.datasets)
      setDatasetsByKind(dsRes.by_kind)
      setKinds(kindRes.kinds)
      setMockMode(dsRes.mock_mode)

      // Reset selection if needed
      if (selectedDataset !== null && !dsRes.datasets.some((d) => d.dataset_idx === selectedDataset)) {
        setSelectedDataset(dsRes.datasets[0]?.dataset_idx ?? null)
      }
      if (selectedKind !== null && !kindRes.kinds.some((k) => k.name === selectedKind)) {
        setSelectedKind(kindRes.kinds[0]?.name ?? null)
      }

      // Warm current selection after reload
      if (mode === 'dataset' && selectedDataset !== null) {
        warmDataset(selectedDataset).catch(() => undefined)
      } else if (mode === 'kind' && selectedKind !== null) {
        warmKind(selectedKind).catch(() => undefined)
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to reload config'
      setError(msg)
      showToast(msg, 'error')
    } finally {
      setLoading(false)
    }
  }, [selectedDataset, selectedKind])

  // Show startup loading screen
  if (initializing) {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="text-2xl text-gray-300 mb-4">Score-qMRI Data Viewer</div>
          <div className="text-gray-500 animate-pulse">Loading datasets...</div>
        </div>
      </div>
    )
  }

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
      <Toast />
    </>
  )
}

export default App
