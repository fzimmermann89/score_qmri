import { useState, useEffect, useCallback, useMemo } from 'react'
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
import type { Dataset, KindInfo, SampleData, ViewerMode, ComplexRenderMode } from './types'

// Number of samples to fetch (3 rows x 7 cols)
const N_SAMPLES = 21

function App() {
  // State
  const [mode, setMode] = useState<ViewerMode>('dataset')
  const [complexRenderMode, setComplexRenderMode] = useState<ComplexRenderMode>('color')
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

  // Stats
  const stats = useMemo(() => {
    if (!sampleData) return null

    const { images, shape, kinds, background } = sampleData
    const [n, , h, w] = shape
    const pixelCount = n * h * w

    const defaultKind =
      mode === 'dataset' && selectedDataset !== null
        ? datasets.find((d) => d.dataset_idx === selectedDataset)?.kind
        : undefined

    const maxValues = complexRenderMode === 'color' ? pixelCount * 2 : pixelCount
    const values = new Float32Array(maxValues)
    let valueCount = 0
    const imageSize = 2 * h * w
    const planeSize = h * w

    const writeScalarValues = (isComplex: boolean, mode: ComplexRenderMode) => {
      for (let i = 0; i < n; i++) {
        const kindName = kinds?.[i] ?? defaultKind
        const sampleIsComplex = kindName
          ? kindName.includes('COMPLEX') || kindName.includes('complex')
          : isComplex
        const effectiveMode = sampleIsComplex ? mode : 'magnitude'

        const start = i * imageSize
        const realStart = start
        const imagStart = start + planeSize

        const maskStart = i * planeSize
        for (let j = 0; j < planeSize; j++) {
          if (background && background[maskStart + j]) {
            continue
          }
          const r = images[realStart + j]
          const im = images[imagStart + j]

          let val = 0
          switch (effectiveMode) {
            case 'real':
              val = r
              break
            case 'imag':
              val = im
              break
            case 'magnitude':
              val = Math.sqrt(r * r + im * im)
              break
            case 'phase':
              val = (Math.atan2(im, r) + Math.PI) / (2 * Math.PI)
              break
            default:
              val = Math.sqrt(r * r + im * im)
              break
          }
          values[valueCount++] = val
        }
      }
    }

    if (complexRenderMode === 'color') {
      const anyComplex = (kinds?.[0] ?? defaultKind ?? '').toLowerCase().includes('complex')
      if (!anyComplex && defaultKind) {
        writeScalarValues(false, 'magnitude')
      } else {
        for (let i = 0; i < n; i++) {
          const maskStart = i * planeSize
          const start = i * imageSize
          const realStart = start
          const imagStart = start + planeSize
          for (let j = 0; j < planeSize; j++) {
            if (background && background[maskStart + j]) {
              continue
            }
            values[valueCount++] = images[realStart + j]
            values[valueCount++] = images[imagStart + j]
          }
        }
      }
    } else {
      writeScalarValues(true, complexRenderMode)
    }

    if (valueCount === 0) return null

    // Compute stats
    let sum = 0
    let sumSq = 0
    const len = valueCount

    // Copy for sorting (to get percentile)
    const sorted = values.slice(0, valueCount)
    sorted.sort()

    for (let i = 0; i < len; i++) {
      const v = sorted[i]
      sum += v
      sumSq += v * v
    }

    const mean = sum / len
    const variance = (sumSq / len) - (mean * mean)
    const std = Math.sqrt(Math.max(0, variance))

    // 99.5th percentile
    const p995Idx = Math.floor(len * 0.995)
    const p995 = sorted[p995Idx] ?? 0

    return { std, p995 }
  }, [sampleData, complexRenderMode, mode, selectedDataset, datasets])

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
        complexRenderMode={complexRenderMode}
        onComplexRenderModeChange={setComplexRenderMode}
        stats={stats}
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
        complexRenderMode={complexRenderMode}
        defaultKind={mode === 'dataset' && selectedDataset !== null ? datasets.find(d => d.dataset_idx === selectedDataset)?.kind : undefined}
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
