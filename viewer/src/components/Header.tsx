import type { Dataset, KindInfo, ViewerMode } from '../types'

interface HeaderProps {
  mode: ViewerMode
  onModeChange: (mode: ViewerMode) => void
  datasets: Dataset[]
  datasetsByKind: Record<string, Dataset[]>
  kinds: KindInfo[]
  selectedDataset: number | null
  onDatasetChange: (idx: number) => void
  selectedKind: string | null
  onKindChange: (kind: string) => void
  showPenumbra: boolean
  onPenumbraChange: (show: boolean) => void
  showBackground: boolean
  onBackgroundChange: (show: boolean) => void
  onReload: () => void
  mockMode: boolean
}

export function Header({
  mode,
  onModeChange,
  datasetsByKind,
  kinds,
  selectedDataset,
  onDatasetChange,
  selectedKind,
  onKindChange,
  showPenumbra,
  onPenumbraChange,
  showBackground,
  onBackgroundChange,
  onReload,
  mockMode,
}: HeaderProps) {
  return (
    <header className="flex items-center gap-4 px-4 py-2 bg-gray-800 border-b border-gray-700">
      {/* Mock mode warning */}
      {mockMode && (
        <div className="bg-yellow-600 text-black px-2 py-0.5 rounded text-xs font-medium">
          MOCK MODE
        </div>
      )}

      {/* Mode toggle */}
      <div className="flex rounded overflow-hidden border border-gray-600">
        <button
          onClick={() => onModeChange('dataset')}
          className={`px-3 py-1 text-sm ${
            mode === 'dataset'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Dataset
        </button>
        <button
          onClick={() => onModeChange('kind')}
          className={`px-3 py-1 text-sm ${
            mode === 'kind'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Kind
        </button>
      </div>

      {/* Selector */}
      <div className="flex-1 max-w-md">
        {mode === 'dataset' ? (
          <select
            value={selectedDataset ?? ''}
            onChange={(e) => onDatasetChange(Number(e.target.value))}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm"
          >
            {Object.entries(datasetsByKind).map(([kind, datasets]) => (
              <optgroup key={kind} label={kind}>
                {datasets.map((ds) => (
                  <option key={ds.dataset_idx} value={ds.dataset_idx}>
                    {ds.h5_filename} ({ds.total_samples})
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        ) : (
          <select
            value={selectedKind ?? ''}
            onChange={(e) => onKindChange(e.target.value)}
            className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm"
          >
            {kinds.map((k) => (
              <option key={k.name} value={k.name}>
                {k.name} ({k.count} datasets)
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Mask toggles */}
      <label className="flex items-center gap-1 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={showPenumbra}
          onChange={(e) => onPenumbraChange(e.target.checked)}
          className="accent-orange-500"
        />
        <span className="text-orange-400">Penumbra</span>
      </label>
      <label className="flex items-center gap-1 text-sm cursor-pointer">
        <input
          type="checkbox"
          checked={showBackground}
          onChange={(e) => onBackgroundChange(e.target.checked)}
          className="accent-blue-500"
        />
        <span className="text-blue-400">Background</span>
      </label>

      {/* Reload button */}
      <button
        onClick={onReload}
        className="px-3 py-1 bg-gray-700 hover:bg-gray-600 border border-gray-600 rounded text-sm"
        title="Reload config.yaml"
      >
        Reload
      </button>

      {/* Hint */}
      <div className="text-gray-500 text-xs">Press Space for new samples</div>
    </header>
  )
}
