import { SampleTile } from './SampleTile'
import type { SampleData } from '../types'

interface SampleGridProps {
  /** Sample data from API */
  sampleData: SampleData | null
  /** Show penumbra mask overlay */
  showPenumbra: boolean
  /** Show background mask overlay */
  showBackground: boolean
  /** Whether we're in kind mode (show labels) */
  showLabels: boolean
  /** Loading state */
  loading: boolean
  /** Error message if any */
  error: string | null
}

const ROWS = 3
const COLS = 7

export function SampleGrid({
  sampleData,
  showPenumbra,
  showBackground,
  showLabels,
  loading,
  error,
}: SampleGridProps) {
  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-red-400 text-center">
          <div className="text-xl mb-2">Error</div>
          <div className="text-sm opacity-75">{error}</div>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-gray-400 animate-pulse">Loading samples...</div>
      </div>
    )
  }

  if (!sampleData) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-gray-500">Press Space to load samples</div>
      </div>
    )
  }

  const [n, , h, w] = sampleData.shape
  const numTiles = Math.min(n, ROWS * COLS)

  // Extract individual tile data
  const tiles = []
  for (let i = 0; i < numTiles; i++) {
    // Image: each tile has 2*h*w values (2 channels)
    const imgStart = i * 2 * h * w
    const imgData = sampleData.images.subarray(imgStart, imgStart + 2 * h * w)

    // Masks: each tile has h*w values
    const maskStart = i * h * w
    const penumbra = sampleData.penumbra.subarray(maskStart, maskStart + h * w)
    const background = sampleData.background.subarray(maskStart, maskStart + h * w)

    tiles.push({
      imageData: imgData,
      penumbra,
      background,
      label: showLabels && sampleData.labels[i] ? sampleData.labels[i] : undefined,
    })
  }

  return (
    <div
      className="flex-1 grid gap-1 p-2"
      style={{
        gridTemplateColumns: `repeat(${COLS}, 1fr)`,
        gridTemplateRows: `repeat(${ROWS}, 1fr)`,
      }}
    >
      {tiles.map((tile, i) => (
        <SampleTile
          key={i}
          imageData={tile.imageData}
          height={h}
          width={w}
          penumbraMask={tile.penumbra}
          backgroundMask={tile.background}
          showPenumbra={showPenumbra}
          showBackground={showBackground}
          label={tile.label}
        />
      ))}
    </div>
  )
}
