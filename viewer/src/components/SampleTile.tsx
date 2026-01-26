import { useEffect, useRef } from 'react'

interface SampleTileProps {
  /** Float32 image data for this tile (2*H*W values) */
  imageData: Float32Array
  /** Height in pixels */
  height: number
  /** Width in pixels */
  width: number
  /** Penumbra mask (H*W uint8) */
  penumbraMask?: Uint8Array
  /** Background mask (H*W uint8) */
  backgroundMask?: Uint8Array
  /** Show penumbra overlay */
  showPenumbra: boolean
  /** Show background overlay */
  showBackground: boolean
  /** Optional label to show */
  label?: string
}

export function SampleTile({
  imageData,
  height,
  width,
  penumbraMask,
  backgroundMask,
  showPenumbra,
  showBackground,
  label,
}: SampleTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Create ImageData
    const imgData = ctx.createImageData(width, height)

    // Compute magnitude from complex (2 channels) and find min/max
    const magnitude = new Float32Array(height * width)
    let minVal = Infinity
    let maxVal = -Infinity

    for (let i = 0; i < height * width; i++) {
      const real = imageData[i]
      const imag = imageData[height * width + i]
      const mag = Math.sqrt(real * real + imag * imag)
      magnitude[i] = mag
      if (mag < minVal) minVal = mag
      if (mag > maxVal) maxVal = mag
    }

    // Normalize and render
    const range = maxVal - minVal || 1

    for (let i = 0; i < height * width; i++) {
      const val = Math.floor(((magnitude[i] - minVal) / range) * 255)
      const idx = i * 4

      // Base grayscale
      let r = val
      let g = val
      let b = val

      // Apply overlays with transparency
      const overlayAlpha = 0.4

      if (showPenumbra && penumbraMask && penumbraMask[i]) {
        // Orange overlay for penumbra
        r = Math.min(255, Math.floor(r * (1 - overlayAlpha) + 255 * overlayAlpha))
        g = Math.floor(g * (1 - overlayAlpha) + 140 * overlayAlpha)
        b = Math.floor(b * (1 - overlayAlpha))
      }

      if (showBackground && backgroundMask && backgroundMask[i]) {
        // Blue overlay for background
        r = Math.floor(r * (1 - overlayAlpha))
        g = Math.floor(g * (1 - overlayAlpha) + 100 * overlayAlpha)
        b = Math.min(255, Math.floor(b * (1 - overlayAlpha) + 255 * overlayAlpha))
      }

      imgData.data[idx] = r
      imgData.data[idx + 1] = g
      imgData.data[idx + 2] = b
      imgData.data[idx + 3] = 255
    }

    ctx.putImageData(imgData, 0, 0)
  }, [imageData, height, width, penumbraMask, backgroundMask, showPenumbra, showBackground])

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full object-contain bg-black"
      />
      {label && (
        <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-white text-[10px] px-1 py-0.5 truncate">
          {label}
        </div>
      )}
    </div>
  )
}
