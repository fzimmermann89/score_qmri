import { useEffect, useRef } from 'react'
import type { ComplexRenderMode } from '../types'

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
  /** Dataset kind (to determine rendering mode) */
  kind?: string
  /** Complex rendering mode */
  complexRenderMode: ComplexRenderMode
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
  kind,
  complexRenderMode,
}: SampleTileProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Create ImageData
    const imgData = ctx.createImageData(width, height)
    const isComplex = kind && (kind.includes('COMPLEX') || kind.includes('complex'))

    // Compute pixel values
    for (let i = 0; i < height * width; i++) {
      const idx = i * 4
      let r = 0
      let g = 0
      let b = 0

      const real = imageData[i]
      const imag = imageData[height * width + i]

      // For non-complex data, always use grayscale/magnitude rendering
      const mode = isComplex ? complexRenderMode : 'magnitude'

      if (mode === 'color') {
        // Complex: Real -> Red, Imag -> Green
        r = Math.floor(Math.min(1, Math.max(0, real)) * 255)
        g = Math.floor(Math.min(1, Math.max(0, imag)) * 255)
        b = 0
      } else {
        let val = 0
        switch (mode) {
          case 'real':
            val = real
            break
          case 'imag':
            val = imag
            break
          case 'phase':
            // Map [-PI, PI] to [0, 1]
            val = (Math.atan2(imag, real) + Math.PI) / (2 * Math.PI)
            break
          case 'magnitude':
          default:
            val = Math.sqrt(real * real + imag * imag)
            break
        }

        const gray = Math.floor(Math.min(1, Math.max(0, val)) * 255)
        r = gray
        g = gray
        b = gray
      }

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
  }, [imageData, height, width, penumbraMask, backgroundMask, showPenumbra, showBackground, kind, complexRenderMode])

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
