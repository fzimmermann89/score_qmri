import type { DatasetsResponse, KindsResponse, SampleData } from './types'

const API_BASE = '/api'

export async function fetchDatasets(): Promise<DatasetsResponse> {
  const res = await fetch(`${API_BASE}/datasets`)
  if (!res.ok) throw new Error(`Failed to fetch datasets: ${res.statusText}`)
  return res.json()
}

export async function fetchKinds(): Promise<KindsResponse> {
  const res = await fetch(`${API_BASE}/kinds`)
  if (!res.ok) throw new Error(`Failed to fetch kinds: ${res.statusText}`)
  return res.json()
}

export async function reloadConfig(): Promise<{
  status: string
  mock_mode: boolean
  num_datasets: number
  changed_datasets: number[]
  message: string
}> {
  const res = await fetch(`${API_BASE}/reload`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to reload config: ${res.statusText}`)
  return res.json()
}

export async function fetchStatus(): Promise<{
  ready: boolean
  warming: boolean
  mock_mode: boolean
  num_datasets: number
  config_path: string
}> {
  const res = await fetch(`${API_BASE}/status`)
  if (!res.ok) throw new Error(`Failed to fetch status: ${res.statusText}`)
  return res.json()
}

export async function warmDataset(datasetIdx: number): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/warm/dataset/${datasetIdx}`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to warm dataset: ${res.statusText}`)
  return res.json()
}

export async function warmKind(kind: string): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/warm/kind/${kind}`, { method: 'POST' })
  if (!res.ok) throw new Error(`Failed to warm kind: ${res.statusText}`)
  return res.json()
}

/**
 * Fetch samples from a specific dataset.
 * Returns binary float32 images + uint8 masks.
 */
export async function fetchSamplesDataset(datasetIdx: number, n: number = 32): Promise<SampleData> {
  const res = await fetch(`${API_BASE}/samples/dataset/${datasetIdx}?n=${n}`)
  if (!res.ok) throw new Error(`Failed to fetch samples: ${res.statusText}`)
  return parseBinaryResponse(res)
}

/**
 * Fetch samples from datasets of a specific kind using training distribution.
 * Returns binary float32 images + uint8 masks with source labels.
 */
export async function fetchSamplesKind(kind: string, n: number = 32): Promise<SampleData> {
  const res = await fetch(`${API_BASE}/samples/kind/${kind}?n=${n}`)
  if (!res.ok) throw new Error(`Failed to fetch samples: ${res.statusText}`)
  return parseBinaryResponse(res)
}

async function parseBinaryResponse(res: Response): Promise<SampleData> {
  // Parse metadata from headers
  const shapeHeader = res.headers.get('X-Shape')
  const labelsHeader = res.headers.get('X-Labels')
  const kindsHeader = res.headers.get('X-Kinds')

  if (!shapeHeader) throw new Error('Missing X-Shape header')

  const shape = JSON.parse(shapeHeader) as [number, number, number, number]
  const labels = labelsHeader ? JSON.parse(labelsHeader) : []
  const kinds = kindsHeader ? JSON.parse(kindsHeader) : []

  const [n, c, h, w] = shape

  // Calculate expected sizes
  const imageBytes = n * c * h * w * 4 // float32
  const maskBytes = n * h * w // uint8 per mask

  const buffer = await res.arrayBuffer()

  // Extract images and masks from buffer
  const images = new Float32Array(buffer, 0, n * c * h * w)
  const penumbra = new Uint8Array(buffer, imageBytes, n * h * w)
  const background = new Uint8Array(buffer, imageBytes + maskBytes, n * h * w)

  return { images, penumbra, background, shape, labels, kinds }
}
