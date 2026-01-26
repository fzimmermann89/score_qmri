/** Dataset metadata from the API */
export interface Dataset {
  dataset_idx: number
  h5_filename: string
  kind: string
  total_samples: number
}

/** Kind summary from the API */
export interface KindInfo {
  name: string
  count: number
}

/** Parsed binary sample result */
export interface SampleData {
  /** Float32 images, shape [n, 2, H, W] */
  images: Float32Array
  /** Penumbra masks, shape [n, H, W] as uint8 */
  penumbra: Uint8Array
  /** Background masks, shape [n, H, W] as uint8 */
  background: Uint8Array
  /** Image dimensions */
  shape: [number, number, number, number]
  /** Source dataset labels (per-kind mode only) */
  labels: string[]
  /** Source dataset kinds (per-kind mode only) */
  kinds: string[]
}

/** API datasets response */
export interface DatasetsResponse {
  datasets: Dataset[]
  by_kind: Record<string, Dataset[]>
  mock_mode: boolean
}

/** API kinds response */
export interface KindsResponse {
  kinds: KindInfo[]
}

/** Viewer mode */
export type ViewerMode = 'dataset' | 'kind'
