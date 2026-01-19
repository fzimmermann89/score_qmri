export interface DatasetInfo {
  dataset_idx: number;
  h5_filename: string;
  h5_path?: string;
  kind: string;
  total_samples: number;
  config: any;
  type?: string;
  mode?: string | null;
  is_mock?: boolean;
}

export interface ImageStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  variance: number;
  signal_fraction: number;
  p99: number;
  p99_9: number;
  components?: {
    real: ImageStats;
    imag: ImageStats;
    abs: ImageStats;
  };
}

export interface NavigationInfo {
  dataset_type: 'coilwise' | 'combined' | 'flat';
  linear_index: number;
  total_samples: number;
  subject_count: number;
  current_subject_index: number;
  current_subject_key: string;
  slice_count: number;
  current_slice_index: number;
  coil_count?: number | null;
  current_coil_index?: number | null;
  can_next_coil: boolean;
  can_prev_coil: boolean;
  can_next_slice: boolean;
  can_prev_slice: boolean;
  can_next_subject: boolean;
  can_prev_subject: boolean;
}

export interface SampleResponse {
  index: number;
  total_samples: number;
  image_left_base64: string;
  image_right_base64: string | null;
  image_left_label: string;
  image_right_label: string | null;
  colorbar_range: {
    left: [number, number];
    right: [number, number];
  };
  penumbra_overlay_base64?: string;
  background_overlay_base64?: string;
  stats: ImageStats;
  metadata: Record<string, any>;
  navigation: NavigationInfo;
  raw_data_left?: number[]; 
  raw_data_right?: number[];
  mock_mode?: boolean;
  colorbar_image_base64?: string;
}

export interface BatchSampleResponse {
  index: number;
  image_left_base64: string;
  image_right_base64: string | null;
  image_left_label: string;
  image_right_label: string | null;
  colorbar_range: {
    left: [number, number];
    right: [number, number];
  };
  shape?: [number, number];
}

export interface HistogramResponse {
  bins: number[];
  counts: number[];
  stats: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
}

export interface BatchDataResponse {
  samples: BatchSampleResponse[];
  global_stats: ImageStats;
  global_histogram: HistogramResponse;
  global_range: [number, number];
}

export interface AnnotationField {
  id: string;
  label: string;
  field_type: 'bool' | 'number' | 'text';
}

export interface AnnotationValues {
  [fieldId: string]: any;
}

export interface AnnotationsResponse {
  fields: AnnotationField[];
  values: AnnotationValues;
}

export type DisplayMode = 'real_imag' | 'mag_phase';

export interface SampleParams {
  index?: number;
  augment: boolean;
  seed: number;
  display_mode: DisplayMode;
  vmin?: number;
  vmax?: number;
}

export interface HierarchyParams extends Omit<SampleParams, 'index'> {
  subject_idx: number;
  slice_idx: number;
  coil_idx?: number;
}
