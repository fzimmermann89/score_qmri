import { 
  DatasetInfo, SampleResponse, SampleParams, HistogramResponse, 
  AnnotationField, AnnotationValues, HierarchyParams, BatchDataResponse 
} from './types';

const API_ROOT = '/api';

export interface DatasetsResponse {
  datasets: DatasetInfo[];
  mock_mode: boolean;
}

export const api = {
  getDatasets: async (): Promise<DatasetsResponse> => {
    const res = await fetch(`${API_ROOT}/datasets`);
    if (!res.ok) throw new Error((await res.json()).detail || "Failed to fetch datasets");
    const data = await res.json();
    return { datasets: data.datasets, mock_mode: data.mock_mode || false };
  },

  getSample: async (datasetIdx: number, params: SampleParams): Promise<SampleResponse> => {
    const search = new URLSearchParams({
      augment: params.augment.toString(),
      seed: params.seed.toString(),
      display_mode: params.display_mode
    });
    if (params.index !== undefined) search.set('index', params.index.toString());
    if (params.vmin !== undefined) search.set('vmin', params.vmin.toString());
    if (params.vmax !== undefined) search.set('vmax', params.vmax.toString());

    const res = await fetch(`${API_ROOT}/datasets/${datasetIdx}/sample?${search.toString()}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Failed to fetch sample");
    return res.json();
  },

  getSampleByHierarchy: async (datasetIdx: number, params: HierarchyParams): Promise<SampleResponse> => {
    const search = new URLSearchParams({
      subject: params.subject_idx.toString(),
      slice: params.slice_idx.toString(),
      augment: params.augment.toString(),
      seed: params.seed.toString(),
      display_mode: params.display_mode
    });
    if (params.coil_idx !== undefined) search.set('coil', params.coil_idx.toString());
    if (params.vmin !== undefined) search.set('vmin', params.vmin.toString());
    if (params.vmax !== undefined) search.set('vmax', params.vmax.toString());

    const res = await fetch(`${API_ROOT}/datasets/${datasetIdx}/sample-by-hierarchy?${search.toString()}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Failed to fetch sample by hierarchy");
    return res.json();
  },

  getBatchData: async (datasetIdx: number, params: { indices: number[]; augment: boolean; seed: number; display_mode: string }): Promise<BatchDataResponse> => {
    const res = await fetch(`${API_ROOT}/datasets/${datasetIdx}/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Failed to fetch batch");
    return res.json();
  },

  getRandomIndex: async (datasetIdx: number): Promise<number> => {
    const res = await fetch(`${API_ROOT}/datasets/${datasetIdx}/random-index`);
    const data = await res.json();
    return data.index;
  },

  getHistogram: async (datasetIdx: number, params: SampleParams): Promise<HistogramResponse> => {
    const search = new URLSearchParams({
      augment: params.augment.toString(),
      seed: params.seed.toString()
    });
    if (params.index !== undefined) {
      search.set('index', params.index.toString());
    }
    const res = await fetch(`${API_ROOT}/datasets/${datasetIdx}/histogram?${search.toString()}`);
    if (!res.ok) throw new Error((await res.json()).detail || "Failed to fetch histogram");
    return res.json();
  },

  getAnnotationFields: async (): Promise<AnnotationField[]> => {
    const res = await fetch(`${API_ROOT}/annotation-fields`);
    const data = await res.json();
    return data.fields;
  },

  addAnnotationField: async (field: AnnotationField): Promise<void> => {
    await fetch(`${API_ROOT}/annotation-fields`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(field)
    });
  },

  removeAnnotationField: async (fieldId: string): Promise<void> => {
    await fetch(`${API_ROOT}/annotation-fields/${fieldId}`, {
      method: 'DELETE'
    });
  },

  getDatasetAnnotations: async (datasetIdx: number): Promise<AnnotationValues> => {
    const res = await fetch(`${API_ROOT}/annotations/${datasetIdx}`);
    const data = await res.json();
    return data.values || {};
  },

  setDatasetAnnotations: async (datasetIdx: number, values: AnnotationValues): Promise<void> => {
    await fetch(`${API_ROOT}/annotations/${datasetIdx}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values })
    });
  },
  
  exportAnnotationsUrl: `${API_ROOT}/annotations/export`
};
