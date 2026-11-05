# Dataset Debug Viewer - Implementation Plan

## Overview

Web-based MRI dataset viewer with React/Vite/Tailwind frontend and FastAPI backend
for inspecting and annotating datasets defined in training config.

## Directory Structure

```
dataset_debug/
├── README.md                    # Usage documentation
├── run.py                       # Entry: python dataset_debug/run.py --config config.yaml
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, serves static + API
│   ├── config_loader.py         # Parse YAML, create dataset configs
│   ├── dataset_manager.py       # Dataset creation and sample loading
│   ├── annotations.py           # Global fields, per-dataset values
│   └── image_utils.py           # Tensor → base64 PNG rendering
└── frontend/
    ├── package.json
    ├── vite.config.ts           # Output to ../backend/static
    ├── tsconfig.json
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── api.ts
        ├── types.ts
        ├── index.css            # Tailwind imports
        ├── components/
        │   ├── DatasetSelector.tsx
        │   ├── ImageViewer.tsx      # Side-by-side complex display
        │   ├── Histogram.tsx        # Toggleable histogram
        │   ├── StatsPanel.tsx
        │   ├── ConfigPanel.tsx
        │   ├── NavigationControls.tsx
        │   ├── AugmentationToggle.tsx
        │   ├── AnnotationPanel.tsx
        │   └── KeyboardHelp.tsx
        └── hooks/
            └── useKeyboardNavigation.ts
```

## Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Returns config file path and summary |
| `/api/datasets` | GET | List all datasets with configs + sample counts |
| `/api/datasets/{idx}/sample` | GET | `?index=N&augment=bool&seed=int&display_mode=real_imag\|mag_phase` |
| `/api/datasets/{idx}/sample-by-hierarchy` | GET | `?subject=int&slice=int&coil=int&augment=bool&seed=int` → sample by position |
| `/api/datasets/{idx}/random-index` | GET | Random valid index |
| `/api/datasets/{idx}/histogram` | GET | `?index=N&augment=bool&seed=int` → histogram data |
| `/api/annotation-fields` | GET | Get all defined fields |
| `/api/annotation-fields` | POST | Add field: `{id, label, field_type}` |
| `/api/annotation-fields/{id}` | DELETE | Remove field |
| `/api/annotations/{dataset_idx}` | GET | Get annotations for this dataset |
| `/api/annotations/{dataset_idx}` | PUT | Save annotations for this dataset |
| `/api/annotations/export` | GET | Export all annotations as JSON download |

### Sample Response

```python
{
    "index": 42,
    "total_samples": 15234,

    # Two images for side-by-side display
    "image_left_base64": "data:image/png;base64,...",
    "image_right_base64": "data:image/png;base64,...",
    "image_left_label": "Real",
    "image_right_label": "Imaginary",

    # Colorbar range
    "colorbar_range": {"left": [-0.5, 1.2], "right": [-0.5, 1.2]},

    # Mask overlays (RGBA PNGs for transparency)
    "penumbra_overlay_base64": "data:image/png;base64,...",
    "background_overlay_base64": "data:image/png;base64,...",

    # Stats (computed on magnitude for complex)
    "stats": {
        "min": 0.0, "max": 1.23, "mean": 0.45,
        "std": 0.12, "variance": 0.0144,
        "signal_fraction": 0.72
    },

    # Metadata
    "metadata": {
        "dataset_idx": 3,
        "h5_path": "./datasets/fastmri_t1.h5",
        "h5_filename": "fastmri_t1.h5",
        "dataset_type": "complex",
        "kind": "COMPLEX_T1W_COILWISE",
        "mode": "coilwise",
        "subject_key": "sub_001",
        "slice_index": 23,
        "coil_index": 5,
        "is_complex": true,
        "shape": [320, 320]
    },

    # Navigation info (for hierarchical navigation)
    "navigation": {
        "dataset_type": "coilwise",  # "coilwise", "combined", or "flat"
        "linear_index": 42,
        "total_samples": 15234,
        "subject_count": 45,
        "current_subject_index": 12,  # Index in subjects list
        "current_subject_key": "sub_001",
        "slice_count": 120,  # For current subject
        "current_slice_index": 23,
        "coil_count": 16,  # Only for coilwise mode
        "current_coil_index": 5,
        "can_next_coil": true,
        "can_prev_coil": true,
        "can_next_slice": true,
        "can_prev_slice": true,
        "can_next_subject": true,
        "can_prev_subject": true
    }
}
```

### Histogram Response

```python
{
    "bins": [0.0, 0.1, 0.2, ...],
    "counts": [120, 456, 890, ...],
    "stats": {"min": 0.0, "max": 1.23, "mean": 0.45, "std": 0.12}
}
```

## Annotation Storage Format

File: `dataset_debug/annotations.json`

```json
{
  "fields": [
    {"id": "is_skullstripped", "label": "Is Skullstripped", "field_type": "bool"},
    {"id": "quality_score", "label": "Quality Score (1-5)", "field_type": "number"},
    {"id": "notes", "label": "Notes", "field_type": "text"}
  ],
  "datasets": {
    "3": {"is_skullstripped": true, "quality_score": 4},
    "7": {"is_skullstripped": false, "notes": "has motion artifacts"},
    "12": {"is_skullstripped": true, "quality_score": 5, "notes": "excellent quality"}
  },
  "metadata": {
    "config_path": "config.yaml",
    "created_at": "2026-01-15T10:30:00Z",
    "last_modified": "2026-01-15T12:45:00Z"
  }
}
```

Key design:
- **Fields are global**: Same fields apply to all datasets
- **Values are per-dataset**: Each dataset (by config index) has its own annotation values
- **Sparse storage**: Only annotated datasets are stored

## UI Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 🔬 Dataset Debug Viewer                              config: config.yaml │
├─────────────────────┬────────────────────────────────────────────────────┤
│                     │  Complex Display: [Real/Imag ▼] [Mag/Phase]        │
│ ▼ Dataset [3/52]    │  ┌─────────────────┬─────────────────┐             │
│ ┌─────────────────┐ │  │                 │                 │             │
│ │ fastmri_t1.h5   │ │  │      REAL       │      IMAG       │  [colorbar] │
│ │ coilwise        │ │  │   (or MAG)      │   (or PHASE)    │             │
│ │ COMPLEX_T1W_... │ │  │                 │                 │             │
│ │ 15234 samples   │ │  │  ▓▓ penumbra    │                 │             │
│ └─────────────────┘ │  │  ░░ background  │                 │             │
│                     │  └─────────────────┴─────────────────┘             │
│ ▼ Config            │                                                    │
│ ┌─────────────────┐ │  [◀ Prev Sample] [▶ Next Sample]  Index: [42] / 15234│
│ │ type: complex   │ │  [◀ Coil] [▶ Coil]  [◀ Slice] [▶ Slice]  [◀ Subject] [▶ Subject]  [🎲 Random]│
│ │ mode: coilwise  │ │  Augment: [✓ ON]  Seed: [1234] [🔄 New Aug]        │
│ │ scaling: 1.0    │ │  Masks: [✓ Penumbra] [✓ Background]               │
│ │ penumbra: 30    │ │  ───────────────────────────────────────────────── │
│ │ aug: scale 0.7… │ │  📊 Histogram [Toggle]                             │
│ └─────────────────┘ │  ┌─────────────────────────────────────────────┐   │
│                     │  │  ▁▂▃▅▇▅▃▂▁   min:0.0 max:1.2 mean:0.45     │   │
│ ▼ Stats             │  └─────────────────────────────────────────────┘   │
│ min: 0.00           │                                                    │
│ max: 1.23           │  ───────────────────────────────────────────────── │
│ mean: 0.45          │  Metadata                                          │
│ std: 0.12           │  Dataset: fastmri_t1.h5 | 15234 samples                │
│ signal: 72%         │  Subject: sub_001 | Slice: 23/120 | Coil: 5/16     │
│                     │                                                    │
├─────────────────────┼────────────────────────────────────────────────────┤
│ ▼ Annotation Fields │  📝 Dataset Annotations (Dataset 3: fastmri_t1.h5) │
│ [+ Add Field]       │  ┌────────────────────────────────────────────┐   │
│ ┌─────────────────┐ │  │ [✓] Is Skullstripped                       │   │
│ │☑ is_skullstrip  │ │  │ Quality Score: [4    ]                     │   │
│ │# quality (1-5)  │ │  │ Notes: [Good quality, some ringing_______] │   │
│ │T notes          │ │  └────────────────────────────────────────────┘   │
│ │[Export JSON]    │ │                                                    │
│ └─────────────────┘ │  ⌨️ Press K or click [?] for full keyboard shortcuts │
└─────────────────────┴────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

### Navigation

| Key | Action |
|-----|--------|
| `→` / `D` | Next sample (hierarchical: coil → slice → subject) |
| `←` / `A` | Previous sample (reverse hierarchy) |
| `C` → `→` | Next coil |
| `C` → `←` | Previous coil |
| `S` → `→` | Next slice |
| `S` → `←` | Previous slice |
| `U` → `→` | Next subject |
| `U` → `←` | Previous subject |
| `R` | Random sample in current dataset |
| `G` | Jump to index (prompts for number) |
| `1-9` | Jump to dataset 1-9 |
| `0` | Jump to dataset 10 |
| `Ctrl+1-9/0` | Jump to dataset 11-20 |

### Display & Annotations

| Key | Action |
|-----|--------|
| `Space` | New augmentation (increment seed, same index) |
| `T` | Toggle augmentation on/off |
| `M` | Toggle mask overlays (penumbra + background) |
| `P` | Toggle penumbra overlay only |
| `B` | Toggle background overlay only |
| `H` | Toggle histogram |
| `V` | Toggle display mode (Real/Imag ↔ Mag/Phase) |
| `F` | Toggle fullscreen image view |
| `Esc` | Exit fullscreen / close modals |

### Annotation Editing

| Key | Action |
|-----|--------|
| `N` | Add new annotation field |
| `E` | Export annotations to JSON |
| `Ctrl+S` | Save current annotation values |
| `Tab` | Move focus to next annotation input |
| `Shift+Tab` | Move focus to previous annotation input |

### Hierarchy Navigation Details

For **coilwise complex datasets**, left/right keys cycle through:
1. Coil index (within current slice)
2. Slice index (within current subject)
3. Subject index (move to next subject, coil=0, slice=0)

For **combined complex datasets**, left/right keys cycle through:
1. Slice index (within current subject)
2. Subject index

For **flat datasets (magnitude, mapping)**, left/right keys simply move through linear sample index.

## Backend Implementation

### Navigation Logic

The DatasetManager implements hierarchical navigation for complex/coilwise datasets:

**For coilwise mode (flat index → hierarchy):**
- Linear index maps to: `(subject_idx, slice_idx, coil_idx)`
- Next sample: increment coil → if end, increment slice, coil=0 → if end, increment subject, slice=0, coil=0

**For combined mode:**
- Linear index maps to: `(subject_idx, slice_idx)`
- Next sample: increment slice → if end, increment subject, slice=0

**For flat mode (magnitude, mapping):**
- Linear index maps to: `(subject_idx, slice_idx)`
- No coil dimension

Navigation functions (`navigate_next`, `navigate_prev`) take a `nav_type` parameter to control which dimension to increment/decrement.

### run.py
- CLI entry point with `--config` and `--port` arguments
- Starts uvicorn server
- Opens browser automatically

### backend/config_loader.py
- Parses YAML using same structure as LightningCLI
- Returns list of `DatasetConfig` objects (reuses score_qmri.datasets)
- Validates H5 file existence

### backend/dataset_manager.py
```python
class DatasetManager:
    def __init__(self, configs: list[DatasetConfig]):
        self.configs = configs
        self._datasets: dict[int, BaseMRIDataset] = {}  # Lazy loaded
        self._datasets_no_aug: dict[int, BaseMRIDataset] = {}

    def get_sample(self, idx: int, sample_idx: int,
                   augment: bool, seed: int | None,
                   display_mode: str) -> dict:
        """Load sample and render images."""

    def get_sample_by_hierarchy(self, idx: int, subject_idx: int,
                               slice_idx: int, coil_idx: int | None,
                               augment: bool, seed: int | None,
                               display_mode: str) -> dict:
        """Load sample by hierarchical position (for nav buttons)."""

    def navigate_next(self, idx: int, linear_idx: int, nav_type: str) -> dict:
        """Get next index based on nav_type: 'sample', 'coil', 'slice', 'subject'"""

    def navigate_prev(self, idx: int, linear_idx: int, nav_type: str) -> dict:
        """Get previous index based on nav_type: 'sample', 'coil', 'slice', 'subject'"""

    def get_histogram(self, idx: int, sample_idx: int,
                      augment: bool, seed: int | None) -> dict:
        """Compute histogram for sample."""
```

### backend/image_utils.py
```python
def render_images(
    image: torch.Tensor,  # (2, H, W) or (H, W)
    display_mode: str,    # "real_imag" or "mag_phase"
    is_complex: bool
) -> tuple[str, str, str, str]:
    """Returns (left_b64, right_b64, left_label, right_label)"""

def render_mask_overlay(
    mask: torch.Tensor,
    color: tuple[int, int, int, int]  # RGBA
) -> str:
    """Render mask as transparent PNG overlay."""
```

### backend/annotations.py
```python
@dataclass
class AnnotationField:
    id: str
    label: str
    field_type: Literal["bool", "number", "text"]

class AnnotationStore:
    def __init__(self, storage_path: Path):
        self.path = storage_path / "annotations.json"

    def add_field(self, field: AnnotationField) -> None: ...
    def remove_field(self, field_id: str) -> None: ...
    def get_fields(self) -> list[AnnotationField]: ...
    def get_dataset_annotations(self, dataset_idx: int) -> dict[str, Any]: ...
    def set_dataset_annotations(self, dataset_idx: int, values: dict[str, Any]) -> None: ...
    def export(self) -> dict: ...
```

## Frontend Components

### App.tsx
- Main state: `datasetIdx`, `sampleIdx`, `augment`, `seed`, `displayMode`
- Keyboard event handling
- Layout with Tailwind grid

### NavigationControls.tsx
- Index input + Next/Prev Sample buttons
- Hierarchical navigation: [◀ Coil] [▶ Coil] [◀ Slice] [▶ Slice] [◀ Subject] [▶ Subject]
- Random sample button
- "Jump to Index" prompt (G key)
- Subject/Slice/Coil breakdown display with counts
- Disables buttons appropriately based on navigation limits

### api.ts
```typescript
export const api = {
  getDatasets: () => fetch('/api/datasets').then(r => r.json()),
  getSample: (idx: number, params: SampleParams) => ...,
  getHistogram: (idx: number, params: SampleParams) => ...,
  getAnnotationFields: () => ...,
  addAnnotationField: (field: FieldDef) => ...,
  getDatasetAnnotations: (idx: number) => ...,
  setDatasetAnnotations: (idx: number, values: Record<string, any>) => ...,
  exportAnnotations: () => window.open('/api/annotations/export'),
};
```

### DatasetSelector.tsx
- Dropdown showing all datasets from config
- Shows: h5_filename, type, kind, **total samples in dataset**, mode (if complex)
- Displays total sample count prominently (distinct from H5 file samples)
- On change: load first sample of selected dataset

### StatsPanel.tsx
- Displays: min, max, mean, std, variance, signal_fraction
- Updates automatically on sample change
- Compact layout

### ConfigPanel.tsx (Collapsible)
- Shows full dataset configuration from YAML
- Displays: type, h5_path, kind, mode, scaling_factor, penumbra_width
- Augmentation parameters (if configured)
- Schedule settings (type, steps, weight)

### ImageViewer.tsx
- Side-by-side display with labels
- Overlay toggle for penumbra/background masks
- Colorbar display

### Histogram.tsx
- SVG bar chart
- Mean/std markers
- Collapsible

### AnnotationPanel.tsx
- Left side: Field definitions (add/remove)
- Right side: Current dataset values (auto-save on change)
- Export button

### KeyboardHelp.tsx
- Displays all keyboard shortcuts in organized sections
- Collapsible panel (toggle with K key or button)
- Shows: Navigation, Display, Annotation sections
- Highlights hierarchical navigation rules

## Development Setup

```bash
# Backend dependencies (add to pyproject.toml)
uv add --dev fastapi uvicorn[standard]

# Frontend setup
cd dataset_debug/frontend
npm create vite@latest . -- --template react-ts
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install

# Build frontend
npm run build  # Outputs to ../backend/static

# Run viewer
python dataset_debug/run.py --config config.yaml --port 8765
```

## File Summary

| File | Lines (est.) | Purpose |
|------|--------------|---------|
| `run.py` | 40 | CLI entry, uvicorn launch |
| `backend/main.py` | 150 | FastAPI routes |
| `backend/config_loader.py` | 80 | YAML parsing |
| `backend/dataset_manager.py` | 150 | Dataset loading |
| `backend/annotations.py` | 100 | Annotation storage |
| `backend/image_utils.py` | 120 | Image rendering |
| `frontend/src/App.tsx` | 200 | Main component |
| `frontend/src/api.ts` | 60 | API client |
| `frontend/src/types.ts` | 50 | TypeScript types |
| `frontend/src/components/*` | 500 | UI components |
| `frontend/src/hooks/*` | 50 | Custom hooks |
| Config files | 100 | vite, tailwind, etc. |
| **Total** | **~1600** | |
