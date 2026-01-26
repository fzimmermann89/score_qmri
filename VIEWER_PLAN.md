# Data Viewer Plan

## Goal
Create a fast data viewer that reuses the training dataset logic and config parsing so outputs match training behavior. Provide two modes (per-dataset, per-kind), 4x8 random samples, fast canvas rendering, and mask overlays, with a config reload button.

## Explicit Requirements
- Use the same dataset logic and config parsing as training. No custom parsing or reimplementation of dataset behavior.
- Use jsonargparse (as training does) to parse config.yaml and instantiate data module or dataset configs.
- Always apply augmentation as configured. No disable toggle.
- Pipeline order: dataset scaling + augmentation -> 256x256 center crop -> downsample by 2 before sending to client.
- Per-dataset mode: select dataset, show 4x8 random samples.
- Per-kind mode: select kind, sample datasets with same ratios as training (respect schedule weights; phase-in fully in, phase-out not yet out). Show dataset labels on tiles.
- Datasets sorted/grouped by kind in the UI.
- Support penumbra and background mask overlays.
- If all datasets are missing (H5 files not found), use a mock dataset.
- Use fast rendering: binary float32 transfer + Canvas2D (no matplotlib/base64 in hot path).
- Use DataLoader/multiprocessing and caching to keep sampling fast.
- Ignore brainweb for now.

## High-Level Architecture

Backend (FastAPI) provides:
- /api/datasets
- /api/kinds
- /api/samples/dataset/{idx}
- /api/samples/kind/{kind}
- /api/reload

Frontend (React) provides:
- Mode toggle (dataset/kind)
- Dataset/kind select dropdown (grouped by kind)
- Reload config button
- 4x8 grid view
- Mask overlay toggles (penumbra, background)
- Spacebar for refresh

## Reuse of Training Logic (No Reinvention)
Use these exact modules/classes from the current codebase:
- Config parsing: jsonargparse + LightningArgumentParser
- Dataset configs: DatasetConfig dataclasses from `score_qmri.datasets.lightning`
- Dataset instantiation: `cfg.create()`
- Augmentation: `AugmentationPipeline` via dataset __getitem__
- Mask computation: `BaseMRIDataset.__getitem__`
- Center crop/pad: `Collate` from `score_qmri.datasets.lightning`
- Sampling distribution: `ScheduledBatchSampler` logic for dataset weights

## Backend Plan

### 1) Config Parsing (jsonargparse, same as training)
Goal: Avoid manual parsing. Use jsonargparse to produce the same dataclasses that training uses.

Approach:
- Use `LightningArgumentParser` and add dataclass args for MRIDataModule.
- Parse config.yaml using jsonargparse so the `data` section resolves to the correct DatasetConfig types.
- Extract `datamodule.dataset_configs` from the instantiated object (no dataset creation yet).
- Filter out any datasets whose h5_path does not exist.
- Filter out brainweb configs.
- If no datasets remain, fall back to mock manager.

Pseudo-flow:
```
parser = LightningArgumentParser()
parser.add_lightning_class_args(MRIDataModule, "data")
cfg = parser.parse_path(config_path)
dm = parser.instantiate_classes(cfg).data
dataset_configs = [c for c in dm.dataset_configs if c.type != "brainweb" and Path(c.h5_path).exists()]
```

### 2) Dataset and DataLoader Caching
Keep sampling fast by caching:
- Dataset instances per dataset config (augment always on)
- DataLoader instances per dataset and per kind
- Iterator for each DataLoader to avoid re-creating workers for each request

Per-dataset DataLoader:
- batch_size = 32
- shuffle = True
- num_workers = config.data.num_workers (or sane default if missing)
- persistent_workers = True
- pin_memory = torch.cuda.is_available()
- prefetch_factor = 4
- collate_fn = Collate(256, 256)

Per-kind DataLoader:
- ConcatDataset over all datasets of that kind
- ScheduledBatchSampler for sampling (batch_size = 32)
- ScheduledBatchSampler uses schedule weights with phase-in/out simplified to constant weights

Iterator behavior:
- On each request, call next(iterator)
- If StopIteration, create a new iterator
- Seed each DataLoader with base_seed + request_count to keep randomness while avoiding full re-init

### 3) Sampling Pipeline (Exact Order)
For both modes:
1. Dataset __getitem__ applies scaling + augmentation (exact training pipeline)
2. Collate (256x256 center crop/pad, 2-channel conversion)
3. Downsample by factor 2 to 128x128
4. Send to client

Downsample details:
- Images: torch.nn.functional.interpolate(mode="area" or "bilinear")
- Masks: nearest-neighbor interpolation, then threshold to bool

### 4) Per-Kind Sampling Weights
Use training ratios exactly:
- Compute dataset weights = schedule.weight * dataset_length
- For phase_in: use full weight (assume fully in)
- For phase_out: use full weight (assume not yet out)
- Respect per-dataset schedule weights
- Since only one kind is active, normalization is just weights / sum

Implementation detail:
- Use ScheduledBatchSampler or a light wrapper that uses its weight computation logic
- Prefer ScheduledBatchSampler for parity; pass ConstantSchedule instances with weights derived from original schedules

### 5) Binary Response Format
Return a single binary payload with metadata in headers:
- X-Shape: [n, 2, H, W]
- X-Labels: list of dataset filenames (per-kind only)
- X-Has-Masks: true

Payload order:
1) Images: float32, shape (n, 2, H, W)
2) Penumbra masks: uint8, shape (n, H, W)
3) Background masks: uint8, shape (n, H, W)

## Frontend Plan

### UI Layout (Modern, Tight, No Wasted Space)
- Single top bar with mode toggle, dataset/kind selector, reload button, mask toggles
- 4x8 grid fills viewport
- Minimal padding, tight gaps

### Controls
- Spacebar: fetch new samples
- Penumbra toggle: show orange overlay
- Background toggle: show blue overlay
- Reload: POST /api/reload, re-fetch datasets and kinds

### Rendering
- Use Canvas2D per tile
- Convert float32 to grayscale in JS
- Apply mask overlays per pixel (penumbra/background)
- For per-kind mode, show dataset label overlay on each tile

### Data Flow
- Fetch datasets on load
- Set initial selection (first dataset or first kind)
- Fetch samples on selection change or Spacebar
- Always request augmented samples (no UI toggle)

## Mock Fallback
If all configured datasets are missing:
- Use mock dataset generator (Shepp-Logan based)
- Provide same output format (images + masks)
- Mark mock_mode in /api/datasets response for UI warning

## Implementation Steps

Phase 1: Backend
1) Create `src/score_qmri/viewer/` package
2) Implement config loading via jsonargparse + MRIDataModule
3) Implement SampleManager with dataset/dataloader caching
4) Implement mock manager
5) Implement FastAPI endpoints + reload
6) Add CLI entry point `score-qmri-viewer`

Phase 2: Frontend
7) Create Vite React app in `viewer/`
8) Implement API client for binary responses
9) Implement SampleTile with mask overlays
10) Implement SampleGrid 4x8 layout
11) Implement Header bar + controls
12) Build and copy to `src/score_qmri/viewer/static/`

Phase 3: Verification
13) Validate per-dataset sampling vs training shapes
14) Validate per-kind ratios (spot-check weights)
15) Verify config reload
16) Verify mock fallback when all H5 missing
17) Verify mask overlay correctness

## Open Questions
- None at this point.
