# Normalization Rework Plan (Fixed Scales, No EMA)

## Goal
Replace the current **EMA-based per-kind running normalization** with a deterministic, configuration-driven normalization scheme:

- **Per-dataset fixed scaling** for complex + magnitude-like data.
- **Per-kind fixed scaling** for quantitative maps (T1MAP/T2MAP/T2STARMAP), shared across all datasets of the same kind.
- Preserve conventions:
  - Complex-valued normalized range approximately **[-1, 1]** (via magnitude-based scaling + magnitude clipping while preserving phase).
  - Real-valued normalized range approximately **[0, 1]**.
  - Background/air stays at **0**.
- Remove `RunningNormalizer` and all code paths relying on EMA stats.

This plan is focused on training and unconditional sampling consistency, while keeping a clear path for future inverse problems (e.g. DPS with quantitative signal models).

## Why change the current approach
### Current behavior (what happens today)
Normalization is currently split across two layers:

1) **Dataset-level coarse scaling** (optional):
- Each dataset config can specify a `scaling_factor`.
- Applied in `BaseMRIDataset.__getitem__` by multiplying raw image values.
- File: `src/score_qmri/datasets/base.py`

2) **Model-level EMA running normalization** (always on):
- `DiffusionModel` constructs a `RunningNormalizer` which tracks per-`DatasetKind` mean/variance via EMA.
- For complex kinds, normalization uses `3*std` scaling; for real kinds uses `mean + 3*std`.
- Updated during training only; frozen in eval mode; synced periodically across DDP ranks.
- Files:
  - `src/score_qmri/models/normalization.py`
  - `src/score_qmri/models/diffusion.py`
  - `src/score_qmri/training/lightning_module.py`

3) **Sampling uses normalizer state**
- Unconditional samplers sample in normalized space and denormalize afterwards.
- DPS uses `normalizer.get_scale()` for initialization and to convert DDIM updates between normalized and denormalized space.
- Files:
  - `src/score_qmri/sampling/generate.py`
  - `src/score_qmri/sampling/dps_sampler.py`

### Why EMA per-kind running normalization is a bad fit here
- **Non-stationary normalization**: EMA stats drift during training; diffusion training assumes a stationary data distribution for x0 in normalized space.
- **Batch composition dependence**: stats depend on which datasets/kinds appear recently.
- **Coupling to training dynamics**: normalization becomes part of the learned system state, complicating reproducibility and inference.
- **Inverse problems become fragile**: DPS currently relies on the learned/collected scale; any mismatch between train-time stats and inference-time data breaks scale consistency.

## Target behavior (new normalization contract)
### Key decisions already agreed
- Complex clipping must be **magnitude-based** and must **preserve phase**.
- Mask for stat estimation uses `abs(x) > 1e-6`.
- No `shift`.
- No storing clip value per dataset in config; clip is used only to *derive* a scale.
- No backwards compatibility.
- Global post-scaled clip limits:
  - Training clamp: `train_clip = 1.2`
  - Sampling clamp: `sample_clip = 1.0`
- DPS and forward models assume **dataset scaling is already applied** to inputs/measurements.
- For samplers / validation plots:
  - Non-map kinds output in **normalized space**.
  - Mapping kinds output in **physical units** (seconds), i.e. undo kind scaling.

### Two independent scaling layers
1) **Per-dataset scaling** (applies to most kinds)
- Stored in `config.yaml` per dataset.
- Applied in dataset `__getitem__`.
- Considered part of the dataset definition and assumed to already be applied for downstream tasks (including DPS).

2) **Per-kind scaling** (applies mainly to mapping kinds)
- Stored in `config.yaml` once per map kind.
- Applied and undone in `DiffusionModel.normalize/denormalize`.
- Purpose: ensure *all* datasets of the same map kind share a canonical normalized distribution, while allowing conversion back to seconds.

## Correctness review: training, diffusion, sampling

### Calibration script caveat (feedback)
- The dataset calibration tool must respect complex H5 group layout. Complex files may contain multiple groups (e.g. `coilwise/` and `combined/`) and different dataset configs will target different groups via `mode`.
- Do not fall back to the first group in the H5 file when the root has no datasets; this can compute a scaling_factor from the wrong group.
- Preferred implementation: instantiate the actual dataset classes from `dataset_configs` (with augmentation disabled) and draw samples through `__getitem__`, so the calibration logic matches training exactly.
This section checks the new plan against diffusion math and implementation details.

### Training distribution and noise
- After dataset scaling + clamping, the model sees x0 in approximately:
  - Real kinds: [0, train_clip] (typically [0,1])
  - Complex kinds: magnitude in [0, train_clip], phase preserved
- Diffusion adds Gaussian noise in normalized space:
  - This remains correct: noise is always ~N(0,1) and training objective uses the same.
- Loss scaling:
  - ε-prediction loss remains well-conditioned when x0 is O(1).
  - Keeping values near [0,1] is standard and improves stability.

### Complex data handling
- Scaling must not treat real/imag independently.
- We will scale complex values by a single scalar derived from magnitude statistics.
- Clipping strategy:
  - Compute magnitude m = |z|.
  - Clip m to [0, clip].
  - Rescale z by factor (m_clipped / (m + eps)).
- This avoids phase distortions that symmetric per-channel clipping can introduce.

### Mapping kinds (T1/T2/T2*)
- Mapping datasets are in seconds.
- Kind scaling provides a stable normalized range for diffusion training.
- Denormalization returns seconds for downstream physical models.
- Critical: kind scale must be shared across all datasets of same kind.

### Sampling consistency
- Unconditional samplers (DDIM/DDPM) operate in normalized space and apply `clamp_x0`.
- With the new rules:
  - For non-map kinds, output stays normalized.
  - For map kinds, output is denormalized to seconds.
- Global `sample_clip=1.0` prevents unrealistic extremes at inference.

### DPS consistency (future)
- DPS currently only supports complex kinds.
- Under the new contract:
  - Dataset scaling is assumed already applied to both the image domain state and the measurements.
  - Therefore DPS must not attempt to infer scales from the model.
  - DPS runs entirely in normalized space for the image prior.
- When adding mapping DPS later, only kind denormalization should be applied for the forward operator if it expects seconds.

## Implementation plan (no code changes yet)

### Step 0 — Config design
- Update `config.yaml`:
  - Rename `scaling_factor` -> `scale` for all dataset configs.
  - Add top-level normalization section:
    - `normalization.train_clip: 1.2`
    - `normalization.sample_clip: 1.0`
    - `normalization.kind_scales: { T1MAP: <float>, T2MAP: <float>, T2STARMAP: <float> }`

Rationale:
- Per-dataset scaling is dataset-specific and belongs with dataset config.
- Per-kind scaling is model semantics and belongs to a top-level normalization block.

### Step 1 — Dataset scaling + clipping
- Modify `BaseMRIDataset.__getitem__`:
  - Apply per-dataset `scale` multiplication.
  - Apply training-time global clip (`train_clip`) in post-scaled units.
  - Complex clip: magnitude-based with phase preservation.
  - Real clip: clamp to [0, train_clip].

Rationale:
- Ensures the model always trains on a stable bounded distribution.
- Removes need for EMA/online stats.
- Makes data distribution deterministic and reproducible.

### Step 2 — Replace model normalizer
- Remove `RunningNormalizer`:
  - Delete or orphan `src/score_qmri/models/normalization.py`.
  - Remove import/export references.
  - Remove DDP stat syncing call in training.
- Update `DiffusionModel.normalize/denormalize`:
  - For mapping kinds: divide/multiply by fixed `kind_scale`.
  - For all other kinds: identity.
  - Mask argument becomes unused; can keep signature for now but ignore.

Rationale:
- Keeps map outputs convertible to seconds.
- Removes hidden state and train/eval behavior differences.

### Step 3 — Clamp behavior
- Make `clamp_x0` depend on context (train vs sample):
  - Use `train_clip` during training.
  - Use `sample_clip` during sampling.

Implementation approaches:
- Pass limit explicitly from callers (preferred for clarity).
- Or store both in model and have samplers call `clamp_x0(..., limit=...)`.

Rationale:
- Training needs slight headroom (1.2) to avoid over-clamping gradients.
- Sampling uses stricter bound (1.0) for cleaner outputs.

### Step 4 — Sampling outputs (normalized vs physical)
Update:
- `src/score_qmri/sampling/generate.py`
- Lightning validation sampling log path in `src/score_qmri/training/lightning_module.py`

Rule:
- If kind is mapping: denormalize to seconds before returning/logging.
- Else: return/log normalized values.

Rationale:
- Matches your requirement and future DPS signal-model usage.

### Step 5 — DPS updates
- Remove any usage of `model.normalizer.get_scale()`.
- Ensure DPS does not attempt any scaling other than the model’s kind scaling (which is identity for complex kinds anyway).

Rationale:
- Scale handling is externalized: measurements and operators are assumed already dataset-scaled.
- Keeps DPS consistent with training scale.

### Step 6 — Calibration scripts (semi-automatic)
Add a cell-style script (Python with `# %%` cells) that:
- Loads `config.yaml`.
- Groups dataset configs by kind.
- For a selected kind:
  - Loads 100 random samples per dataset.
  - Uses `abs(x) > 1e-6` mask.
  - Computes quantile clip q (default 0.99).
  - Proposes `scale = 1 / clip_value`.
  - Shows preview grid: 8 random samples per dataset (columns), datasets as rows.
  - For complex, preview magnitude after scaling + optional phase plot.
  - Emits YAML fragments for:
    - per-dataset `scale` (complex/magnitude)
    - per-kind `kind_scales` (mapping)

Rationale:
- Produces deterministic scale constants.
- Makes it easy to iterate on q and visually validate.

### Step 7 — Tests and cleanup
- Remove EMA-related tests:
  - `tests/test_training/test_normalization.py`
- Update diffusion normalize/denormalize tests:
  - Non-mapping kinds: identity normalization.
  - Mapping kinds: fixed scale roundtrip.

Rationale:
- Tests must reflect the new deterministic normalization contract.

## Risk analysis / pitfalls to avoid
- **Complex clip implementation must be numerically safe**: handle m≈0 with eps to avoid NaNs.
- **Clipping inside dataset is destructive**: acceptable and desired here (outlier suppression), but must be consistent across training runs.
- **Sampler outputs change semantics**: downstream code must treat non-map outputs as normalized values.
- **Mapping visualization**: seconds can have larger numeric range; plotting should set colormap limits explicitly.

## Acceptance criteria
- No references to `RunningNormalizer` remain.
- Training does not depend on online stats.
- Unconditional sampling:
  - T1W/T2W/FLAIR/M0 produce normalized outputs.
  - Maps produce outputs in seconds.
- Calibration script produces plausible scales and previews.
- `pytest` passes after updating/removing affected tests.
