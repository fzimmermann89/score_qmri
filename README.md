# Score-Based Posterior Sampling for Quantitative MRI

A diffusion model framework for learning priors over brain MRI images, designed for score-based posterior sampling in quantitative MRI reconstruction.

## Overview

This project addresses the inverse problem in quantitative MRI:

```
k-data = F(signal(θ)) + Noise
```
with signal an non-linear signal mdoel and F an undersampled multicoil Fouriertransformation
where we aim to recover tissue parameters θ (T1, T2, etc.) from undersampled k-space measurements using a learned image prior /
a learned score ∇p(θ).

**Current capabilities:**
- Training denoising score models on diverse brain MRI data
- Validation sampling with DDIM

**Planned:**
- Score-based posterior sampling for qMRI reconstruction

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   H5 Files ──► Dataset Classes ──► Augmentation ──► BatchSampler │
│                (Complex/Mag/Map)    (kornia)        (scheduled)  │
│                                                                  │
│                              ▼                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    Diffusion Model                        │  │
│   │                                                           │  │
│   │   x_t + t + kind ──► ConditioningMLP ──► UNet (FiLM) ──┬─► ε (noise)  │
│   │                                                        └─► x₀ (clean) │
│   └──────────────────────────────────────────────────────────┘  │
│                              ▼                                   │
│              MSE(ε) + λ·(SSIM + L1)(x₀) ──► Total Loss          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **VP-SDE with continuous time** | Flexible inference with any number of steps without retraining |
| **Dual prediction heads** | Noise head for score/sampling; x₀ head as auxiliary task improves representations |
| **FiLM conditioning** | Efficient modulation of features by timestep and dataset type |
| **14 dataset kinds** | Single model handles complex/magnitude images and quantitative maps |
| **2-channel input** | Unified format: (real, imag) for complex; (mag, zeros) for magnitude |

## Data Types

The model is conditioned on the type of MRI data via `DatasetKind`:

| Kind | Description |
|------|-------------|
| `COMPLEX_T1W_COILWISE` | T1-weighted complex, per-coil |
| `COMPLEX_T1W_COIL_COMBINED` | T1-weighted complex, RSS combined |
| `COMPLEX_T2W_*` | T2-weighted complex variants |
| `COMPLEX_FLAIR_*` | FLAIR complex variants |
| `T1MAP`, `T2MAP`, `T2STARMAP` | Quantitative parameter maps |
| `T1W_MAGNITUDE`, `T2W_MAGNITUDE`, `FLAIR_MAGNITUDE` | Magnitude images |
| `M0_COMPLEX`, `M0_MAGNITUDE` | Proton density from fitting |

## Data Format

### HDF5 Structure

**Format A: Complex data** (e.g., fastMRI with coilwise/combined groups):
```
file.h5
├── coilwise/
│   ├── subject_001: (N_z, N_coils, H, W) complex64
│   └── subject_002: (N_z, N_coils, H, W) complex64
└── combined/
    ├── subject_001: (N_z, H, W) complex64
    └── subject_002: (N_z, H, W) complex64
```

**Format B: Magnitude/Mapping data** (flat subject datasets at root):
```
file.h5
├── subject_001: (N_z, H, W) float32
├── subject_002: (N_z, H, W) float32
└── subject_003: (N_z, H, W) float32
```

This format is used for:
- Magnitude images (T1w, T2w, FLAIR)
- Quantitative parameter maps (T1map, T2map, T2*map)

### Input Tensor Format

Complex data is stored as `torch.complex64` tensors `(H, W)` during loading.
Magnitude data is stored as `torch.float32` tensors `(H, W)`.

For network input, the `DiffusionModel.prepare_input()` converts to 2-channel tensors `(B, 2, H, W)`:
- **Complex**: Channel 0 = real, Channel 1 = imaginary
- **Magnitude**: Channel 0 = magnitude, Channel 1 = zeros

### Coil Selection

For coilwise training, the `ComplexImageDataset` supports:
- `random_coil=True` (default): Randomly samples a coil per slice access
- `random_coil=False`: Iterates through all coil/slice combinations sequentially

## Installation

```bash
# Clone repository
git clone <repo-url>
cd score_qmri

# Install with uv
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Usage

### Training

```bash
# Activate environment
source .venv/bin/activate

# Train with config file
python train.py fit --config configs/train.yaml

# Or with CLI arguments
python train.py fit \
    --model.base_channels 64 \
    --model.num_blocks 4 \
    --data.batch_size 16 \
    --trainer.max_steps 100000
```

### Configuration

Training uses PyTorch Lightning CLI. Key hyperparameters:

```yaml
model:
  base_channels: 64      # UNet base width
  num_blocks: 4          # Encoder/decoder depth
  cond_dim: 128          # Conditioning vector size
  noise_weight: 1.0      # Weight for noise prediction loss
  x0_weight: 0.1         # Weight for clean image prediction loss

data:
  batch_size: 8
  num_workers: 4
  dataset_configs:
    - type: complex
      h5_path: /path/to/fastmri_t1.h5
      mode: combined
      kind: 1  # COMPLEX_T1W_COIL_COMBINED
    - type: mapping
      h5_path: /path/to/t1map.h5
      scaling_factor: 0.0005  # Normalize to ~[0,1]
      kind: 6  # T1MAP

trainer:
  max_steps: 100000
  precision: 16-mixed
```

## Development

```bash
# Run tests
pytest

# Linting
ruff check .
ruff format .
mypy .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
score_qmri/
├── models/
│   ├── diffusion.py      # VP-SDE diffusion model
│   ├── unet.py           # Dual-head UNet
│   └── conditioning.py   # Embeddings
├── datasets/
│   ├── complex.py        # Complex MRI datasets
│   ├── magnitude.py      # Magnitude datasets
│   ├── mapping.py        # qMRI parameter maps
│   └── kind.py           # DatasetKind enum
├── training/
│   ├── lightning_module.py
│   └── losses.py
├── sampling/
│   ├── ddpm_sampler.py
│   └── ddim_sampler.py
└── tests/
```

## References

- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", ICLR 2021
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
