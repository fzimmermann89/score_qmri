# (q)MRI Foundation Models

Framework for training diffusion models and self-supervised JEPA models for quantitative MRI reconstruction and representation learning.

## Highlights

- **VE-SDE (EDM/Karras) Diffusion**: Learning MRI priors (2-channel real/imag or mag/zeros) with dual x₀ heads.
- **JEPA Pretraining**: Self-supervised representation learning via Joint Embedding Predictive Architecture.
- **Unified MRI Format**: Supports complex (coilwise/combined), magnitude, and quantitative maps (T1, T2, T2*) in a unified 2-channel format.
- **Advanced Architecture**: Pre-Norm UNet with SDPA attention, FiLM conditioning on noise levels and dataset kinds.

## Quick Start

### Diffusion Training
```bash
source .venv/bin/activate
score-qmri fit --config config.yaml
```

### JEPA Pretraining
```bash
source .venv/bin/activate
jepa-qmri fit --config jepa_config.yaml
```

## Configuration

### Diffusion Config (`config.yaml`)
```yaml
# Values shown are defaults in code unless explicitly set in config.yaml.
model:
  attention_levels: [3, 4]
  head_dim: 64
  learning_rate: 1e-4

  # EDM parameters (defaults shown)
  sigma_min: 0.002
  sigma_max: 10.0
  sigma_data: 0.2
  p_mean: -1.2
  p_std: 1.2
  rho: 7.0

  normalization:
    train_clip: 1.2
    sample_clip: 1.0
    kind_scales:
      T1MAP: 0.0003
      T2MAP: 0.005
      T2STARMAP: 0.01

data:
  batch_size: 64
  sample_size: [256, 256]
  kind_normalization: 0.3

trainer:
  max_steps: 1000000
  gradient_clip_val: 1.0
```

### JEPA Config (`jepa_config.yaml`)
```yaml
model:
  img_size: 256
  patch_size: 16
  embed_dim: 384
  encoder_depth: 12
  encoder_num_heads: 6
  predictor_depth: 6
  predictor_embed_dim: 192
  num_targets: 4
  target_scale: 0.25
  target_aspect_ratio: [0.75, 1.5]
  context_scale: 0.85
  learning_rate: 1e-4
  weight_decay: 0.05
  warmup_steps: 5000

data:
  batch_size: 64
  sample_size: [256, 256]

trainer:
  max_steps: 100000
  precision: bf16-mixed
```

## Development
```bash
pytest
ruff check .
ruff format .
mypy .
pre-commit run --all-files
```

## References
- Karras et al., NeurIPS 2022 (EDM)
- Assran et al., CVPR 2023 (I-JEPA)
- Song et al., ICLR 2021 (Score-Based SDEs)
- Chung et al., ICLR 2023 (DPS)
- Perez et al., AAAI 2018 (FiLM)
