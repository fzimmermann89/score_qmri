# Score-Based Posterior Sampling for Quantitative MRI

Diffusion model framework for learning MRI priors (2-channel real/imag or mag/zeros) for qMRI reconstruction.

## Highlights

- VE-SDE (EDM/Karras) diffusion with dual x₀ heads (main + aux)
- AdaGN (scale/shift norm) conditioning on log(σ) + dataset kind
- Pre-Norm UNet with optional SDPA attention (`attention_levels`)
- Karras (poly-ρ) sigma schedule for sampling
- Fixed per-kind normalization for mapping kinds (`kind_scales`)
- EDM validation sampling with fixed noise

## Quick Start

```bash
source .venv/bin/activate
score-qmri-train fit --config config.yaml
```

## Key Config (config.yaml)

```yaml
# Values shown are defaults in code unless explicitly set in config.yaml.
model:
  learning_rate: 1e-4

  # UNet parameters (defaults shown)
  # base_channels: 128
  # channel_mult: [1, 1, 2, 2, 4, 4]
  # num_res_blocks: 2
  # attention_levels: [-1, -2, -3]
  # head_dim: 64

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
      T1MAP: 0.25
      T2MAP: 2
      T2STARMAP: 8

# Optional overrides (defaults in code)
# model.base_channels: 128
# model.num_res_blocks: 2


data:
  batch_size: 64
  sample_size: [256, 256]
  kind_normalization: 0.3

trainer:
  max_steps: 1000000
  gradient_clip_val: 1.0

```

## Dev

```bash
pytest
ruff check .
ruff format .
mypy .
pre-commit run --all-files
```

## References

- Karras et al., NeurIPS 2022 (EDM)
- Song et al., ICLR 2021 (Score-Based SDEs)
- Chung et al., ICLR 2023 (DPS)
- Perez et al., AAAI 2018 (FiLM)
