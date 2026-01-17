# Score-Based Posterior Sampling for Quantitative MRI

Diffusion model framework for learning MRI priors (2-channel real/imag or mag/zeros) for qMRI reconstruction.

## Highlights
- VP-SDE diffusion with dual heads (noise ε + clean x₀)
- FiLM conditioning on timestep + dataset kind
- Pre-Norm UNet with optional SDPA attention (`attention_levels`)
- Cosine or linear noise schedule (cosine default)
- Fixed per-kind normalization for mapping kinds (`kind_scales`)
- DDIM validation sampling with fixed noise

## Quick Start
```bash
source .venv/bin/activate
score-qmri-train fit --config config.yaml
```

## Key Config (config.yaml)
```yaml
# Values shown are defaults in code unless explicitly set in config.yaml.
model:
  attention_levels: [3, 4]
  head_dim: 64
  noise_schedule: cosine
  learning_rate: 1e-4
  normalization:
    train_clip: 1.2
    sample_clip: 1.0
    kind_scales:
      T1MAP: 0.0003
      T2MAP: 0.005
      T2STARMAP: 0.01

# Optional overrides (defaults in code)
# model.base_channels: 64
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
- Song et al., ICLR 2021 (Score-Based SDEs)
- Ho et al., NeurIPS 2020 (DDPM)
- Nichol & Dhariwal, 2021 (Cosine schedule)
- Perez et al., AAAI 2018 (FiLM)
