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
score-qmri fit --config data.yaml --config model.yaml --job_name my-run
```

Training datasets and loader settings are in `data.yaml`; model and optimizer overrides are in `model.yaml`. Pass both files in that order. Dataset paths resolve relative to the working directory.

See [checkpoint compatibility](docs/checkpoints.md) for strict model loading and portable checkpoint conversion.

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

See [reconstruction smoke checks](docs/reconstruction-smoke.md) for bounded retrospective model/operator diagnostics.
