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

## Joint T₁/M₀ reconstruction

[The joint example](examples/qmri_joint.py) simulates undersampled Cartesian saturation-recovery measurements using MR2 and reconstructs both T₁ and complex M₀ with separate learned priors.

![Joint T1 and M0 reconstruction with CG+fit, DPS and DAPS](docs/images/qmri_joint.png)

The figure compares CG+fit with a single draw and the mean of 30 draws from DPS and DAPS, alongside sample SD, absolute mean error, and interval coverage. Coverage is descriptive for one simulated case; the sample spread is not calibrated posterior uncertainty.

```bash
python examples/qmri_joint.py sample model.ckpt case.npz results 0.2383 --method dps --draws 30
python examples/qmri_joint.py sample model.ckpt case.npz results 0.2383 --method daps --draws 30
python examples/qmri_joint.py plot results overview --t1_colors lipari.txt
```

The input NPZ contains `t1`, `m0`, `times`, a centered `mask`, and coil sensitivities `csm`; shapes and units are documented in the script. Adjust the M₀ prior scale (`0.2383` here) to the checkpoint's training normalization. The optional Lipari RGB table controls the T₁ colormap.

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
