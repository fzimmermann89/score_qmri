"""Fixed-sigma pULA for a small centered single-coil Cartesian acquisition.

Input NPZ: complex 2D ``measurement`` (zero at unmeasured locations), 2D
binary ``mask``, and optional physical complex 2D ``initial``. Both k-space
arrays use mrpro's centered orthonormal Fourier convention. ``noise_std`` is
the physical SD per real coordinate. Use a VE/EDM checkpoint as in DPS/DAPS.
This example retains correlated chain states; burn-in and thinning require
problem-specific mixing assessment. It does not tune reconstruction settings.

For general MRI, pass a caller-composed tuple-returning operator directly to
PULASampler.sample. For a single nonlinear map, also supply a fixed positive
model-coordinate diagonal. Joint multi-map priors are outside this API.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from jsonargparse import CLI
from mrpro.operators import FastFourierOp

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.pula_sampler import PULAConfig, PULASampler


def main(
    checkpoint: Path,
    case: Path,
    output: Path,
    noise_std: float,
    cfg: PULAConfig = PULAConfig(num_samples=8, thinning=10),
    kind: DatasetKind = DatasetKind.M0_COMPLEX,
    chains: int = 2,
    seed: int = 20261031,
    device: str = "cpu",
) -> None:
    """Save physical pULA chain states and the settings needed to interpret them."""
    if not kind.is_complex or chains < 1:
        raise ValueError("This single-coil example requires a complex dataset kind and positive chain count")
    model = DiffusionModel.from_checkpoint(checkpoint, compile_model=False, weights="ema").to(device).eval()
    with np.load(case) as arrays:
        measurement = torch.as_tensor(arrays["measurement"], device=device)
        mask = torch.as_tensor(arrays["mask"], device=device)
        if measurement.ndim != 2 or not measurement.is_complex() or mask.shape != measurement.shape:
            raise ValueError("Require 2D complex measurement and matching binary mask")
        if not torch.all((mask == 0) | (mask == 1)) or torch.any(measurement[mask == 0] != 0):
            raise ValueError("Require a binary mask and zero-filled unmeasured entries")
        template = measurement.new_empty(chains, 1, 1, *measurement.shape)
        initial = None
        if "initial" in arrays:
            warm_start = torch.as_tensor(arrays["initial"], device=device)
            if warm_start.shape != measurement.shape or not warm_start.is_complex():
                raise ValueError("initial must be a physical complex image with measurement's 2D shape")
            initial = warm_start.expand_as(template)
    operator = mask * FastFourierOp(dim=(-2, -1)).to(device)
    (samples,) = PULASampler(model, cfg).sample(
        operator,
        measurement[None, None, None],
        noise_std=noise_std,
        template=template,
        kind=kind,
        generator=torch.Generator(device=device).manual_seed(seed),
        initial=initial,
        return_chain=True,
    )
    np.savez_compressed(
        output, samples=samples.cpu().numpy(), noise_std=noise_std, kind=kind.name, seed=seed, checkpoint=str(checkpoint), **asdict(cfg)
    )


if __name__ == "__main__":
    CLI(main)
