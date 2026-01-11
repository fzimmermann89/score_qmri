"""Image generation from trained diffusion models."""

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from .ddim_sampler import DDIMSampler
from .ddpm_sampler import DDPMSampler


def generate_samples(
    model: DiffusionModel,
    kind: DatasetKind,
    n_samples: int = 1,
    size: int = 256,
    sampler_type: str = "ddim",
    num_steps: int = 100,
    eta: float = 0.0,
    seed: int | None = None,
) -> Tensor:
    """Generate samples from diffusion model.

    Parameters
    ----------
    model
        Trained DiffusionModel in eval mode.
    kind
        DatasetKind to condition generation on.
    n_samples
        Number of samples to generate (batch size).
    size
        Image size (height = width).
    sampler_type
        Either "ddim" or "ddpm".
    num_steps
        Number of sampling steps.
    eta
        DDIM stochasticity parameter (0 = deterministic).
    seed
        Random seed for reproducibility. None for random.

    Returns
    -------
    Generated samples of shape (N, H, W), denormalized.
    """
    device = model.device

    # Use device-specific generator for reproducibility on CUDA
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    if kind.is_complex:
        noise = torch.complex(
            torch.randn(n_samples, size, size, device=device, generator=gen),
            torch.randn(n_samples, size, size, device=device, generator=gen),
        )
    else:
        noise = torch.randn(n_samples, size, size, device=device, generator=gen)

    kind_tensor = torch.full((n_samples,), int(kind), dtype=torch.long, device=device)

    sampler: DDIMSampler | DDPMSampler
    if sampler_type.lower() == "ddim":
        sampler = DDIMSampler(model, num_steps=num_steps, eta=eta)
    elif sampler_type.lower() == "ddpm":
        sampler = DDPMSampler(model, num_steps=num_steps)
    else:
        raise ValueError(f"Unknown sampler: {sampler_type!r}. Use 'ddim' or 'ddpm'.")

    with torch.no_grad():
        samples = sampler.sample((n_samples, size, size), kind_tensor, noise)
        samples_denorm = model.denormalize(samples, kind_tensor)

    return samples_denorm
