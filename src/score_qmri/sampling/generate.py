"""Image generation from trained diffusion models."""

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel, _create_noise

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
    Complex kinds return complex64, real kinds return float32.
    """
    device = model.device

    kind_tensor = torch.full((n_samples,), int(kind), dtype=torch.long, device=device)

    if seed is not None:
        torch.manual_seed(seed)
    shape = (n_samples, 2, size, size)
    noise = _create_noise(shape, kind_tensor, device, torch.float32)

    sampler: DDIMSampler | DDPMSampler
    if sampler_type.lower() == "ddim":
        sampler = DDIMSampler(model, num_steps=num_steps, eta=eta)
    elif sampler_type.lower() == "ddpm":
        sampler = DDPMSampler(model, num_steps=num_steps)
    else:
        raise ValueError(f"Unknown sampler: {sampler_type!r}. Use 'ddim' or 'ddpm'.")

    with torch.no_grad():
        samples = sampler.sample(shape, kind_tensor, noise)
        samples_denorm = model.denormalize(samples, kind_tensor)

    if kind.is_complex:
        return torch.complex(samples_denorm[:, 0], samples_denorm[:, 1])
    else:
        return samples_denorm[:, 0]
