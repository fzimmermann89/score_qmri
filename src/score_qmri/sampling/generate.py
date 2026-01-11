"""Image generation from trained diffusion models."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.ddim_sampler import DDIMSampler
from score_qmri.sampling.ddpm_sampler import DDPMSampler


def load_ema_weights(model: DiffusionModel, checkpoint_path: Path | str) -> None:
    """Load EMA weights from checkpoint into model in-place.

    Prefers EMA shadow weights if available, falls back to regular model weights.

    Parameters
    ----------
    model
        DiffusionModel to load weights into.
    checkpoint_path
        Path to Lightning checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema" in checkpoint:
        model.load_state_dict(checkpoint["ema"]["shadow"])
    else:
        state_dict = checkpoint["state_dict"]
        model_state = {
            k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")
        }
        model.load_state_dict(model_state)


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

    # Create initial noise with correct dtype for complex vs real
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


def plot_samples_grid(
    samples: Tensor,
    output_path: Path | str | None = None,
    title: str | None = None,
) -> None:
    """Plot samples in an auto-square grid.

    Parameters
    ----------
    samples
        Tensor of shape (N, H, W), complex or real.
    output_path
        If provided, save to PNG. Otherwise show interactive window.
    title
        Optional title for the figure.
    """
    n = samples.shape[0]
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    samples_np = samples.abs().cpu().numpy() if samples.is_complex() else samples.cpu().numpy()

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n:
            ax.imshow(samples_np[i], cmap="gray")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to: {output_path}")
    else:
        plt.show()
