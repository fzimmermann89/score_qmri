"""DDIM sampler for diffusion models."""

import torch
from einops import rearrange
from torch import Tensor

from score_qmri.models.diffusion import DiffusionModel


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) sampler."""

    def __init__(self, model: DiffusionModel, num_steps: int = 100, eta: float = 0.0) -> None:
        """
        Parameters
        ----------
        model
            Diffusion model instance.
        num_steps
            Number of sampling steps.
        eta
            Stochasticity (0 = deterministic, 1 = DDPM-like).
        """
        self.model = model
        self.num_steps = num_steps
        self.eta = eta

    def sample(
        self,
        shape: tuple[int, ...],
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """
        Sample from model using DDIM.

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        kind
            Dataset kind indices of shape (B,). Defaults to zeros.
        initial_noise
            Optional fixed noise for reproducibility.

        Returns
        -------
        Tensor
            Generated sample of shape (B, 2, H, W).
        """
        device = self.model.device
        with torch.no_grad():
            B = shape[0]
            kind = torch.zeros(B, dtype=torch.long, device=device) if kind is None else kind.to(device)

            x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
            timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

            for i in range(self.num_steps):
                t = timesteps[i].expand(B)
                t_prev = timesteps[i + 1].expand(B)
                x = self._step(x, t, t_prev, kind)

            return x

    def _step(self, x: Tensor, t: Tensor, t_prev: Tensor, kind: Tensor) -> Tensor:
        """Compute x_{t-1} from x_t using DDIM update rule."""
        output = self.model(x, t, kind)
        x0_pred = self.model.predict_x0(x, t, output.noise)

        ab_t = rearrange(self.model.alpha_bar(t), "b -> b 1 1 1")
        ab_prev = rearrange(self.model.alpha_bar(t_prev), "b -> b 1 1 1")

        # Clamp before sqrt to prevent NaN from numerical errors
        one_minus_ab_t = (1 - ab_t).clamp(min=1e-8)
        one_minus_ab_prev = (1 - ab_prev).clamp(min=0.0)
        sigma_sq = self.eta**2 * (one_minus_ab_prev / one_minus_ab_t * (1 - ab_t / ab_prev.clamp(min=1e-8)))
        sigma = sigma_sq.clamp(min=0.0).sqrt()

        pred_dir = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt() * output.noise
        x_prev: Tensor = ab_prev.sqrt() * x0_pred + pred_dir

        if self.eta > 0 and t_prev[0] > 1e-8:
            x_prev = x_prev + sigma * torch.randn_like(x)

        return x_prev


def create_fixed_noise(image_shape: tuple[int, int], seed: int = 42) -> dict[int, Tensor]:
    """
    Create reproducible noise for each dataset kind.

    Parameters
    ----------
    image_shape
        (H, W) shape of images.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Dictionary mapping dataset kind index to noise tensor of shape (1, 2, H, W).
    """
    from score_qmri.datasets.kind import NUM_DATASET_KINDS

    gen = torch.Generator().manual_seed(seed)
    return {k: torch.randn(1, 2, *image_shape, generator=gen) for k in range(NUM_DATASET_KINDS)}
