"""DDIM sampler for diffusion models."""

import torch
from torch import Tensor

from models.diffusion import DiffusionModel


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) sampler."""

    def __init__(self, model: DiffusionModel, num_steps: int = 100, eta: float = 0.0) -> None:
        """
        Initialize DDIM sampler.

        Parameters
        ----------
        model
            Diffusion model instance.
        num_steps
            Number of sampling steps.
        eta
            Stochasticity parameter (0 = deterministic, 1 = DDPM).
        """
        self.model = model
        self.num_steps = num_steps
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """
        Sample from model using DDIM.

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        device
            Computation device.
        kind
            Dataset kind indices of shape (B,).
        initial_noise
            Optional fixed noise for reproducibility.

        Returns
        -------
        Samples of shape (B, 2, H, W).
        """
        B = shape[0]
        x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
        timesteps = torch.linspace(self.model.num_timesteps - 1, 0, self.num_steps + 1, device=device).long()

        for i in range(self.num_steps):
            t = timesteps[i].expand(B)
            t_prev = int(timesteps[i + 1].item())
            x = self._step(x, t, t_prev, kind)

        return x

    def _step(self, x: Tensor, t: Tensor, t_prev: int, kind: Tensor) -> Tensor:
        """Execute single DDIM reverse step."""
        t_val = int(t[0].item())
        output = self.model(x, t, kind)

        alpha_t = self.model.alphas_cumprod[t_val]
        alpha_prev = self.model.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        x0_pred = self.model.predict_x0(x, t, output.noise)

        sigma = self.eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        dir_xt = (1 - alpha_prev - sigma**2).sqrt() * output.noise
        x_prev: Tensor = alpha_prev.sqrt() * x0_pred + dir_xt

        if self.eta > 0 and t_prev > 0:
            x_prev = x_prev + sigma * torch.randn_like(x)

        return x_prev


def create_fixed_noise(num_kinds: int, image_shape: tuple[int, int], seed: int = 42) -> dict[int, Tensor]:
    """
    Create reproducible noise for each dataset kind.

    Parameters
    ----------
    num_kinds
        Number of dataset types.
    image_shape
        (H, W) shape of images.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Mapping from kind index to noise tensor of shape (1, 2, H, W).
    """
    gen = torch.Generator().manual_seed(seed)
    return {k: torch.randn(1, 2, *image_shape, generator=gen) for k in range(num_kinds)}
