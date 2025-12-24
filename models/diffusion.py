"""Diffusion model wrapper around UNet."""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .film_conditioning import FilmConditioning
from .unet import UNet


class DiffusionModel(nn.Module):
    """Diffusion model with noise scheduling and FiLM conditioning.

    Wraps a UNet denoiser with linear noise schedule and dataset-kind
    conditioning via FiLM.
    """

    # Declare buffer types for mypy
    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor

    def __init__(
        self,
        unet: UNet | None = None,
        num_kinds: int = 3,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.unet = unet or UNet(in_channels=2, out_channels=2, use_film=True)
        self.film_conditioning = FilmConditioning(num_kinds=num_kinds, num_blocks=self.unet.num_blocks)

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def prepare_input(self, image: Tensor, kind: Tensor) -> Tensor:
        """Prepare two-channel input from dataset sample.

        For magnitude/mapping: stacks [data, zeros].
        For complex: passes through if already 2 channels.
        """
        if image.ndim == 4 and image.shape[1] == 2:
            return image
        if image.ndim == 4:
            image = image.squeeze(1)
        return torch.stack([image, torch.zeros_like(image)], dim=1)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample from forward diffusion q(x_t | x_0).

        Args:
            x0: Clean images of shape (B, C, H, W).
            t: Timesteps of shape (B,).
            noise: Optional noise tensor. If None, sampled from N(0, I).

        Returns:
            Noisy images x_t of shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha: Tensor = rearrange(self.sqrt_alphas_cumprod[t], "b -> b 1 1 1")
        sqrt_one_minus: Tensor = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Predict x0 from noisy x_t.

        Args:
            x_t: Noisy images of shape (B, C, H, W).
            t: Timesteps of shape (B,).
            kind: Dataset kind of shape (B,).

        Returns:
            Predicted x0 of shape (B, C, H, W).
        """
        film_params = self.film_conditioning(kind)
        pred: Tensor = self.unet(x_t, film_params)
        return pred
