"""Diffusion model wrapper around UNet with dual prediction heads."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .film_conditioning import FilmConditioning
from .unet import UNet


@dataclass
class DiffusionOutput:
    """Output from diffusion model forward pass."""

    x0_pred: Tensor  # Clean image prediction (B, 2, H, W)
    noise_pred: Tensor  # Noise prediction (B, 2, H, W)


class DiffusionModel(nn.Module):
    """Diffusion model with noise scheduling and FiLM conditioning.

    Predicts both clean image (x0) and noise (epsilon) from noisy input.
    Uses dataset-kind conditioning via FiLM.
    """

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
        self.num_kinds = num_kinds
        self.unet = unet or UNet(in_channels=2, base_channels=64, use_film=True)
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

        Args:
            image: Input image (B, H, W) or (B, 2, H, W).
            kind: Dataset kind tensor (unused, kept for API compatibility).

        Returns:
            Two-channel tensor of shape (B, 2, H, W).
        """
        if image.ndim == 4 and image.shape[1] == 2:
            return image
        if image.ndim == 4:
            image = image.squeeze(1)
        return torch.stack([image, torch.zeros_like(image)], dim=1)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample from forward diffusion q(x_t | x_0).

        Args:
            x0: Clean images of shape (B, 2, H, W).
            t: Timesteps of shape (B,).
            noise: Optional noise tensor. If None, sampled from N(0, I).

        Returns:
            Noisy images x_t of shape (B, 2, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha: Tensor = rearrange(self.sqrt_alphas_cumprod[t], "b -> b 1 1 1")
        sqrt_one_minus: Tensor = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput:
        """Predict x0 and noise from noisy x_t.

        Args:
            x_t: Noisy images of shape (B, 2, H, W).
            t: Timesteps of shape (B,).
            kind: Dataset kind of shape (B,).

        Returns:
            DiffusionOutput with x0_pred and noise_pred.
        """
        film_params = self.film_conditioning(kind)
        x0_pred, noise_pred = self.unet(x_t, film_params)
        return DiffusionOutput(x0_pred=x0_pred, noise_pred=noise_pred)

    def predict_x0_from_noise(self, x_t: Tensor, t: Tensor, noise_pred: Tensor) -> Tensor:
        """Compute x0 prediction from noise prediction using diffusion formula.

        x0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
        """
        sqrt_alpha = rearrange(self.sqrt_alphas_cumprod[t], "b -> b 1 1 1")
        sqrt_one_minus = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        x0: Tensor = (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha.clamp(min=1e-8)
        return x0
