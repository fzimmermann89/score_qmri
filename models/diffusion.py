"""Diffusion model with dual prediction heads."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .conditioning import ConditioningMLP
from .unet import UNet


@dataclass
class DiffusionOutput:
    """Output from diffusion model forward pass."""

    x0: Tensor  # Clean image prediction (B, 2, H, W)
    noise: Tensor  # Noise prediction (B, 2, H, W)


class DiffusionModel(nn.Module):
    """Diffusion model predicting both x0 and noise.

    Args:
        num_kinds: Number of dataset types for conditioning.
        num_timesteps: Total diffusion timesteps.
        beta_start: Starting noise schedule value.
        beta_end: Ending noise schedule value.
        base_channels: UNet base channel count.
        num_blocks: UNet encoder/decoder depth.
        cond_dim: Conditioning embedding dimension.
    """

    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor

    def __init__(
        self,
        num_kinds: int = 3,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        base_channels: int = 64,
        num_blocks: int = 4,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_kinds = num_kinds

        self.cond_mlp = ConditioningMLP(num_kinds, cond_dim)
        self.unet = UNet(
            in_channels=2,
            base_channels=base_channels,
            num_blocks=num_blocks,
            cond_dim=cond_dim,
        )

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput:
        """Predict x0 and noise from noisy input.

        Args:
            x_t: Noisy input (B, 2, H, W).
            t: Timesteps (B,).
            kind: Dataset kind indices (B,).

        Returns:
            DiffusionOutput with x0 and noise predictions.
        """
        cond = self.cond_mlp(t, kind)
        x0, noise = self.unet(x_t, cond)
        return DiffusionOutput(x0=x0, noise=noise)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = rearrange(self.sqrt_alphas_cumprod[t], "b -> b 1 1 1")
        sqrt_one_minus = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Compute x0 from x_t and noise prediction."""
        sqrt_alpha = rearrange(self.sqrt_alphas_cumprod[t], "b -> b 1 1 1")
        sqrt_one_minus = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        return (x_t - sqrt_one_minus * noise) / sqrt_alpha.clamp(min=1e-8)

    def score(self, x_t: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Compute score function: -noise / sqrt(1 - alpha_bar).

        The score is the gradient of log p(x_t).
        """
        output = self.forward(x_t, t, kind)
        sqrt_one_minus = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        return -output.noise / sqrt_one_minus.clamp(min=1e-8)

    @staticmethod
    def prepare_input(image: Tensor) -> Tensor:
        """Prepare 2-channel input from image.

        Complex: already (B, 2, H, W) -> pass through
        Magnitude: (B, H, W) -> stack with zeros
        """
        if image.ndim == 4 and image.shape[1] == 2:
            return image
        if image.ndim == 4:
            image = image.squeeze(1)
        return torch.stack([image, torch.zeros_like(image)], dim=1)
