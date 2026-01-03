"""Continuous-time VP-SDE diffusion model with dual prediction heads.

This module implements a diffusion model using the Variance Preserving SDE (VP-SDE)
framework (Song et al., ICLR 2021). Time is continuous t ∈ [0, 1], allowing flexible
inference with any number of discretization steps.

The model predicts both:
- Noise ε (trained with L_simple MSE, used for score/sampling)
- Clean image x0 (trained with SSIM + L1, auxiliary denoising task)
"""

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor

from .conditioning import ConditioningMLP
from .unet import UNet


@dataclass
class DiffusionOutput:
    """Output from diffusion model forward pass."""

    x0: Tensor  # (B, 2, H, W)
    noise: Tensor  # (B, 2, H, W)


class DiffusionModel(torch.nn.Module):
    """VP-SDE diffusion model with continuous time t ∈ [0, 1].

    The forward SDE is: dx = -½β(t)x dt + √β(t) dw
    with linear noise schedule: β(t) = β_min + t(β_max - β_min)

    Key quantities:
        ᾱ(t) = exp(-∫₀ᵗ β(s)ds) = exp(-β_min·t - ½(β_max - β_min)·t²)
        σ(t) = √(1 - ᾱ(t))
        SNR(t) = ᾱ(t) / (1 - ᾱ(t))

    At t=0: ᾱ=1, σ=0 (clean image)
    At t=1: ᾱ≈0, σ≈1 (pure noise)
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        base_channels: int = 64,
        num_blocks: int = 4,
        cond_dim: int = 128,
        num_kinds: int = 14,
    ) -> None:
        """
        Parameters
        ----------
        beta_min
            Minimum noise rate β(0).
        beta_max
            Maximum noise rate β(1).
        base_channels
            UNet base channel count (doubled at each level).
        num_blocks
            UNet encoder/decoder depth.
        cond_dim
            Conditioning embedding dimension.
        num_kinds
            Number of dataset kind classes for embedding.
        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.cond_mlp = ConditioningMLP(cond_dim=cond_dim, num_kinds=num_kinds)
        self.unet = UNet(
            in_channels=2,
            base_channels=base_channels,
            num_blocks=num_blocks,
            cond_dim=cond_dim,
        )

    @property
    def device(self) -> torch.device:
        """Get device of the model (inferred from first parameter)."""
        return next(self.parameters()).device

    def beta(self, t: Tensor) -> Tensor:
        """Instantaneous noise rate β(t) = β_min + t(β_max - β_min)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t: Tensor) -> Tensor:
        """Cumulative signal variance ᾱ(t) = exp(-∫₀ᵗ β(s)ds)."""
        return torch.exp(-self.beta_min * t - 0.5 * (self.beta_max - self.beta_min) * t**2)

    def sigma(self, t: Tensor) -> Tensor:
        """Noise standard deviation σ(t) = √(1 - ᾱ(t))."""
        return (1.0 - self.alpha_bar(t)).sqrt()

    def snr(self, t: Tensor) -> Tensor:
        """Signal-to-noise ratio SNR(t) = ᾱ(t) / (1 - ᾱ(t))."""
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab).clamp(min=1e-8)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        Sample from q(xₜ | x₀): xₜ = √ᾱ(t)·x₀ + √(1-ᾱ(t))·ε.

        Parameters
        ----------
        x0
            Clean images (B, 2, H, W).
        t
            Continuous time in [0, 1] of shape (B,).
        noise
            Optional pre-sampled noise. Defaults to N(0, I).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = rearrange(self.alpha_bar(t), "b -> b 1 1 1")
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput:
        """
        Predict x₀ and noise from noisy input.

        Parameters
        ----------
        x_t
            Noisy input (B, 2, H, W).
        t
            Continuous time in [0, 1] of shape (B,).
        kind
            Dataset kind indices (B,).
        """
        cond = self.cond_mlp(t, kind)
        x0_pred, noise_pred = self.unet(x_t, cond)
        return DiffusionOutput(x0=x0_pred, noise=noise_pred)

    def predict_x0(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Recover x₀ = (xₜ - σ(t)·ε) / √ᾱ(t)."""
        ab = rearrange(self.alpha_bar(t), "b -> b 1 1 1")
        return (x_t - (1.0 - ab).sqrt() * noise) / ab.sqrt().clamp(min=1e-8)

    def score(self, x_t: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Compute score ∇ₓ log p(xₜ) = -ε / σ(t)."""
        output = self.forward(x_t, t, kind)
        sigma_t = rearrange(self.sigma(t), "b -> b 1 1 1")
        return -output.noise / sigma_t.clamp(min=1e-8)

    @staticmethod
    def prepare_input(image: Tensor) -> Tensor:
        """
        Prepare 2-channel input from image tensor.

        Complex data (B, 2, H, W) passes through unchanged.
        Magnitude data (B, 1, H, W) or (B, H, W) is stacked with zeros for imaginary part.

        Raises
        ------
        ValueError
            If 4D input has unexpected channel count.
        """
        if image.ndim == 4 and image.shape[1] == 2:
            return image
        if image.ndim == 4 and image.shape[1] == 1:
            image = image.squeeze(1)
        elif image.ndim == 4:
            raise ValueError(f"Expected 4D input with 1 or 2 channels, got {image.shape[1]}")
        return torch.stack([image, torch.zeros_like(image)], dim=1)
