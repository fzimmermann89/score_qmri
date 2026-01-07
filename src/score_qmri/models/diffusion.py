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
from torch import Tensor

from .conditioning import ConditioningMLP
from .normalization import RunningNormalizer
from .unet import UNet


@dataclass
class DiffusionOutput:
    """Output from diffusion model forward pass."""

    x0: Tensor  # (B, H, W)
    noise: Tensor  # (B, H, W)


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
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8, 16),
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
            UNet base channel count.
        num_res_blocks
            Number of ResBlocks per level.
        channel_mult
            Channel multipliers per level (e.g., (1, 2, 4, 8) gives 4 levels).
        cond_dim
            Conditioning embedding dimension.
        num_kinds
            Number of dataset kind classes for embedding.
        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.cond_mlp = ConditioningMLP(cond_dim=cond_dim, num_kinds=num_kinds)
        self.unet = torch.compile(
            UNet(
                in_channels=2,
                base_channels=base_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                cond_dim=cond_dim,
            ),
            mode="reduce-overhead",
        )
        self.normalizer = RunningNormalizer(num_kinds=num_kinds)

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
        return (1.0 - self.alpha_bar(t)).clamp(min=0.0).sqrt()

    def snr(self, t: Tensor) -> Tensor:
        """Signal-to-noise ratio SNR(t) = ᾱ(t) / (1 - ᾱ(t))."""
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab).clamp(min=1e-8)

    @staticmethod
    def prepare_input(image: Tensor) -> Tensor:
        """Convert native (B, H, W) to 2-channel (B, 2, H, W) for UNet."""
        if image.is_complex():
            return torch.moveaxis(torch.view_as_real(image), -1, 1)
        return torch.stack([image, torch.zeros_like(image)], dim=1)

    @staticmethod
    def undo_prepare(image: Tensor, to_complex: bool) -> Tensor:
        """Convert 2-channel (B, 2, H, W) back to native (B, H, W) format."""
        if to_complex:
            return torch.view_as_complex(torch.moveaxis(image, 1, -1).contiguous())
        return image[:, 0]

    def normalize(self, x: Tensor, kind: Tensor, mask: Tensor) -> Tensor:
        """Normalize native (B, H, W) input using per-kind running stats."""
        was_complex = x.is_complex()
        x_2ch = self.prepare_input(x)
        x_norm = self.normalizer.normalize(x_2ch, mask, kind)
        return self.undo_prepare(x_norm, was_complex)

    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor:
        """Denormalize native (B, H, W) output back to original scale."""
        was_complex = x.is_complex()
        x_2ch = self.prepare_input(x)
        x_denorm = self.normalizer.denormalize(x_2ch, kind)
        return self.undo_prepare(x_denorm, was_complex)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample from q(xₜ | x₀): xₜ = √ᾱ(t)·x₀ + √(1-ᾱ(t))·ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar(t).view(-1, 1, 1)
        return ab.clamp(min=0.0).sqrt() * x0 + (1.0 - ab).clamp(min=0.0).sqrt() * noise

    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput:
        """Predict x₀ and noise from noisy input. Output format matches input."""
        is_native = x_t.ndim == 3
        was_complex = x_t.is_complex()
        x_t_2ch = self.prepare_input(x_t)
        cond = self.cond_mlp(t, kind)
        x0_pred, noise_pred = self.unet(x_t_2ch, cond)
        if is_native:
            x0_pred = self.undo_prepare(x0_pred, was_complex)
            noise_pred = self.undo_prepare(noise_pred, was_complex)
        return DiffusionOutput(x0=x0_pred, noise=noise_pred)

    def predict_x0(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Recover x₀ = (xₜ - σ(t)·ε) / √ᾱ(t)."""
        ab = self.alpha_bar(t).view(-1, 1, 1)
        return (x_t - (1.0 - ab).clamp(min=0.0).sqrt() * noise) / ab.clamp(min=0.0).sqrt().clamp(min=1e-8)

    def score(self, x_t: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Compute score ∇ₓ log p(xₜ) = -ε / σ(t)."""
        output = self.forward(x_t, t, kind)
        sigma_t = self.sigma(t).view(-1, 1, 1)
        return -output.noise / sigma_t.clamp(min=1e-8)

    def __call__(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput:
        """Forward pass."""
        ret: DiffusionOutput = super().__call__(x_t, t, kind)
        return ret
