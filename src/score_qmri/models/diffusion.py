"""Continuous-time VP-SDE diffusion model with dual prediction heads.

This module implements a diffusion model using the Variance Preserving SDE (VP-SDE)
framework (Song et al., ICLR 2021). Time is continuous t ∈ [0, 1], allowing flexible
inference with any number of discretization steps.

The model predicts both:
- Noise ε (trained with L_simple MSE, used for score/sampling)
- Clean image x0 (trained with SSIM + L1, auxiliary denoising task)
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import Literal, cast, overload

import torch
from torch import Tensor

from score_qmri.datasets.kind import is_complex_kind

from .conditioning import ConditioningMLP
from .normalization import RunningNormalizer
from .unet import UNet

NoiseSchedule = Literal["linear", "cosine"]


class InputFormat(IntEnum):
    REAL_1CH = auto()
    COMPLEX_1CH = auto()
    REAL_2CH = auto()
    LIST_2CH = auto()


@dataclass
class ListMetadata:
    """Metadata for reconstructing list outputs from concatenated 2ch tensor."""

    formats: list[InputFormat]
    batch_sizes: list[int]


@dataclass
class DiffusionOutput:
    x0: Tensor
    noise: Tensor


@dataclass
class DiffusionOutputList:
    x0: list[Tensor]
    noise: list[Tensor]


def _to_2ch_single(x: Tensor) -> tuple[Tensor, InputFormat]:
    """Convert single tensor to canonical (B, 2, H, W) format."""
    match (x.ndim, x.is_complex()):
        case (4, False) if x.shape[1] == 2:
            return x, InputFormat.REAL_2CH
        case (3, True):
            return torch.moveaxis(torch.view_as_real(x), -1, 1), InputFormat.COMPLEX_1CH
        case (3, False):
            return torch.stack([x, torch.zeros_like(x)], dim=1), InputFormat.REAL_1CH
        case _:
            raise ValueError(f"Unsupported input shape/dtype: {x.shape}, complex={x.is_complex()}")


def _to_2ch(x: Tensor | Sequence[Tensor], kind: Tensor | None = None) -> tuple[Tensor, InputFormat, ListMetadata | None]:
    """Convert input to canonical (B, 2, H, W) format.

    For list/sequence inputs, tensors are concatenated along the batch dimension.
    Each element can be real (B_i, H, W), complex (B_i, H, W), or 2ch (B_i, 2, H, W).

    Returns
    -------
    x_2ch
        Tensor in canonical format (B, 2, H, W).
    fmt
        Input format (for reconstruction).
    metadata
        For list inputs, contains per-element formats and batch sizes.
        None for single tensor inputs.
    """
    if isinstance(x, (list, tuple)):
        if not all(isinstance(tensor, Tensor) for tensor in x):
            bad = next(tensor for tensor in x if not isinstance(tensor, Tensor))
            raise TypeError(f"List elements must be Tensors, got {type(bad)}")
        converted = [_to_2ch_single(tensor) for tensor in x]
        chunks_2ch = [tensor for tensor, _ in converted]
        formats = [fmt for _, fmt in converted]
        batch_sizes = [tensor.shape[0] for tensor in chunks_2ch]
        x_2ch = torch.cat(chunks_2ch, dim=0)
        if kind is not None:
            x_2ch = _apply_kind_constraints(x_2ch, kind)
        return x_2ch, InputFormat.LIST_2CH, ListMetadata(formats, batch_sizes)
    x_2ch, fmt = _to_2ch_single(cast(Tensor, x))
    if kind is not None:
        x_2ch = _apply_kind_constraints(x_2ch, kind)
    return x_2ch, fmt, None


def _from_2ch_single(x_2ch: Tensor, fmt: InputFormat) -> Tensor:
    match fmt:
        case InputFormat.REAL_2CH:
            return x_2ch
        case InputFormat.COMPLEX_1CH:
            return torch.view_as_complex(torch.moveaxis(x_2ch, 1, -1).contiguous())
        case InputFormat.REAL_1CH:
            return x_2ch[:, 0]
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def _from_2ch(x_2ch: Tensor, fmt: InputFormat, kind: Tensor | None = None, metadata: ListMetadata | None = None) -> Tensor | list[Tensor]:
    """Convert canonical (B, 2, H, W) back to original format."""
    if kind is not None:
        x_2ch = _apply_kind_constraints(x_2ch, kind)

    match fmt:
        case InputFormat.LIST_2CH:
            if metadata is None:
                raise ValueError("LIST_2CH format requires metadata")
            chunks = x_2ch.split(metadata.batch_sizes, dim=0)
            return [_from_2ch_single(chunk, elem_fmt) for chunk, elem_fmt in zip(chunks, metadata.formats, strict=True)]
        case _:
            return _from_2ch_single(x_2ch, fmt)


def _apply_kind_constraints(x_2ch: Tensor, kind: Tensor) -> Tensor:
    """Zero out imaginary channel for real kinds."""
    result = x_2ch.clone()
    for k in kind.unique():
        k_idx = int(k.item())
        if not is_complex_kind(k_idx):
            idx = kind == k
            result[idx, 1] = 0.0
    return result


def _create_noise(shape: tuple[int, ...], kind: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create noise with imag=0 for real kinds."""
    noise = torch.randn(shape, device=device, dtype=dtype)
    return _apply_kind_constraints(noise, kind)


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
        attention_levels: tuple[int, ...] = (),
        head_dim: int = 64,
        noise_schedule: NoiseSchedule = "linear",
    ) -> None:
        """
        Parameters
        ----------
        beta_min
            Minimum noise rate β(0) (for linear schedule).
        beta_max
            Maximum noise rate β(1) (for linear schedule).
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
        attention_levels
            Tuple of level indices where self-attention is applied (0-indexed).
            E.g., (3, 4) applies attention at 32×32 and 16×16 for 256×256 input.
        head_dim
            Dimension per attention head (default 64, following Stable Diffusion).
        noise_schedule
            Noise schedule type: "linear" (VP-SDE default) or "cosine" (Nichol & Dhariwal).
        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.noise_schedule = noise_schedule

        self.cond_mlp = ConditioningMLP(cond_dim=cond_dim, num_kinds=num_kinds)
        unet = UNet(
            in_channels=2,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            cond_dim=cond_dim,
            attention_levels=attention_levels,
            head_dim=head_dim,
        )
        unet.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        self.unet = torch.compile(unet, mode="reduce-overhead")
        self.normalizer = RunningNormalizer(num_kinds=num_kinds)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def beta(self, t: Tensor) -> Tensor:
        """Instantaneous noise rate β(t)."""
        if self.noise_schedule == "cosine":
            t = t.float().clamp(min=0.0, max=1.0 - 1e-5)
            theta = (t + 0.008) / (1 + 0.008) * math.pi / 2
            return math.pi / (1 + 0.008) * torch.tan(theta)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t: Tensor) -> Tensor:
        """Cumulative signal variance ᾱ(t) = exp(-∫₀ᵗ β(s)ds)."""
        if self.noise_schedule == "cosine":
            return self._alpha_bar_cosine(t)
        return torch.exp(-self.beta_min * t - 0.5 * (self.beta_max - self.beta_min) * t**2)

    def _alpha_bar_cosine(self, t: Tensor, s: float = 0.008) -> Tensor:
        """Cosine schedule: ᾱ(t) = f(t)/f(0) where f(t) = cos²((t+s)/(1+s) * π/2)."""
        # Ensure t is float32 to avoid precision issues
        t = t.float()
        f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        return f_t / f_0

    def sigma(self, t: Tensor) -> Tensor:
        """Noise standard deviation σ(t) = √(1 - ᾱ(t))."""
        return (1.0 - self.alpha_bar(t)).clamp(min=0.0).sqrt()

    def snr(self, t: Tensor) -> Tensor:
        """Signal-to-noise ratio SNR(t) = ᾱ(t) / (1 - ᾱ(t))."""
        ab = self.alpha_bar(t)
        return ab / (1.0 - ab).clamp(min=1e-8)

    @overload
    def normalize(self, x: Tensor, kind: Tensor, mask: Tensor) -> Tensor: ...
    @overload
    def normalize(self, x: Sequence[Tensor], kind: Tensor, mask: Tensor) -> list[Tensor]: ...
    def normalize(self, x: Tensor | Sequence[Tensor], kind: Tensor, mask: Tensor) -> Tensor | list[Tensor]:
        """Normalize input using per-kind running stats.

        Accepts tensor or list of tensors in (B, H, W) real/complex or (B, 2, H, W).
        Returns same format as input (list returns list).
        """
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x_norm = self.normalizer.normalize(x_2ch, mask, kind)
        return _from_2ch(x_norm, fmt, kind, metadata)

    @overload
    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor: ...
    @overload
    def denormalize(self, x: Sequence[Tensor], kind: Tensor) -> list[Tensor]: ...
    def denormalize(self, x: Tensor | Sequence[Tensor], kind: Tensor) -> Tensor | list[Tensor]:
        """Denormalize output back to original scale.

        Accepts tensor or list of tensors in (B, H, W) real/complex or (B, 2, H, W).
        Returns same format as input (list returns list).
        """
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x_denorm = self.normalizer.denormalize(x_2ch, kind)
        return _from_2ch(x_denorm, fmt, kind, metadata)

    @overload
    def q_sample(
        self,
        x0: Tensor,
        t: Tensor,
        kind: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor: ...
    @overload
    def q_sample(
        self,
        x0: Sequence[Tensor],
        t: Tensor,
        kind: Tensor,
        noise: Sequence[Tensor] | None = None,
    ) -> list[Tensor]: ...
    def q_sample(
        self,
        x0: Tensor | Sequence[Tensor],
        t: Tensor,
        kind: Tensor,
        noise: Tensor | Sequence[Tensor] | None = None,
    ) -> Tensor | list[Tensor]:
        """Sample from q(xₜ | x₀): xₜ = √ᾱ(t)·x₀ + √(1-ᾱ(t))·ε.

        Accepts tensor or list of tensors in (B, H, W) real/complex or (B, 2, H, W).
        Returns same format as input (list returns list).
        For real kinds, noise has zero imaginary component.
        """
        x0_2ch, fmt, metadata = _to_2ch(x0, kind)

        if noise is None:
            noise_2ch = _create_noise(x0_2ch.shape, kind, x0_2ch.device, x0_2ch.dtype)
        else:
            noise_2ch, _, _ = _to_2ch(noise, kind)

        ab = self.alpha_bar(t).view(-1, 1, 1, 1)
        x_t_2ch = ab.clamp(min=0.0).sqrt() * x0_2ch + (1.0 - ab).clamp(min=0.0).sqrt() * noise_2ch

        return _from_2ch(x_t_2ch, fmt, kind, metadata)

    @overload
    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput: ...
    @overload
    def forward(self, x_t: Sequence[Tensor], t: Tensor, kind: Tensor) -> DiffusionOutputList: ...
    def forward(self, x_t: Tensor | Sequence[Tensor], t: Tensor, kind: Tensor) -> DiffusionOutput | DiffusionOutputList:
        """Predict x₀ and noise from noisy input.

        Accepts tensor or list of tensors in (B, H, W) real/complex or (B, 2, H, W).
        Returns DiffusionOutputList for list inputs, DiffusionOutput otherwise.
        For real kinds, imaginary channel of outputs is zeroed.
        """
        x_t_2ch, fmt, metadata = _to_2ch(x_t, kind)
        cond = self.cond_mlp(t, kind)
        x0_pred_2ch, noise_pred_2ch = self.unet(x_t_2ch, cond)

        x0_pred = _from_2ch(x0_pred_2ch, fmt, kind, metadata)
        noise_pred = _from_2ch(noise_pred_2ch, fmt, kind, metadata)

        if fmt == InputFormat.LIST_2CH:
            return DiffusionOutputList(x0=cast(list[Tensor], x0_pred), noise=cast(list[Tensor], noise_pred))
        return DiffusionOutput(x0=cast(Tensor, x0_pred), noise=cast(Tensor, noise_pred))

    @overload
    def predict_x0(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
        kind: Tensor,
    ) -> Tensor: ...
    @overload
    def predict_x0(
        self,
        x_t: Sequence[Tensor],
        t: Tensor,
        noise: Sequence[Tensor],
        kind: Tensor,
    ) -> list[Tensor]: ...
    def predict_x0(
        self,
        x_t: Tensor | Sequence[Tensor],
        t: Tensor,
        noise: Tensor | Sequence[Tensor],
        kind: Tensor,
    ) -> Tensor | list[Tensor]:
        """Recover x₀ = (xₜ - σ(t)·ε) / √ᾱ(t).

        Accepts tensor or list of tensors in (B, H, W) real/complex or (B, 2, H, W).
        Returns same format as input (list returns list).
        """
        x_t_2ch, fmt, metadata = _to_2ch(x_t, kind)
        noise_2ch, _, _ = _to_2ch(noise, kind)

        ab = self.alpha_bar(t).view(-1, 1, 1, 1)
        x0_2ch = (x_t_2ch - (1.0 - ab).clamp(min=0.0).sqrt() * noise_2ch) / ab.clamp(min=0.0).sqrt().clamp(min=1e-8)

        return _from_2ch(x0_2ch, fmt, kind, metadata)

    @overload
    def score(self, x_t: Tensor, t: Tensor, kind: Tensor) -> Tensor: ...
    @overload
    def score(self, x_t: Sequence[Tensor], t: Tensor, kind: Tensor) -> list[Tensor]: ...
    def score(self, x_t: Tensor | Sequence[Tensor], t: Tensor, kind: Tensor) -> Tensor | list[Tensor]:
        """Compute score ∇ₓ log p(xₜ) = -ε / σ(t).

        Accepts tensor or list of tensors in (B, H, W) real/complex or (B, 2, H, W).
        Returns same format as input (list returns list).
        """
        output = self.forward(x_t, t, kind)

        x_t_2ch, fmt, metadata = _to_2ch(x_t, kind)
        noise_2ch, _, _ = _to_2ch(output.noise, kind)

        sigma_t = self.sigma(t).view(-1, 1, 1, 1)
        score_2ch = -noise_2ch / sigma_t.clamp(min=1e-8)

        return _from_2ch(score_2ch, fmt, kind, metadata)

    @overload
    def __call__(self, x_t: Tensor, t: Tensor, kind: Tensor) -> DiffusionOutput: ...
    @overload
    def __call__(self, x_t: Sequence[Tensor], t: Tensor, kind: Tensor) -> DiffusionOutputList: ...
    def __call__(self, x_t: Tensor | Sequence[Tensor], t: Tensor, kind: Tensor) -> DiffusionOutput | DiffusionOutputList:
        return self.forward(x_t, t, kind)

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load weights from checkpoint, preferring EMA shadow weights if available."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "ema" in checkpoint:
            self.load_state_dict(checkpoint["ema"]["shadow"])
        else:
            state_dict = checkpoint["state_dict"]
            model_state = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
            self.load_state_dict(model_state)
