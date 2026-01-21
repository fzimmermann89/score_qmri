"""EDM/Karras sampler for VE diffusion models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel, _create_noise

Solver = Literal["euler", "heun"]


def _enforce_real_imag_zero_(x: Tensor, real_mask: Tensor) -> None:
    """In-place: set imag channel to zero for real kinds."""
    if real_mask.any():
        x[real_mask, 1] = 0.0


def karras_sigmas(
    num_sigmas: int,
    *,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: torch.device,
) -> Tensor:
    """Construct a Karras (poly-rho) sigma schedule.

    Returns a tensor of length `num_sigmas` starting at `sigma_max` and ending
    at `sigma_min`.
    """
    if num_sigmas < 2:
        raise ValueError("num_sigmas must be >= 2")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be > 0")
    if sigma_min >= sigma_max:
        raise ValueError("sigma_min must be < sigma_max")
    if rho <= 0:
        raise ValueError("rho must be > 0")

    i = torch.linspace(0.0, 1.0, num_sigmas, device=device)
    inv_rho = 1.0 / rho
    sigma_min_t = torch.tensor(sigma_min, device=device, dtype=torch.float32)
    sigma_max_t = torch.tensor(sigma_max, device=device, dtype=torch.float32)
    ramp = sigma_max_t**inv_rho + i * (sigma_min_t**inv_rho - sigma_max_t**inv_rho)
    sigmas = ramp**rho
    return sigmas


@dataclass
class SamplingStats:
    """Optional per-step statistics for debugging."""

    steps: list[int] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    x_min: list[float] = field(default_factory=list)
    x_max: list[float] = field(default_factory=list)
    x_mean: list[float] = field(default_factory=list)
    x_std: list[float] = field(default_factory=list)
    x0_min: list[float] = field(default_factory=list)
    x0_max: list[float] = field(default_factory=list)
    x0_mean: list[float] = field(default_factory=list)
    x0_std: list[float] = field(default_factory=list)
    intermediates: list[Tensor] = field(default_factory=list)


class EDMSampler:
    """Deterministic EDM/Karras sampler (Euler or Heun)."""

    def __init__(
        self,
        model: DiffusionModel,
        *,
        num_steps: int = 50,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        rho: float = 7.0,
        solver: Solver = "heun",
        clamp_denoised: bool = True,
    ) -> None:
        self.model = model
        self.num_steps = int(num_steps)
        self.sigma_min = float(model.sigma_min if sigma_min is None else sigma_min)
        self.sigma_max = float(model.sigma_max if sigma_max is None else sigma_max)
        self.rho = float(rho)
        self.solver: Solver = solver
        self.clamp_denoised = bool(clamp_denoised)
        self.last_stats: SamplingStats | None = None

    def sample(
        self,
        shape: tuple[int, ...],
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
        collect_stats: bool = False,
    ) -> Tensor:
        """Sample using EDM/Karras schedule.

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        kind
            Dataset kind indices of shape (B,).
        initial_noise
            Unit Gaussian noise of shape (B, 2, H, W) (or (B, H, W) for real).
            If None, sampled from N(0, 1).
        collect_stats
            If True, collect per-step statistics.

        Returns
        -------
        Tensor
            Denoised sample x0_hat in normalized space (B, 2, H, W).
        """
        device = self.model.device
        B = shape[0]
        kind = torch.zeros(B, dtype=torch.long, device=device) if kind is None else kind.to(device)
        real_mask = ~is_complex_kind(kind)

        if initial_noise is not None:
            x = initial_noise.to(device)
            if x.ndim == 3:
                x = torch.stack([x, torch.zeros_like(x)], dim=1)
        else:
            x = _create_noise(shape, kind, device, torch.float32)

        # Scale first to avoid mutating initial_noise in-place, then enforce constraints.
        x = x * self.sigma_max
        _enforce_real_imag_zero_(x, real_mask)

        sigmas = karras_sigmas(
            self.num_steps + 1,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )

        stats = SamplingStats() if collect_stats else None

        with torch.no_grad():
            for i in range(self.num_steps):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                x = self._step(x, kind, real_mask=real_mask, sigma=sigma, sigma_next=sigma_next, step_idx=i, stats=stats)

        sigma_final = sigmas[-1]
        x0 = self.model.denoise(x, sigma_final, kind)
        x0 = self.model.clamp_x0(x0, kind) if self.clamp_denoised else x0
        self.last_stats = stats
        return x0

    def _step(
        self,
        x: Tensor,
        kind: Tensor,
        real_mask: Tensor,
        *,
        sigma: Tensor,
        sigma_next: Tensor,
        step_idx: int,
        stats: SamplingStats | None,
    ) -> Tensor:
        x0 = self.model.denoise(x, sigma, kind)
        x0 = self.model.clamp_x0(x0, kind) if self.clamp_denoised else x0

        if stats is not None:
            should_log = step_idx < 5 or step_idx % 25 == 0 or step_idx == self.num_steps - 1
            if should_log:
                # Log magnitude for readability. For real kinds imag is enforced to zero.
                x_img = x.square().sum(dim=1).sqrt()
                x0_img = x0.square().sum(dim=1).sqrt()

                stats.steps.append(step_idx)
                stats.sigma.append(float(sigma.item()))
                stats.x_min.append(x_img.min().item())
                stats.x_max.append(x_img.max().item())
                stats.x_mean.append(x_img.mean().item())
                stats.x_std.append(x_img.std().item())
                stats.x0_min.append(x0_img.min().item())
                stats.x0_max.append(x0_img.max().item())
                stats.x0_mean.append(x0_img.mean().item())
                stats.x0_std.append(x0_img.std().item())
                stats.intermediates.append(x0.detach().cpu())

        sigma_in = sigma.to(dtype=x.dtype).clamp(min=1e-8).reshape(1, 1, 1, 1)
        d1 = (x - x0) / sigma_in
        dt = (sigma_next - sigma).to(dtype=x.dtype)
        x_euler = x + dt * d1
        _enforce_real_imag_zero_(x_euler, real_mask)

        if self.solver == "euler" or step_idx == self.num_steps - 1:
            return x_euler

        x0_next = self.model.denoise(x_euler, sigma_next, kind)
        x0_next = self.model.clamp_x0(x0_next, kind) if self.clamp_denoised else x0_next
        sigma_next_in = sigma_next.to(dtype=x.dtype).clamp(min=1e-8).reshape(1, 1, 1, 1)
        d2 = (x_euler - x0_next) / sigma_next_in
        x_heun = x + dt * (d1 + d2) / 2.0
        _enforce_real_imag_zero_(x_heun, real_mask)
        return x_heun


def create_fixed_noise(image_shape: tuple[int, int], num_samples: int = 4, seed: int = 42) -> dict[int, Tensor]:
    """Create reproducible unit Gaussian noise for each dataset kind in 2ch format."""
    gen = torch.Generator().manual_seed(seed)
    result: dict[int, Tensor] = {}
    for kind in DatasetKind:
        shape = (num_samples, 2, *image_shape)
        noise = torch.randn(shape, generator=gen)
        if not is_complex_kind(int(kind)):
            noise[:, 1] = 0.0
        result[int(kind)] = noise
    return result
