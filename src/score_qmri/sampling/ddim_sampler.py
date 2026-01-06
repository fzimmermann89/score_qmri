"""DDIM sampler for diffusion models."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel


@dataclass
class SamplingStats:
    """Statistics collected during sampling for debugging."""

    steps: list[int] = field(default_factory=list)
    t_values: list[float] = field(default_factory=list)
    alpha_bar: list[float] = field(default_factory=list)
    x_min: list[float] = field(default_factory=list)
    x_max: list[float] = field(default_factory=list)
    x_mean: list[float] = field(default_factory=list)
    x_std: list[float] = field(default_factory=list)
    x0_min: list[float] = field(default_factory=list)
    x0_max: list[float] = field(default_factory=list)
    x0_mean: list[float] = field(default_factory=list)
    x0_std: list[float] = field(default_factory=list)
    noise_min: list[float] = field(default_factory=list)
    noise_max: list[float] = field(default_factory=list)
    noise_mean: list[float] = field(default_factory=list)
    noise_std: list[float] = field(default_factory=list)
    intermediates: list[Tensor] = field(default_factory=list)


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) sampler."""

    def __init__(self, model: DiffusionModel, num_steps: int = 100, eta: float = 0.0) -> None:
        self.model = model
        self.num_steps = num_steps
        self.eta = eta
        self.last_stats: SamplingStats | None = None

    def sample(
        self,
        shape: tuple[int, ...],
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
        collect_stats: bool = False,
    ) -> Tensor:
        """Sample from model using DDIM.

        Parameters
        ----------
        shape
            Output shape (B, H, W).
        kind
            Dataset kind indices (B,).
        initial_noise
            Starting noise tensor. If None, samples from N(0,1).
        collect_stats
            If True, collect per-step statistics and intermediate images.

        Returns
        -------
        Sampled tensor in normalized space (B, H, W).
        """
        device = self.model.device
        B = shape[0]
        kind = torch.zeros(B, dtype=torch.long, device=device) if kind is None else kind.to(device)
        x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)

        eps = 1e-3
        timesteps = torch.linspace(1.0 - eps, eps, self.num_steps + 1, device=device)

        stats = SamplingStats() if collect_stats else None

        with torch.no_grad():
            for i in range(self.num_steps):
                t = timesteps[i].expand(B)
                t_prev = timesteps[i + 1].expand(B)
                x = self._step(x, t, t_prev, kind, step_idx=i, stats=stats)

        self.last_stats = stats
        return x

    def _step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        kind: Tensor,
        step_idx: int = 0,
        stats: SamplingStats | None = None,
    ) -> Tensor:
        output = self.model(x, t, kind)
        x0_pred = self.model.predict_x0(x, t, output.noise)
        x0_pred = torch.complex(x0_pred.real.clamp(-6, 6), x0_pred.imag.clamp(-6, 6)) if x0_pred.is_complex() else x0_pred.clamp(-6, 6)

        ab_t = self.model.alpha_bar(t).view(-1, 1, 1)
        ab_prev = self.model.alpha_bar(t_prev).view(-1, 1, 1)

        if stats is not None:
            should_log = step_idx < 5 or step_idx % 25 == 0 or step_idx == self.num_steps - 1
            if should_log:
                x_abs = x.abs() if x.is_complex() else x
                x0_abs = x0_pred.abs() if x0_pred.is_complex() else x0_pred
                noise_abs = output.noise.abs() if output.noise.is_complex() else output.noise

                stats.steps.append(step_idx)
                stats.t_values.append(t[0].item())
                stats.alpha_bar.append(ab_t[0, 0, 0].item())
                stats.x_min.append(x_abs.min().item())
                stats.x_max.append(x_abs.max().item())
                stats.x_mean.append(x_abs.mean().item())
                stats.x_std.append(x_abs.std().item())
                stats.x0_min.append(x0_abs.min().item())
                stats.x0_max.append(x0_abs.max().item())
                stats.x0_mean.append(x0_abs.mean().item())
                stats.x0_std.append(x0_abs.std().item())
                stats.noise_min.append(noise_abs.min().item())
                stats.noise_max.append(noise_abs.max().item())
                stats.noise_mean.append(noise_abs.mean().item())
                stats.noise_std.append(noise_abs.std().item())
                stats.intermediates.append(x0_pred.detach().cpu())

        one_minus_ab_t = (1 - ab_t).clamp(min=1e-8)
        one_minus_ab_prev = (1 - ab_prev).clamp(min=0.0)
        sigma_sq = self.eta**2 * (one_minus_ab_prev / one_minus_ab_t * (1 - ab_t / ab_prev.clamp(min=1e-8)))
        sigma = sigma_sq.clamp(min=0.0).sqrt()

        pred_dir = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt() * output.noise
        x_prev = ab_prev.sqrt() * x0_pred + pred_dir

        if self.eta > 0 and t_prev[0] > 1e-8:
            x_prev = x_prev + sigma * torch.randn_like(x)

        return x_prev


def create_fixed_noise(image_shape: tuple[int, int], seed: int = 42) -> dict[int, Tensor]:
    """Create reproducible noise for each dataset kind with correct dtype."""
    gen = torch.Generator().manual_seed(seed)
    result = {}
    for kind in DatasetKind:
        shape = (1, *image_shape)
        if kind.is_complex:
            noise = torch.complex(
                torch.randn(shape, generator=gen),
                torch.randn(shape, generator=gen),
            )
        else:
            noise = torch.randn(shape, generator=gen)
        result[int(kind)] = noise
    return result
