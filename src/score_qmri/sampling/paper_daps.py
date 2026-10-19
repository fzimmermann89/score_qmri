"""Paper-form DAPS with an EDM PF-ODE and unit-temperature Langevin updates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from .daps_sampler import to_2ch, to_complex
from .protocols import LinearOperatorProtocol


def reference_sigma_grid(steps: int, minimum: float, maximum: float, rho: float, device: torch.device) -> Tensor:
    """Reference scheduler: steps+1 polynomial points, drop minimum, append zero."""
    ramp = torch.linspace(0, 1, steps + 1, dtype=torch.float64, device=device)
    positive = (maximum ** (1 / rho) + ramp * (minimum ** (1 / rho) - maximum ** (1 / rho))) ** rho
    return torch.cat([positive[:-1].float(), positive.new_zeros(1).float()])


@dataclass
class PaperDAPSConfig:
    """Gaussian-likelihood DAPS settings from equations 6/7 and Algorithm 1.

    tau is measurement-noise standard deviation per real coordinate. The
    reference code's extra factor two in its likelihood gradient is corrected.
    ``gaussian`` uses the paper's PF-centered r=sigma conditional approximation;
    ``score`` uses the paper's alternative dynamic low-noise prior score.
    Neither finite Langevin nor a learned score makes the ideal theorem exact.
    """

    tau: float
    anneal_steps: int = 200
    reverse_steps: int = 5
    sigma_start: float | None = None
    sigma_min: float = 0.1
    sigma_x0: float = 0.01
    rho: float = 7.0
    langevin_steps: int = 100
    langevin_lr: float = 1.5e-5
    lr_min_ratio: float = 0.01
    prior: Literal["gaussian", "score"] = "gaussian"
    prior_sigma: float | None = None

    def __post_init__(self) -> None:
        values = (self.tau, self.sigma_min, self.sigma_x0, self.rho, self.langevin_lr, self.lr_min_ratio)
        if not all(isfinite(x) and x > 0 for x in values) or self.lr_min_ratio > 1:
            raise ValueError("DAPS settings must be finite and positive; lr_min_ratio must be <=1")
        if self.anneal_steps < 2 or self.reverse_steps < 1 or self.langevin_steps < 1:
            raise ValueError("DAPS needs >=2 outer steps and positive reverse/inner step counts")
        if self.sigma_x0 > self.sigma_min or self.prior not in ("gaussian", "score"):
            raise ValueError("Require sigma_x0 <= sigma_min and a valid prior approximation")


class PaperDAPSSampler:
    """DAPS paper update without extra clipping, tempering or hidden step caps."""

    def __init__(self, model: DiffusionModel, cfg: PaperDAPSConfig) -> None:
        self.model, self.cfg = model, cfg
        self.last_sigmas: Tensor | None = None
        self.last_step_sizes: list[float] = []

    @torch.no_grad()
    def _estimate(self, x: Tensor, sigma: float, kind: Tensor) -> Tensor:
        cfg = self.cfg
        ramp = torch.linspace(0, 1, cfg.reverse_steps + 1, device=x.device, dtype=torch.float64)
        times = (sigma ** (1 / cfg.rho) + ramp * (cfg.sigma_x0 ** (1 / cfg.rho) - sigma ** (1 / cfg.rho))) ** cfg.rho
        for current, following in zip(times[:-1], times[1:], strict=True):
            s = float(current)
            sigma_t = x.new_full((x.shape[0],), s)
            denoised = self.model.denoise(x, sigma_t, kind)
            x = x + float(following - current) * (x - denoised) / s
        return x

    @torch.enable_grad()
    def _conditional(
        self,
        xt: Tensor,
        center: Tensor,
        sigma: float,
        step_size: float,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        cfg = self.cfg
        x = center.detach().clone()
        for _ in range(cfg.langevin_steps):
            x = x.detach().requires_grad_(True)
            residual = forward_op(to_complex(x))[0] - measurement
            energy = residual.abs().square().sum() / (2 * cfg.tau**2)
            (data_gradient,) = torch.autograd.grad(energy, x)
            with torch.no_grad():
                if cfg.prior == "gaussian":
                    prior_score = (center - x) / sigma**2
                else:
                    prior_sigma = self.model.sigma_min if cfg.prior_sigma is None else cfg.prior_sigma
                    sigmas = x.new_full((x.shape[0],), prior_sigma)
                    prior_score = (xt - x) / sigma**2 + self.model.score(x, sigmas, kind)
                epsilon = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                x = x + step_size * (prior_score - data_gradient) + sqrt(2 * step_size) * epsilon
                if not x.isfinite().all():
                    raise RuntimeError("Nonfinite paper-DAPS trajectory; reduce the explicit step size or check operator/noise units")
        return x.detach()

    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw an approximate posterior sample using the paper's full-noise updates."""
        if not is_complex_kind(int(kind)):
            raise NotImplementedError("Paper DAPS currently supports complex image kinds")
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        prior_sigma = self.model.sigma_min if cfg.prior_sigma is None else cfg.prior_sigma
        if not isfinite(start) or not cfg.sigma_min < start <= self.model.sigma_max:
            raise ValueError("Require sigma_min < sigma_start <= model.sigma_max")
        if cfg.sigma_x0 < self.model.sigma_min or not self.model.sigma_min <= prior_sigma <= self.model.sigma_max:
            raise ValueError("DAPS score/ODE evaluations must stay within the model's supported sigma range")
        measurement = measurement.to(self.model.device)
        template = forward_op.H(measurement)[0]
        batch, _, _, height, width = template.shape
        kinds = torch.full((batch,), int(kind), device=template.device, dtype=torch.long)
        sigmas = reference_sigma_grid(cfg.anneal_steps, cfg.sigma_min, start, cfg.rho, template.device)
        self.last_sigmas = sigmas.detach().clone()
        self.last_step_sizes = []
        shape = (batch, 2, height, width)
        epsilon = (
            torch.randn(shape, dtype=template.real.dtype, device=template.device, generator=generator)
            if noise is None
            else to_2ch(noise.to(template.device))
        )
        if epsilon.shape != shape:
            raise ValueError("Initial noise must match the image shape")
        xt = start * epsilon
        for i, (current, following) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            sigma = float(current)
            center = self._estimate(xt, sigma, kinds)
            ratio = i / (cfg.anneal_steps + 1)
            step_size = cfg.langevin_lr * (1 + ratio * (cfg.lr_min_ratio - 1))
            self.last_step_sizes.append(step_size)
            draw = self._conditional(xt, center, sigma, step_size, kinds, forward_op, measurement, generator)
            # Consume the final zero-scaled draw as in the reference loop.
            epsilon = torch.randn(shape, dtype=draw.dtype, device=draw.device, generator=generator)
            xt = draw + float(following) * epsilon
        return to_complex(xt)
