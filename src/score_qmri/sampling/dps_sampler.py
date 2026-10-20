"""Ancestral diffusion posterior sampling with a noise-aware Gaussian likelihood."""

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from ._utils import conjugate_gradient, to_2ch, to_complex
from .edm_sampler import karras_sigmas
from .protocols import LinearOperatorProtocol


@dataclass
class DPSConfig:
    """tau is measurement standard deviation per real coordinate.

    The likelihood integrates an isotropic approximation to x0|x_sigma. Its
    variance is estimated from the denoiser Jacobian in the residual direction.
    A nonpositive estimate falls back to sigma²*sigma_data²/(sigma²+sigma_data²).
    This Gaussian closure is exact for an isotropic Gaussian prior. For learned image priors
    it is an approximation beyond the original DPS plug-in likelihood.
    """

    tau: float
    num_steps: int = 1000
    sigma_start: float | None = None
    rho: float = 7.0
    cg_max_iter: int = 50
    cg_rtol: float = 1e-4

    def __post_init__(self) -> None:
        if self.num_steps < 2 or self.cg_max_iter < 1:
            raise ValueError("Require >=2 diffusion steps and positive CG iterations")
        if not all(isfinite(x) and x > 0 for x in (self.tau, self.rho, self.cg_rtol)):
            raise ValueError("tau, rho and CG tolerance must be finite and positive")


class DPSSampler:
    """Denoiser-Jacobian likelihood guidance with the ancestral VE/EDM bridge.

    Guidance is integrated with the diffusion variance increment, not a fixed
    step at every discretization point. The Gaussian conditional approximation
    remains an approximation; there is no clipping, projection or noise tempering.
    """

    def __init__(self, model: DiffusionModel, cfg: DPSConfig) -> None:
        self.model, self.cfg = model, cfg

    @torch.enable_grad()
    def _step(
        self, x: Tensor, kind: Tensor, operator: LinearOperatorProtocol, measurement: Tensor, sigma: float, sigma_next: float, noise: Tensor
    ) -> Tensor:
        cfg = self.cfg
        x = x.detach().requires_grad_(True)
        denoised = self.model.denoise(x, x.new_full((x.shape[0],), sigma), kind)
        residual = (operator(to_complex(denoised))[0] - measurement).detach()
        direction = to_2ch(operator.H(residual)[0])
        (response,) = torch.autograd.grad(denoised, x, direction, retain_graph=True)
        # Tweedie's covariance identity: Cov(x0|x_sigma) = sigma² * J_D.
        # Estimate its scale in the data-residual direction. The checkpoint's
        # isotropic Gaussian closure supplies a fallback for invalid estimates.
        # A positive estimate can be smaller than this fallback; flooring it
        # would bias even an exact Gaussian prior with smaller variance.
        squared = direction.square().flatten(1).sum(1)
        rayleigh = (direction * response).flatten(1).sum(1) / squared.clamp_min(torch.finfo(x.dtype).tiny)
        baseline = self.model.sigma_data**2 / (sigma**2 + self.model.sigma_data**2)
        variance = sigma**2 * torch.where(rayleigh > 0, rayleigh, baseline).detach()
        variance = variance.reshape((x.shape[0],) + (1,) * (measurement.ndim - 1))

        def covariance(v: Tensor) -> Tensor:
            return cfg.tau**2 * v + variance * operator(operator.H(v)[0])[0]

        weighted = conjugate_gradient(covariance, residual, cfg.cg_max_iter, cfg.cg_rtol)
        (gradient,) = torch.autograd.grad(denoised, x, to_2ch(operator.H(weighted)[0]))
        with torch.no_grad():
            ratio = (sigma_next / sigma) ** 2
            delta = sigma**2 - sigma_next**2
            mean = ratio * x + (1 - ratio) * denoised - delta * gradient
            result = mean + sqrt(sigma_next**2 * (1 - ratio)) * noise
            if not result.isfinite().all():
                raise RuntimeError("Nonfinite DPS trajectory; check likelihood units and the operator")
        return result

    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw a complex image conditional on a fixed measurement."""
        if not is_complex_kind(int(kind)):
            raise NotImplementedError("DPS currently supports complex image kinds")
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not self.model.sigma_min < start <= self.model.sigma_max:
            raise ValueError("Starting sigma must be within the model's supported range")
        measurement = measurement.to(self.model.device)
        template = forward_op.H(measurement)[0]
        shape = (template.shape[0], 2, *template.shape[-2:])
        kinds = torch.full((shape[0],), int(kind), device=template.device, dtype=torch.long)
        x = (
            torch.randn(shape, device=template.device, dtype=template.real.dtype, generator=generator)
            if noise is None
            else to_2ch(noise.to(template.device))
        )
        if x.shape != shape:
            raise ValueError("Initial noise must match the image shape")
        x = start * x
        grid = karras_sigmas(cfg.num_steps, sigma_min=self.model.sigma_min, sigma_max=start, rho=cfg.rho, device=x.device)
        grid = torch.cat([grid, grid.new_zeros(1)])
        for sigma, following in zip(grid[:-1], grid[1:], strict=True):
            epsilon = torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
            x = self._step(x, kinds, forward_op, measurement, float(sigma), float(following), epsilon)
        return to_complex(x.detach())
