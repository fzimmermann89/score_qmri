"""Covariance-aware and DPS-style guidance on an ancestral VE/EDM bridge."""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from ._problem import SamplingProblem
from ._utils import conjugate_gradient
from .edm_sampler import karras_sigmas
from .protocols import OperatorProtocol


@dataclass
class DPSConfig:
    """Ancestral VE/EDM steps and likelihood guidance settings.

    The likelihood integrates an isotropic approximation to x0|x_sigma. Its
    variance is estimated from the denoiser Jacobian in the residual direction.
    A nonpositive estimate falls back to sigma²*sigma_data²/(sigma²+sigma_data²).
    This Gaussian closure is exact for an isotropic Gaussian prior. For learned image priors
    it is an approximation beyond the original DPS plug-in likelihood.

    guidance="dps" instead applies guidance_scale times the residual-norm
    gradient once per step, following the official DPS conditioning code.
    This remains a VE/EDM adaptation of its VP-DDPM update, not an exact
    posterior sampler. Its tuned strength does not use noise_std.
    """

    num_steps: int = 1000
    sigma_start: float | None = None
    rho: float = 7.0
    cg_max_iter: int = 50
    cg_rtol: float = 1e-4
    guidance: Literal["covariance", "dps"] = "covariance"
    guidance_scale: float = 0.3

    def __post_init__(self) -> None:
        if self.num_steps < 2 or self.cg_max_iter < 1:
            raise ValueError("Require >=2 diffusion steps and positive CG iterations")
        if not all(isfinite(x) and x > 0 for x in (self.rho, self.cg_rtol)):
            raise ValueError("rho and CG tolerance must be finite and positive")
        if self.guidance not in ("covariance", "dps") or not isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("Require covariance/dps guidance and a finite nonnegative guidance_scale")


class DPSSampler:
    """Denoiser-Jacobian likelihood guidance with the ancestral VE/EDM bridge.

    Covariance guidance uses the diffusion variance increment. The optional
    DPS-style baseline uses a fixed per-step residual-norm correction instead.
    Both are approximations, without clipping, projection or noise tempering.
    """

    def __init__(self, model: DiffusionModel, cfg: DPSConfig) -> None:
        self.model, self.cfg = model, cfg

    @torch.enable_grad()
    def _step(self, x: Tensor, problem: SamplingProblem, sigma: float, sigma_next: float, noise: Tensor) -> Tensor:
        cfg = self.cfg
        x = x.detach().requires_grad_(True)
        denoised = self.model.denoise(x, x.new_full((x.shape[0],), sigma), problem.kind)
        if cfg.guidance == "dps":
            residual = problem.residual(denoised)
            residual_norm = torch.linalg.vector_norm(residual.flatten(1), dim=1)
            if problem.linear:
                _, adjoint = problem.linearize(denoised)
                norm = residual_norm.clamp_min(torch.finfo(x.dtype).tiny).reshape((-1,) + (1,) * (residual.ndim - 1))
                pullback = adjoint(residual.detach() / norm.detach())
                (gradient,) = torch.autograd.grad(denoised, x, pullback)
            else:
                (gradient,) = torch.autograd.grad(residual_norm.sum(), x)
            return problem.constrain(self._ancestral(x, denoised, cfg.guidance_scale * gradient, sigma, sigma_next, noise))
        residual = problem.residual(denoised).detach()
        jacobian, adjoint = problem.linearize(denoised.detach())
        direction = adjoint(residual).detach()
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
        variance = variance.reshape((x.shape[0],) + (1,) * (residual.ndim - 1))

        def covariance(v: Tensor) -> Tensor:
            return problem.noise_std**2 * v + variance * jacobian(adjoint(v))

        weighted = conjugate_gradient(covariance, residual, cfg.cg_max_iter, cfg.cg_rtol)
        (gradient,) = torch.autograd.grad(denoised, x, adjoint(weighted).detach())
        return problem.constrain(self._ancestral(x, denoised, (sigma**2 - sigma_next**2) * gradient, sigma, sigma_next, noise))

    @staticmethod
    @torch.no_grad()
    def _ancestral(x: Tensor, denoised: Tensor, correction: Tensor, sigma: float, sigma_next: float, noise: Tensor) -> Tensor:
        ratio = (sigma_next / sigma) ** 2
        mean = ratio * x + (1 - ratio) * denoised - correction
        result = mean + sqrt(sigma_next**2 * (1 - ratio)) * noise
        if not result.isfinite().all():
            raise RuntimeError("Nonfinite DPS trajectory; check likelihood units and the operator")
        return result

    def sample(
        self,
        forward_op: OperatorProtocol,
        measurement: Tensor,
        *,
        noise_std: float,
        template: Tensor,
        kind: DatasetKind | int,
        measurement_batch_dim: int = 0,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor]:
        """Draw one physical image/map per template sample, returning ``(image,)``.

        template has shape (B, ..., H, W) with singleton nonspatial dimensions;
        its values are ignored. Physics uses physical units. noise_std is per
        real coordinate. measurement_batch_dim identifies B in forward outputs
        and measurements (e.g. 1 when inversion time precedes B). The forward
        model must act independently across samples.
        Supplied noise is unit noise in normalized diffusion coordinates.
        """
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not self.model.sigma_min < start <= self.model.sigma_max:
            raise ValueError("Starting sigma must be within the model's supported range")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        x = problem.start(start, generator, noise)
        grid = karras_sigmas(cfg.num_steps, sigma_min=self.model.sigma_min, sigma_max=start, rho=cfg.rho, device=x.device)
        grid = torch.cat([grid, grid.new_zeros(1)])
        for sigma, following in zip(grid[:-1], grid[1:], strict=True):
            epsilon = problem.noise(generator)
            x = self._step(x, problem, float(sigma), float(following), epsilon)
        return (problem.decode(x.detach()),)
