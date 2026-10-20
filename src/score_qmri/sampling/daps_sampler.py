"""DAPS with the conditional prior score and preconditioned Langevin sampling."""

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from ._utils import conjugate_gradient, normal_like, to_2ch, to_complex
from .edm_sampler import EDMSampler, karras_sigmas
from .protocols import LinearOperatorProtocol


@dataclass
class DAPSConfig:
    """Likelihood tau is per real coordinate; Langevin step is dimensionless.

    A fixed, positive preconditioner removes stiffness from the data likelihood
    and the x_sigma|x0 Gaussian. It changes mixing, not the conditional target.
    The target uses the prior smoothed at prior_sigma, rather than the exact
    clean-image prior. Finite Langevin integration adds numerical approximation.
    """

    tau: float
    num_steps: int = 32
    sigma_start: float | None = None
    sigma_min: float = 0.01
    prior_sigma: float = 0.01
    reverse_steps: int = 50
    langevin_steps: int = 200
    langevin_step: float = 0.1
    rho: float = 7.0
    cg_max_iter: int = 50
    cg_rtol: float = 1e-4

    def __post_init__(self) -> None:
        values = (self.tau, self.sigma_min, self.prior_sigma, self.langevin_step, self.rho, self.cg_rtol)
        if not all(isfinite(x) and x > 0 for x in values):
            raise ValueError("Noise levels, step size, rho and CG tolerance must be finite and positive")
        if self.num_steps < 2 or self.langevin_steps < 1 or self.reverse_steps < 1 or self.cg_max_iter < 1:
            raise ValueError("Require >=2 outer steps and positive Langevin/CG iterations")


class DAPSSampler:
    """Sample x0|x_sigma,y using the prior score, then independently re-noise.

    The conditional score is score_prior(x0)+(x_sigma-x0)/sigma² minus the
    Gaussian likelihood gradient. The driving noise has covariance
    2*step*preconditioner; the same increment is reused in the Heun corrector.
    The first chain starts at a probability-flow endpoint; subsequent chains
    retain x0 for Gibbs updates. The conditional prior score is evaluated at
    every update.
    """

    def __init__(self, model: DiffusionModel, cfg: DAPSConfig) -> None:
        self.model, self.cfg = model, cfg

    @torch.no_grad()
    def _conditional(
        self,
        xt: Tensor,
        kind: Tensor,
        operator: LinearOperatorProtocol,
        measurement: Tensor,
        sigma: float,
        generator: torch.Generator | None,
        initial: Tensor | None = None,
    ) -> Tensor:
        cfg = self.cfg
        sigmas = xt.real.new_full((xt.shape[0],), sigma)
        noisy = to_2ch(xt)
        if sigma > self.model.sigma_min:
            flow = EDMSampler(
                self.model,
                num_steps=cfg.reverse_steps,
                sigma_max=sigma,
                sigma_min=self.model.sigma_min,
                rho=cfg.rho,
                clamp_denoised=False,
            )
            center = to_complex(flow.sample(noisy.shape, kind=kind, initial_noise=noisy / sigma))
        else:
            center = to_complex(self.model.denoise(noisy, sigmas, kind))
        prior_sigmas = sigmas.new_full(sigmas.shape, cfg.prior_sigma)
        # Estimate score stiffness once at the PF endpoint, then freeze Q for
        # the entire conditional chain. Using prior_sigma^-2 everywhere makes
        # even a smooth Gaussian prior mix thousands of times too slowly.
        with torch.enable_grad():
            point = to_2ch(center).detach().requires_grad_(True)
            probe = torch.empty_like(point).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
            score = self.model.score(point, prior_sigmas, kind)
            (response,) = torch.autograd.grad(score, point, probe)
            curvature = response.square().flatten(1).mean(1).sqrt().clamp_min(self.model.sigma_data**-2)
        precision = (sigma**-2 + curvature).reshape((xt.shape[0],) + (1,) * (xt.ndim - 1))

        def preconditioner_precision(v: Tensor) -> Tensor:
            return precision * v + operator.H(operator(v)[0])[0] / cfg.tau**2

        def conditional_score(x: Tensor) -> Tensor:
            residual = operator(x)[0] - measurement
            prior = to_complex(self.model.score(to_2ch(x), prior_sigmas, kind))
            return prior + (xt - x) / sigma**2 - operator.H(residual)[0] / cfg.tau**2

        x = center if initial is None else initial
        for _ in range(cfg.langevin_steps):
            score = conditional_score(x)
            # Cov(innovation)=Q, hence Cov(Q^-1 innovation)=Q^-1.
            innovation = precision.sqrt() * normal_like(x, generator)
            innovation += operator.H(normal_like(measurement, generator))[0] / cfg.tau
            rhs = cfg.langevin_step * score + sqrt(2 * cfg.langevin_step) * innovation
            proposal = x + conjugate_gradient(preconditioner_precision, rhs, cfg.cg_max_iter, cfg.cg_rtol)
            # Stochastic Heun reuses the same noise; fresh noise here would
            # change the temperature. The correction reduces finite-step bias.
            correction = 0.5 * cfg.langevin_step * (conditional_score(proposal) - score)
            x = proposal + conjugate_gradient(preconditioner_precision, correction, cfg.cg_max_iter, cfg.cg_rtol)
            if not x.isfinite().all():
                raise RuntimeError("Nonfinite DAPS trajectory; reduce langevin_step and increase langevin_steps")
        return x

    @torch.no_grad()
    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw a complex image; all sampling noise uses the supplied generator."""
        if not is_complex_kind(int(kind)):
            raise NotImplementedError("DAPS currently supports complex image kinds")
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not self.model.sigma_min <= cfg.sigma_min < start <= self.model.sigma_max:
            raise ValueError("DAPS schedule must stay within the model's supported sigma range")
        if not self.model.sigma_min <= cfg.prior_sigma <= self.model.sigma_max:
            raise ValueError("Prior score sigma must be within the model's supported range")
        measurement = measurement.to(self.model.device)
        template = forward_op.H(measurement)[0]
        kinds = torch.full((template.shape[0],), int(kind), device=template.device, dtype=torch.long)
        epsilon = normal_like(template, generator) if noise is None else noise.to(template.device)
        if epsilon.shape != template.shape:
            raise ValueError("Initial noise must match the image shape")
        xt = start * epsilon
        grid = karras_sigmas(cfg.num_steps, sigma_min=cfg.sigma_min, sigma_max=start, rho=cfg.rho, device=xt.device)
        grid = torch.cat([grid, grid.new_zeros(1)])
        x = None
        for sigma, following in zip(grid[:-1], grid[1:], strict=True):
            # Re-noising and a conditional invariant kernel form a Gibbs step.
            # Retain x0 so a finite inner chain preserves its conditioning.
            x = self._conditional(xt, kinds, forward_op, measurement, float(sigma), generator, initial=x)
            xt = x + float(following) * normal_like(x, generator)
        return xt
