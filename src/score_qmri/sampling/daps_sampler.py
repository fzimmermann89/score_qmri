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

    A fixed, positive spatial preconditioner removes stiffness from the data likelihood
    and the x_sigma|x0 Gaussian. It changes mixing, not the conditional target.
    The target uses the prior smoothed at prior_sigma (default: model.sigma_min).
    Finite Langevin integration adds numerical approximation.
    """

    tau: float
    num_steps: int = 32
    sigma_start: float | None = None
    sigma_min: float = 0.01
    prior_sigma: float | None = None
    reverse_steps: int = 50
    langevin_steps: int = 200
    langevin_step: float = 0.1
    rho: float = 7.0
    cg_max_iter: int = 50
    cg_rtol: float = 1e-4

    def __post_init__(self) -> None:
        values: tuple[float, ...] = (self.tau, self.sigma_min, self.langevin_step, self.rho, self.cg_rtol)
        if self.prior_sigma is not None:
            values += (self.prior_sigma,)
        if not all(isfinite(x) and x > 0 for x in values):
            raise ValueError("Noise levels, step size, rho and CG tolerance must be finite and positive")
        if self.num_steps < 2 or self.langevin_steps < 1 or self.reverse_steps < 1 or self.cg_max_iter < 1:
            raise ValueError("Require >=2 outer steps and positive Langevin/CG iterations")


class DAPSSampler:
    """Sample x0|x_sigma,y using the prior score, then independently re-noise.

    The conditional score is score_prior(x0)+(x_sigma-x0)/sigma² minus the
    Gaussian likelihood gradient. The driving noise has covariance
    2*step*preconditioner; the same increment is reused in the Heun corrector.
    The first chain starts at a supplied image or a probability-flow endpoint;
    subsequent chains retain x0 for Gibbs updates. The conditional prior score
    is evaluated at every update.
    """

    def __init__(self, model: DiffusionModel, cfg: DAPSConfig) -> None:
        self.model, self.cfg = model, cfg
        self.prior_sigma = model.sigma_min if cfg.prior_sigma is None else cfg.prior_sigma

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
        prior_sigmas = sigmas.new_full(sigmas.shape, self.prior_sigma)
        # Estimate local score stiffness once at the PF endpoint, then freeze Q
        # for the entire conditional chain. A global RMS lets stiff background
        # pixels freeze the foreground. Multiple probes stabilize the local RMS.
        with torch.enable_grad():
            point = to_2ch(center).detach().requires_grad_(True)
            score = self.model.score(point, prior_sigmas, kind)
            squared = torch.zeros_like(point)
            for i in range(4):
                probe = torch.empty_like(point).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
                (response,) = torch.autograd.grad(score, point, probe, retain_graph=i < 3)
                squared += response.square()
            curvature = (squared.mean(1) / 4).sqrt().clamp_min(self.model.sigma_data**-2)
        precision = (sigma**-2 + curvature).unsqueeze(1).unsqueeze(1)
        scale = precision.rsqrt()

        def preconditioner_precision(v: Tensor) -> Tensor:
            # Whiten the diagonal before CG: heterogeneous prior stiffness
            # must not make the linear solve ill-conditioned too.
            return v + scale * operator.H(operator(scale * v)[0])[0] / cfg.tau**2

        def conditional_score(x: Tensor) -> Tensor:
            residual = operator(x)[0] - measurement
            prior = to_complex(self.model.score(to_2ch(x), prior_sigmas, kind))
            return prior + (xt - x) / sigma**2 - operator.H(residual)[0] / cfg.tau**2

        x = center if initial is None else initial
        for _ in range(cfg.langevin_steps):
            score = conditional_score(x)
            # B = scale Q scale. Cov(innovation)=B, giving Q^-1 driving noise
            # after the whitened solve and the final multiplication by scale.
            innovation = normal_like(x, generator)
            innovation += scale * operator.H(normal_like(measurement, generator))[0] / cfg.tau
            rhs = cfg.langevin_step * scale * score + sqrt(2 * cfg.langevin_step) * innovation
            proposal = x + scale * conjugate_gradient(preconditioner_precision, rhs, cfg.cg_max_iter, cfg.cg_rtol)
            # Stochastic Heun reuses the same noise; fresh noise here would
            # change the temperature. The correction reduces finite-step bias.
            correction = 0.5 * cfg.langevin_step * scale * (conditional_score(proposal) - score)
            x = proposal + scale * conjugate_gradient(preconditioner_precision, correction, cfg.cg_max_iter, cfg.cg_rtol)
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
        initial: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw a complex image; optionally warm-start x0 from an existing draw.

        An initial image changes the starting distribution, not the conditional
        target. It is independently re-noised before the first Gibbs update.
        Finite chains still require mixing checks, including across initializations.
        """
        if not is_complex_kind(int(kind)):
            raise NotImplementedError("DAPS currently supports complex image kinds")
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not self.model.sigma_min <= cfg.sigma_min < start <= self.model.sigma_max:
            raise ValueError("DAPS schedule must stay within the model's supported sigma range")
        if not self.model.sigma_min <= self.prior_sigma <= self.model.sigma_max:
            raise ValueError("Prior score sigma must be within the model's supported range")
        measurement = measurement.to(self.model.device)
        template = forward_op.H(measurement)[0]
        kinds = torch.full((template.shape[0],), int(kind), device=template.device, dtype=torch.long)
        epsilon = normal_like(template, generator) if noise is None else noise.to(template.device)
        if epsilon.shape != template.shape:
            raise ValueError("Initial noise must match the image shape")
        x = None if initial is None else initial.to(device=template.device, dtype=template.dtype).detach()
        if x is not None and (x.shape != template.shape or not x.isfinite().all()):
            raise ValueError("Initial image must be finite and match the image shape")
        xt = start * epsilon if x is None else x + start * epsilon
        grid = karras_sigmas(cfg.num_steps, sigma_min=cfg.sigma_min, sigma_max=start, rho=cfg.rho, device=xt.device)
        grid = torch.cat([grid, grid.new_zeros(1)])
        for sigma, following in zip(grid[:-1], grid[1:], strict=True):
            # Re-noising and a conditional invariant kernel form a Gibbs step.
            # Retain x0 so a finite inner chain preserves its conditioning.
            x = self._conditional(xt, kinds, forward_op, measurement, float(sigma), generator, initial=x)
            xt = x + float(following) * normal_like(x, generator)
        return xt
