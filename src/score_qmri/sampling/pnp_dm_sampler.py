"""PnP-DM comparison sampler with stochastic split-Gibbs conditionals.

Reference: Wu et al., *Principled Probabilistic Imaging using Diffusion Models
as Plug-and-Play Priors*, NeurIPS 2024, arXiv:2405.18782v2, Eqs. 4, 7, 26
and Algorithm 2: https://arxiv.org/html/2405.18782v2 . Official source:
https://github.com/zihuiwu/PnP-DM-public/tree/9ae885c74c8d0b1dc099c59ff08a8dd3282b3121
(pnpdm/samplers/pnp_edm/{pnp_edm,denoiser_edm}.py, SDE branch).

In MODEL coordinates, the ideal fixed-rho joint target is
``pi(x,z|y) proportional to p(x) L(y|decode(z)) N(z;x,rho² I)``.
We return the prior variable x, whose marginal is p(x) times the Gaussian
convolution of the likelihood. The auxiliary marginal is the likelihood times
the Gaussian-smoothed prior. Neither is the unsplit posterior at finite rho.

The prior block uses Euler-Maruyama in VE variance time t=sigma², starting
exactly at z and sigma=rho, without re-noising or likelihood guidance. Unlike
the released code's grid snapping and omitted terminal noise, we include rho
as the exact start and take a stochastic final Euler step to zero. No denoiser
is evaluated below its supported sigma_min. Finite reverse steps (including
the terminal interval), learned scores, finite equilibration and CG tolerance
remain approximations. General nonlinear likelihoods use finite unadjusted
Langevin, as in Algorithm 2; they are not exact conditional draws.

This is the diffusion comparison method, separate from dictionary or
missing-k-space Gibbs samplers. A caller may compose a differentiable signal
model with a tuple-returning acquisition operator. This API samples one kind;
joint priors over multiple physical maps require a separate integration.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from ._problem import SamplingProblem
from ._utils import conjugate_gradient
from .protocols import OperatorProtocol


@dataclass
class PnPDMConfig:
    """Split coupling, reverse SDE and likelihood solver settings.

    rho and rho_start are per-real-coordinate SDs in normalized MODEL units;
    their physical SD is divided by the model's fixed kind scale. rho is held
    fixed during burn_in and sample collection. Optional geometric annealing
    from rho_start to rho precedes burn_in; none of those states are retained.
    burn_in is an equilibration budget, not a convergence guarantee. thin
    counts Gibbs transitions between retained states, which remain correlated.

    reverse_steps counts positive noise evaluations, with reverse_sigma_min
    defaulting to the checkpoint minimum. reverse_schedule_power is the
    Karras grid exponent, unrelated to coupling rho. The last interval to zero
    is stochastic and has finite discretization error.

    likelihood="auto" selects Gaussian perturb-and-solve for operators with H,
    otherwise finite_langevin. linear_gaussian requires a true linear operator
    and its real-inner-product adjoint. CG must converge to cg_rtol or raises.
    finite_langevin starts at x for each likelihood block; langevin_step is in
    squared MODEL units and must be small enough for the physical likelihood.
    No Metropolis correction or nonlinear local-Gaussian solve is used.
    """

    rho: float = 0.05
    rho_start: float | None = None
    anneal_steps: int = 0
    burn_in: int = 50
    num_samples: int = 1
    thin: int = 1
    reverse_steps: int = 128
    reverse_sigma_min: float | None = None
    reverse_schedule_power: float = 7.0
    likelihood: Literal["auto", "linear_gaussian", "finite_langevin"] = "auto"
    cg_max_iter: int = 100
    cg_rtol: float = 1e-5
    langevin_steps: int = 100
    langevin_step: float = 1e-5

    def __post_init__(self) -> None:
        positive = (self.rho, self.reverse_schedule_power, self.cg_rtol, self.langevin_step)
        optional = (self.rho_start, self.reverse_sigma_min)
        if not all(isfinite(v) and v > 0 for v in (*positive, *(v for v in optional if v is not None))):
            raise ValueError("Coupling, noise levels, schedule power and solver tolerances must be finite and positive")
        if self.cg_rtol >= 1:
            raise ValueError("CG relative tolerance must be in (0, 1)")
        counts = (self.num_samples, self.thin, self.reverse_steps, self.cg_max_iter, self.langevin_steps)
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in counts) or self.reverse_steps < 2:
            raise ValueError("Require positive iteration counts and >=2 reverse steps")
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (self.anneal_steps, self.burn_in)):
            raise ValueError("anneal_steps and burn_in must be nonnegative integers")
        if self.rho_start is not None and (self.rho_start < self.rho or (self.rho_start != self.rho and self.anneal_steps == 0)):
            raise ValueError("rho_start must be >= rho, with positive anneal_steps when different")
        if self.likelihood not in ("auto", "linear_gaussian", "finite_langevin"):
            raise ValueError("likelihood must be auto, linear_gaussian or finite_langevin")


class PnPDMSampler:
    """Approximate PnP-DM split Gibbs with an unguided stochastic prior block."""

    def __init__(self, model: DiffusionModel, cfg: PnPDMConfig) -> None:
        """
        Parameters
        ----------
        model
            VE/EDM model with fixed public normalization and main denoiser.
            Use frozen evaluation weights, as for the other reference samplers.
        cfg
            Coupling, collection and numerical approximation controls.
        """
        self.model, self.cfg = model, cfg

    @torch.no_grad()
    def _likelihood(self, x: Tensor, problem: SamplingProblem, rho: float, generator: torch.Generator) -> Tensor:
        cfg = self.cfg
        if cfg.likelihood == "linear_gaussian" or (cfg.likelihood == "auto" and problem.linear):
            if not problem.linear:
                raise ValueError("linear_gaussian requires a linear operator with an adjoint H")
            forward, adjoint = problem.linearize(x)

            def precision(v: Tensor) -> Tensor:
                return problem.constrain(v / rho**2 + adjoint(forward(v)) / problem.noise_std**2)

            measurement_noise = torch.randn(problem.measurement.shape, device=x.device, dtype=x.dtype, generator=generator)
            if problem.measurement.is_complex():
                imaginary = torch.randn(problem.measurement.shape, device=x.device, dtype=x.dtype, generator=generator)
                measurement_noise = torch.complex(measurement_noise, imaginary)
            # The random RHS has covariance Q, hence solving Q z = rhs gives Q^-1,
            # rather than Q^-2 from solving against white image noise alone.
            rhs = x / rho**2 + adjoint(problem.measurement) / problem.noise_std**2
            rhs = problem.constrain(rhs + problem.noise(generator) / rho + adjoint(measurement_noise) / problem.noise_std)
            return conjugate_gradient(precision, rhs, cfg.cg_max_iter, cfg.cg_rtol)

        z = x.clone()
        for _ in range(cfg.langevin_steps):
            gradient = problem.gradient(z) + (z - x) / rho**2
            z = problem.constrain(z - cfg.langevin_step * gradient + sqrt(2 * cfg.langevin_step) * problem.noise(generator))
            if not z.isfinite().all():
                raise RuntimeError("Nonfinite finite-Langevin likelihood trajectory; reduce langevin_step")
        return z

    def _reverse_grid(self, rho: float, device: torch.device) -> Tensor:
        cfg = self.cfg
        minimum = self.model.sigma_min if cfg.reverse_sigma_min is None else cfg.reverse_sigma_min
        # Float64 and explicit endpoints keep evaluations inside checkpoint support.
        ramp = torch.linspace(0, 1, cfg.reverse_steps, device=device, dtype=torch.float64)
        grid = rho ** (1 / cfg.reverse_schedule_power) * (1 - ramp) + minimum ** (1 / cfg.reverse_schedule_power) * ramp
        grid = grid**cfg.reverse_schedule_power
        grid[0], grid[-1] = rho, minimum
        return torch.cat([grid, grid.new_zeros(1)])

    @torch.no_grad()
    def _prior(self, observation: Tensor, problem: SamplingProblem, rho: float, generator: torch.Generator) -> Tensor:
        x = observation.clone()
        grid = self._reverse_grid(rho, x.device)
        for current, following in zip(grid[:-1].tolist(), grid[1:].tolist(), strict=True):
            denoised = self.model.denoise(x, x.new_full((len(x),), current), problem.kind)
            variance = current**2 - following**2
            x = problem.constrain(x + variance / current**2 * (denoised - x) + sqrt(variance) * problem.noise(generator))
            if not x.isfinite().all():
                raise RuntimeError("Nonfinite PnP-DM reverse SDE; check the score and reverse discretization")
        return x

    @torch.no_grad()
    def sample(
        self,
        forward_op: OperatorProtocol,
        measurement: Tensor,
        *,
        noise_std: float,
        template: Tensor,
        kind: DatasetKind | int,
        generator: torch.Generator,
        measurement_batch_dim: int = 0,
        initial: Tensor | None = None,
    ) -> tuple[Tensor]:
        """Return ``(draws,)`` in physical units, shaped (num_samples, *template.shape).

        template specifies (B, ..., H, W) with singleton nonspatial dimensions;
        its values are ignored. B indexes independent chains, while the leading
        output axis holds correlated retained draws. initial optionally supplies
        a physical x state; otherwise x starts at zero. No re-noising is applied.

        noise_std is the physical Gaussian SD PER REAL coordinate, including
        each real/imaginary component of complex measurements. Physics acts on
        decoded physical values; coupling and reverse noise use model units.
        measurement_batch_dim identifies B in tuple-returning forward outputs.
        The forward model must not mix chains. Use centered orthonormal Fourier
        operators when composing MRI acquisitions. The supplied RNG controls
        every Gaussian perturbation, inner Langevin and reverse-SDE increment.
        """
        cfg = self.cfg
        minimum = self.model.sigma_min if cfg.reverse_sigma_min is None else cfg.reverse_sigma_min
        start = cfg.rho if cfg.rho_start is None else cfg.rho_start
        if not self.model.sigma_min <= minimum < cfg.rho <= start <= self.model.sigma_max:
            raise ValueError("Require model.sigma_min <= reverse_sigma_min < rho <= rho_start <= model.sigma_max")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        if cfg.likelihood == "linear_gaussian" and not problem.linear:
            raise ValueError("linear_gaussian requires a linear operator with an adjoint H")
        if initial is None:
            x = problem.encode(torch.zeros_like(problem.template))
        else:
            if initial.shape != template.shape or initial.is_complex() != problem.complex or not initial.isfinite().all():
                raise ValueError("initial must match template shape and real/complex kind and be finite")
            x = problem.encode(initial.to(problem.template)).detach()

        for i in range(cfg.anneal_steps):
            ratio = i / max(cfg.anneal_steps - 1, 1)
            rho = start * (cfg.rho / start) ** ratio
            z = self._likelihood(x, problem, rho, generator)
            x = self._prior(z, problem, rho, generator)
        retained = []
        for i in range(cfg.burn_in + cfg.num_samples * cfg.thin):
            z = self._likelihood(x, problem, cfg.rho, generator)
            x = self._prior(z, problem, cfg.rho, generator)
            if i >= cfg.burn_in and (i - cfg.burn_in + 1) % cfg.thin == 0:
                retained.append(problem.decode(x).detach())
        return (torch.stack(retained),)
