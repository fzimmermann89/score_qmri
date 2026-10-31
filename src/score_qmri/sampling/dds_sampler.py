"""Noisy DDS reconstruction with a VE/EDM prior and optional Gauss–Newton DC.

Reference: Chung, Lee & Ye, ICLR 2024, arXiv:2303.05754v3, Eq. (26),
Algorithm 3 and Appendix B Eq. (66): https://arxiv.org/html/2303.05754v3.
Official source: https://github.com/hyungjin-chung/DDS/blob/
509b140ee8ba76772ab7d98929270033d4306f2f/solver_2d.py (noisy multicoil branch).

For model coordinates u, denoised center d, physical decoder S and Gaussian
SD tau per real coordinate, the data update approximately minimizes

    Q(u; d) = ||u-d||²/2 + gamma * ||F(S(u))-y||²/(2*tau²).

This is Eq. (26) with whitened physics L=F o S / tau, b=y/tau; the paper's
coefficient on the unwhitened residual is gamma/tau². gamma is a tunable
proximal strength, independent of tau and diffusion sigma. It is not a
denoising-posterior variance. Linear DDS uses a short CG solve starting at d.

The released VP update becomes, under u=x_VP/sqrt(alpha_bar) and
sigma=sqrt((1-alpha_bar)/alpha_bar),

    epsilon_hat = (u-d)/sigma
    u_next = u_dc + sigma_next*(sqrt(1-eta²)*epsilon_hat + eta*z).

This uses beta_tilde=1 in Eq. (66), matching the released code's coefficients.
The Karras grid replaces its VP time grid; the last evaluation only denoises,
as in the reference endpoint. This is an EDM schedule adaptation, not a
reproduction of its VP checkpoint, geometric VE grid or rejection heuristic.

``data_consistency='gauss_newton'`` is a nonlinear proximal adaptation: CG
solves a local Jacobian system, with backtracking on Q. It does not solve the
nonlinear physics by linear CG. The caller supplies the differentiable
signal/acquisition composition; this single-kind API does not supply joint
priors or coordinate transforms for multiple unknown parameter maps.

Outputs are randomized reconstructions. Neither finite CG nor noise mixing
establishes calibrated posterior draws; the deterministic final denoiser is
not a stochastic denoising-posterior conditional draw.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from ._problem import SamplingProblem
from .edm_sampler import karras_sigmas
from .protocols import OperatorProtocol


@dataclass
class DDSConfig:
    """DDS schedule, truncated CG budget and nonlinear adaptation settings.

    num_steps counts denoiser calls, including the final denoise-only call.
    eta controls the released-code noise mixing, in [0, 1]. gamma weights the
    physical Gaussian negative log likelihood relative to the model-coordinate
    tether; it is fixed across sigma levels. Defaults are starting settings,
    not tuned parameters for this checkpoint. cg_rtol is relative to the
    initial residual of each independent solve. Exhausting cg_max_iter is
    normal for DDS and is reported without requiring convergence.

    gauss_newton_steps and backtracking_steps apply only to the explicitly
    selected nonlinear adaptation. The denoised tether stays fixed during
    these iterations. A failed line search retains the current point.
    """

    num_steps: int = 50
    sigma_start: float | None = None
    rho: float = 7.0
    eta: float = 0.5
    gamma: float = 1.0
    cg_max_iter: int = 5
    cg_rtol: float = 1e-6
    data_consistency: Literal["linear", "gauss_newton"] = "linear"
    gauss_newton_steps: int = 3
    backtracking_steps: int = 12

    def __post_init__(self) -> None:
        if self.num_steps < 2 or min(self.cg_max_iter, self.gauss_newton_steps, self.backtracking_steps) < 1:
            raise ValueError("Require >=2 denoiser calls and positive CG/Gauss–Newton/backtracking budgets")
        if not all(isfinite(v) and v > 0 for v in (self.rho, self.cg_rtol)) or self.cg_rtol >= 1:
            raise ValueError("Require finite positive rho and CG tolerance in (0, 1)")
        if not isfinite(self.eta) or not 0 <= self.eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
        if not isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if self.sigma_start is not None and (not isfinite(self.sigma_start) or self.sigma_start <= 0):
            raise ValueError("sigma_start must be finite and positive")
        if self.data_consistency not in ("linear", "gauss_newton"):
            raise ValueError("data_consistency must be linear or gauss_newton")


@dataclass(frozen=True)
class DDSCGDiagnostics:
    """Per-sample CG iterations, true residual norms and convergence flags.

    All tensors have shape (B,). relative_residual uses the initial residual
    norm as denominator, with zero assigned to an initially solved system.
    Diagnostics describe the local linear solve, including when used by GN.
    """

    iterations: Tensor
    initial_residual_norm: Tensor
    residual_norm: Tensor
    relative_residual: Tensor
    converged: Tensor


@dataclass(frozen=True)
class DDSStepDiagnostics:
    """One data-consistency update; objectives and accepted steps are per sample.

    objectives use Q from the module documentation. step_sizes has one (B,)
    tensor per solve; zero means the nonlinear line search retained its input.
    No diagnostics are emitted for the final denoise-only call.
    """

    sigma: float
    mode: Literal["linear", "gauss_newton"]
    cg: tuple[DDSCGDiagnostics, ...]
    objective_before: Tensor
    objective_after: Tensor
    step_sizes: tuple[Tensor, ...]


def _squared_norm(x: Tensor) -> Tensor:
    return x.abs().square().flatten(1).sum(1)


@torch.no_grad()
def _truncated_cg(
    matvec: Callable[[Tensor], Tensor], initial: Tensor, rhs: Tensor, max_iter: int, rtol: float
) -> tuple[Tensor, DDSCGDiagnostics]:
    """DDS-local warm-started, independently batched real-coordinate CG.

    The shared solver requires convergence and starts at zero; pinned mrpro CG
    uses one global inner product across samples. Neither preserves DDS's
    independent, truncated Krylov updates. No implicit backward is used here.
    """
    x = initial.detach().clone()
    residual = rhs - matvec(x)
    if not x.isfinite().all() or not residual.isfinite().all():
        raise RuntimeError("Nonfinite DDS CG input or residual; check the physics and units")
    direction = residual.clone()
    squared = _squared_norm(residual)
    initial_squared = squared.clone()
    threshold = rtol**2 * initial_squared
    active = squared > threshold
    iterations = torch.zeros(len(x), device=x.device, dtype=torch.long)
    expand = (-1,) + (1,) * (x.ndim - 1)
    for _ in range(max_iter):
        if not active.any():
            break
        product = matvec(direction)
        curvature = (direction * product).flatten(1).sum(1)
        if not product.isfinite().all() or (active & (curvature <= 0)).any():
            raise RuntimeError("DDS CG requires finite positive curvature; check the operator and adjoint")
        alpha = torch.where(active, squared / torch.where(active, curvature, 1), 0)
        x = x + alpha.reshape(expand) * direction
        residual = residual - alpha.reshape(expand) * product
        iterations += active
        following = _squared_norm(residual)
        next_active = active & (following > threshold)
        beta = torch.where(next_active, following / torch.where(active, squared, 1), 0)
        direction = torch.where(next_active.reshape(expand), residual + beta.reshape(expand) * direction, 0)
        squared, active = following, next_active
    error = _squared_norm(rhs - matvec(x))
    if not x.isfinite().all() or not error.isfinite().all():
        raise RuntimeError("Nonfinite DDS CG result; check the physics and units")
    denominator = torch.where(initial_squared > 0, initial_squared, 1)
    return x, DDSCGDiagnostics(iterations, initial_squared.sqrt(), error.sqrt(), (error / denominator).sqrt(), error <= threshold)


class DDSSampler:
    """Noisy linear DDS reconstruction, or explicitly selected Gauss–Newton DDS."""

    def __init__(self, model: DiffusionModel, cfg: DDSConfig) -> None:
        """
        Parameters
        ----------
        model
            VE/EDM model with public physical normalization and denoising APIs.
            The caller should put learned models in evaluation mode.
        cfg
            Schedule, proximal strength and data-consistency approximation.
        """
        self.model, self.cfg = model, cfg

    def _objective(self, x: Tensor, center: Tensor, problem: SamplingProblem) -> Tensor:
        return 0.5 * _squared_norm(x - center) + 0.5 * self.cfg.gamma * _squared_norm(problem.residual(x) / problem.noise_std)

    @torch.no_grad()
    def _data_consistency(self, center: Tensor, problem: SamplingProblem, sigma: float) -> tuple[Tensor, DDSStepDiagnostics]:
        cfg = self.cfg
        if cfg.data_consistency == "linear" and not problem.linear:
            raise ValueError("Linear DDS requires an adjoint; select data_consistency='gauss_newton' for nonlinear physics")
        point = problem.constrain(center.detach())
        before = self._objective(point, center, problem)
        reports: list[DDSCGDiagnostics] = []
        steps: list[Tensor] = []
        for _ in range(1 if cfg.data_consistency == "linear" else cfg.gauss_newton_steps):
            jacobian, adjoint = problem.linearize(point)

            def precision(v: Tensor, jvp: Callable[[Tensor], Tensor] = jacobian, vjp: Callable[[Tensor], Tensor] = adjoint) -> Tensor:
                return v + cfg.gamma * vjp(jvp(v) / problem.noise_std) / problem.noise_std

            # A correction solve from zero is exactly CG warm-started at point.
            # For linear DDS point=d; for GN, the tether remains fixed at d.
            rhs = center - point - cfg.gamma * adjoint(problem.residual(point) / problem.noise_std) / problem.noise_std
            correction, report = _truncated_cg(precision, torch.zeros_like(point), rhs, cfg.cg_max_iter, cfg.cg_rtol)
            reports.append(report)
            if cfg.data_consistency == "linear":
                point = problem.constrain(point + correction)
                steps.append(point.new_ones(len(point)))
                break
            current = self._objective(point, center, problem)
            accepted = torch.zeros(len(point), device=point.device, dtype=torch.bool)
            step = point.new_ones(len(point))
            used_step = torch.zeros_like(step)
            updated = point.clone()
            expand = (-1,) + (1,) * (point.ndim - 1)
            for _ in range(cfg.backtracking_steps):
                candidate = problem.constrain(point + step.reshape(expand) * correction)
                value = self._objective(candidate, center, problem)
                take = ~accepted & value.isfinite() & (value <= current)
                updated = torch.where(take.reshape(expand), candidate, updated)
                used_step = torch.where(take, step, used_step)
                accepted |= take
                if accepted.all():
                    break
                step = step * 0.5
            point = updated
            steps.append(used_step)
        after = self._objective(point, center, problem)
        if not before.isfinite().all() or not after.isfinite().all():
            raise RuntimeError("Nonfinite DDS data-consistency objective; check the physics and units")
        return point, DDSStepDiagnostics(sigma, cfg.data_consistency, tuple(reports), before, after, tuple(steps))

    @staticmethod
    def _transition(x: Tensor, center: Tensor, consistent: Tensor, sigma: float, following: float, eta: float, noise: Tensor) -> Tensor:
        epsilon_hat = (x - center) / sigma
        return consistent + following * (sqrt(1 - eta**2) * epsilon_hat + eta * noise)

    @torch.no_grad()
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
        initial: Tensor | None = None,
        generator: torch.Generator | None = None,
        callback: Callable[[DDSStepDiagnostics], None] | None = None,
    ) -> tuple[Tensor]:
        """Return ``(physical_reconstruction,)`` using one fixed dataset kind.

        Physics accepts physical images/maps and returns a tuple, as in mrpro.
        Linear mode requires its true adjoint ``.H``. Nonlinear mode accepts a
        caller-composed differentiable signal/acquisition operator. The operator
        must act independently on samples. template has shape (B, ..., H, W)
        with singleton nonspatial dimensions; its values are ignored.

        noise_std is the physical Gaussian SD per real coordinate, including
        each component of complex noise. measurement_batch_dim identifies B
        in measurements and forward outputs. Supplied initial noise is unit
        noise in MODEL coordinates (physical layout); initial is an optional
        physical warm start. All random draws use generator when supplied.

        callback receives per-sample true CG residuals and proximal objective
        values, including exhausted budgets. The final operation denoises only;
        it can change data consistency and does not emit a CG report.
        """
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not 0 < self.model.sigma_min < start <= self.model.sigma_max:
            raise ValueError("DDS schedule must stay within the model's supported sigma range")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        if cfg.data_consistency == "linear" and not problem.linear:
            raise ValueError("Linear DDS requires an adjoint; select data_consistency='gauss_newton' for nonlinear physics")
        if not problem.measurement.isfinite().all():
            raise ValueError("measurement must be finite")
        x = problem.start(start, generator, noise, initial)
        grid = karras_sigmas(cfg.num_steps, sigma_min=self.model.sigma_min, sigma_max=start, rho=cfg.rho, device=x.device)
        # Pin endpoints to the public supported bounds despite float32 grid rounding.
        grid = grid.to(dtype=x.dtype).clamp(min=self.model.sigma_min, max=start)
        grid[0], grid[-1] = start, self.model.sigma_min
        for sigma, following in zip(grid[:-1], grid[1:], strict=True):
            current = float(sigma)
            center = problem.constrain(self.model.denoise(x, x.new_full((len(x),), current), problem.kind))
            consistent, report = self._data_consistency(center, problem, current)
            epsilon = problem.noise(generator) if cfg.eta > 0 else torch.zeros_like(x)
            x = problem.constrain(self._transition(x, center, consistent, current, float(following), cfg.eta, epsilon))
            if not x.isfinite().all():
                raise RuntimeError("Nonfinite DDS trajectory; check the physics and units")
            if callback is not None:
                callback(report)
        final = problem.constrain(self.model.denoise(x, x.new_full((len(x),), self.model.sigma_min), problem.kind))
        result = problem.decode(final).detach()
        if not result.isfinite().all():
            raise RuntimeError("Nonfinite DDS final denoiser output")
        return (result,)
