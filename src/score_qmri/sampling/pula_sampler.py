"""Fixed-noise, fixed-metric MRI pULA in real VE/EDM coordinates.

Reference: Blumenthal, Holliber, Tamir & Uecker, MRM (2026),
doi:10.1002/mrm.70416; arXiv:2512.05791v2, Eqs. 9--12.
Official artifact: https://gitlab.tugraz.at/ibi/mrirecon/papers/dps-pula,
v0.2 (08b6948239e96dbd5867e2ac9b7cd8a9a79a7112), notably
figure_02_comparison/run.sh. The perturb-and-solve construction was checked
against BART v1.0.00, src/iter/italgos.c:eulermaruyama_precond.
This is an independent EDM port, retaining the paper's Euler transition and
physical likelihood, with fixed sigma and explicit stationary-chain collection
instead of its annealed schedule. No paper image-reconstruction defaults or
convergence claims are transferred to a different learned prior.

For model coordinates u = normalize(x), the intended continuous-time target is
    pi_sigma(u|y) proportional to p_sigma(u) exp(-||F(decode(u))-y||²/(2*tau²)),
where p_sigma = p_data * N(0, sigma² I) for an ideal denoiser. The learned
score is (D(u,sigma)-u)/sigma²; a neural score need not be conservative, so
even this target is an idealization. For real kinds it is restricted to the
active real channel. Sigma and tau are SDs PER REAL coordinate, including
complex data. The physical prior smoothing SD is sigma divided by the model's
fixed normalization scale. It is not a diffused likelihood or a denoising-
posterior draw. There is no final denoise, clipping, or projection.

The step is u' = u + h M score_posterior(u) + sqrt(2h) z, Cov(z)=M.
For linear physics, L includes decoding and whitening, Q=L*L+sigma^-2 I,
M=Q^-1. With independent unit-real noises, z=Q^-1(L*eta_y+eta_u/sigma).
Eq. 11 and the official implementation give covariance M; the M^-1 Gaussian
notation printed beside Eq. 9 is inconsistent with that construction. Applying
Q^-1 to image white noise alone would incorrectly give covariance M².

Finite h has ULA bias (no Metropolis correction); finite CG tolerance adds
solve and noise-covariance error. Burn-in does not remove either error, and
thinning does not guarantee independent draws. Mixing must be assessed by
the caller. The forward operator, model weights and normalization must remain
fixed. Nonlinear likelihood transfer uses a supplied fixed positive diagonal
metric, copied before burn-in; no universal nonlinear normal metric is claimed.
Only a single kind/map is sampled; multi-map product priors require a separate
integration outside this reference API.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from ._problem import SamplingProblem
from ._utils import conjugate_gradient, normal_like
from .protocols import OperatorProtocol


@dataclass(frozen=True)
class PULAConfig:
    """Fixed model-coordinate sigma, Euler step and collection settings.

    burn_in counts discarded transitions at the target sigma. num_samples
    counts retained states, separated by thinning transitions. Defaults are
    engineering starting values, not a mixing or stability guarantee.
    cg_rtol must be small enough for the required covariance accuracy.
    """

    sigma: float = 0.01
    step_size: float = 0.1
    burn_in: int = 100
    num_samples: int = 1
    thinning: int = 1
    cg_max_iter: int = 100
    cg_rtol: float = 1e-6

    def __post_init__(self) -> None:
        if not all(isfinite(v) and v > 0 for v in (self.sigma, self.step_size, self.cg_rtol)):
            raise ValueError("sigma, step_size and cg_rtol must be finite and positive")
        counts = (self.burn_in, self.num_samples, self.thinning, self.cg_max_iter)
        if any(isinstance(n, bool) or not isinstance(n, int) for n in counts):
            raise ValueError("Iteration counts must be integers")
        if self.burn_in < 0 or min(counts[1:]) < 1 or self.cg_rtol >= 1:
            raise ValueError("Require nonnegative burn_in, positive collection/CG counts and cg_rtol < 1")


class PULASampler:
    """Euler pULA with a frozen positive metric and matched stochastic forcing."""

    def __init__(self, model: DiffusionModel, cfg: PULAConfig) -> None:
        """
        Parameters
        ----------
        model
            Frozen VE/EDM denoiser. denoise consumes model coordinates.
        cfg
            Fixed noise level, step size, burn-in and collection settings.
        """
        self.model, self.cfg = model, cfg

    def _metric(
        self, problem: SamplingProblem, diagonal: Tensor | None, generator: torch.Generator
    ) -> tuple[Callable[[Tensor], Tensor], Callable[[], Tensor]]:
        """Return M application and a random RHS with covariance M^-1."""
        cfg = self.cfg
        if diagonal is not None:
            if diagonal.is_complex():
                raise ValueError("diagonal must be real, finite and strictly positive")
            frozen = diagonal.detach().to(problem.scale).clone()
            if not frozen.isfinite().all() or (frozen <= 0).any():
                raise ValueError("diagonal must be real, finite and strictly positive")
            shape = (problem.shape[0], 2, *problem.shape[-2:])
            try:
                frozen = torch.broadcast_to(frozen, shape)
            except RuntimeError as exc:
                raise ValueError("diagonal must broadcast to model coordinates (B, 2, H, W)") from exc
            return lambda rhs: frozen * rhs, lambda: problem.noise(generator) / frozen.sqrt()
        if not problem.linear:
            raise ValueError("Nonlinear likelihood transfer requires an explicit frozen positive diagonal metric")
        jacobian, adjoint = problem.linearize(problem.pack(torch.zeros_like(problem.template)))

        def precision(v: Tensor) -> Tensor:
            return problem.constrain(adjoint(jacobian(v))) / problem.noise_std**2 + v / cfg.sigma**2

        def apply(rhs: Tensor) -> Tensor:
            return conjugate_gradient(precision, rhs, cfg.cg_max_iter, cfg.cg_rtol)

        def perturbation() -> Tensor:
            y = problem.measurement
            if y.is_complex():
                epsilon = normal_like(y, generator)
            else:
                epsilon = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
            return problem.constrain(adjoint(epsilon)) / problem.noise_std + problem.noise(generator) / cfg.sigma

        return apply, perturbation

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
        diagonal: Tensor | None = None,
        return_chain: bool = False,
    ) -> tuple[Tensor]:
        """Collect physical states, returning ``(last,)`` or ``(chain,)``.

        Parameters
        ----------
        forward_op
            Fixed, batch-independent tuple-returning differentiable physics.
            A linear operator supplies H. Nonlinear compositions require diagonal.
        measurement
            Physical observations; sample dimension may be singleton or B.
        noise_std
            Physical measurement SD per real coordinate, also for complex noise.
        template
            Shape (B, ..., H, W), singleton nonspatial dimensions; values ignored.
        kind
            Single dataset kind, with matching real/complex template dtype.
        generator
            Explicit random generator on the model device.
        measurement_batch_dim
            Position of the independent sample axis in forward outputs.
        initial
            Optional physical starting state, used without additional re-noising.
            Otherwise linear normal-metric pULA starts at the paper's flat-prior
            Gaussian posterior with model variance sigma². A supplied diagonal
            uses a zero-mean model Gaussian start with that variance instead.
        diagonal
            Optional fixed positive diagonal of M in model coordinates, scalar
            or broadcastable to (B, 2, H, W). An estimated metric must be estimated
            before this call; its detached copy stays fixed through collection.
            This is the labeled fixed-metric nonlinear likelihood adaptation.
        return_chain
            Return (num_samples, B, ..., H, W) instead of the last (B, ..., H, W).
            Retained states are correlated; each follows thinning transitions.
        """
        cfg = self.cfg
        if not isinstance(generator, torch.Generator):
            raise ValueError("An explicit torch.Generator is required")
        if not self.model.sigma_min <= cfg.sigma <= self.model.sigma_max:
            raise ValueError("sigma must be within the model's supported range")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        if not problem.measurement.isfinite().all():
            raise ValueError("measurement must be finite")
        apply, perturbation = self._metric(problem, diagonal, generator)
        if initial is not None:
            x = problem.start(0.0, generator, torch.zeros_like(problem.template), initial)
        elif diagonal is None:
            _, adjoint = problem.linearize(problem.pack(torch.zeros_like(problem.template)))
            x = apply(problem.constrain(adjoint(problem.measurement)) / noise_std**2 + perturbation())
        else:
            x = problem.start(cfg.sigma, generator, None)
        sigma = x.new_full((x.shape[0],), cfg.sigma)
        samples: list[Tensor] = []
        for i in range(cfg.burn_in + cfg.num_samples * cfg.thinning):
            denoised = self.model.denoise(x, sigma, problem.kind)
            score = problem.constrain((denoised - x) / cfg.sigma**2) - problem.gradient(x)
            x = problem.constrain(x + apply(cfg.step_size * score + sqrt(2 * cfg.step_size) * perturbation()))
            if not x.isfinite().all():
                raise RuntimeError("Nonfinite pULA trajectory; reduce step_size and check likelihood/model units")
            if return_chain and i >= cfg.burn_in and (i - cfg.burn_in + 1) % cfg.thinning == 0:
                samples.append(problem.decode(x).detach().clone())
        return (torch.stack(samples) if return_chain else problem.decode(x).detach(),)
