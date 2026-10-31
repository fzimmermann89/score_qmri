"""ΠGDM's noisy linear score and DDIM update in VE/EDM coordinates.

Reference: Song, Vahdat, Mardani & Kautz, ICLR 2023, published paper
https://jankautz.com/publications/PiGDM_ICLR23.pdf, Eqs. 4, 6, 7, 10, 18, 20
and Appendix A.3. Author source inspected: NVlabs/RED-diff, commit
66482f23e242bb31166c3662002d0a6f9f065030, algos/pgdm.py:
https://github.com/NVlabs/RED-diff/blob/66482f23e242bb31166c3662002d0a6f9f065030/algos/pgdm.py
That source uses VP-DDIM and task-specific pseudoinverses/options. This module
implements the paper's noisy linear formula and VE update directly, with a
Karras grid for the existing EDM checkpoint; it does not copy those options.

For model-coordinate state u, denoiser D, physical decoder T and linear
acquisition A, set B = A T. The declared approximation is

    u0 | u_sigma ~ N(D(u_sigma, sigma), r² I),
    r² = sigma² s² / (sigma² + s²),
    C = noise_std² I + r² B B*,
    g = J_D* B* C^-1 (measurement - B D).

The state-space identity acts on active real coordinates: one channel for real
maps and two for complex images; the dummy imaginary map channel stays zero.
Here s defaults to model.sigma_data, rescaling Appendix A.3's unit Gaussian
closure to the checkpoint's model coordinates. It is not an estimated learned
image covariance or the denoiser-Jacobian covariance used by covariance-DPS.
Only g's final VJP differentiates through the full main-head denoiser, including
its EDM skip connection. C's coefficient and solved vector are frozen.

CG approximates the noisy observation-space inverse C^-1 to the configured
relative residual tolerance, failing on nonconvergence. No dense pseudoinverse,
coil normalization, Fourier-mask projector or noiseless equality is substituted.
All adjoints use the real inner product Re(v* w), including complex data;
noise_std is the physical SD of EACH real/imaginary coordinate. The decoder
scales B and hence its propagated covariance; measurement noise stays physical.

The update is paper Eq. 10: DDIM(u, D, sigma, sigma_next, eta) + r² g.
It is NOT the DPS diffusion-variance-increment correction. Finite schedules,
Gaussian closure and the paper's guidance weights do not yield exact posterior
draws, even with an exact Gaussian denoiser. D is a conditional mean, never a
stochastic denoising-posterior draw.

Nonlinear operators require explicit nonlinear_mode="local_linearization".
This extension replaces B by J_(forward o T) at D and uses the actual nonlinear
residual. It freezes that Jacobian/covariance, omitting their derivatives and
the log-determinant derivative. This is a local Gaussian approximation, NOT
the paper's generalized nonlinear pseudoinverse (Eq. 9). Callers compose their
own differentiable signal and acquisition operators. The first reference API
handles one unknown map/image and one DatasetKind; joint multiple-map priors
require a separate integration boundary and are not inferred here.
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
from .edm_sampler import karras_sigmas
from .protocols import OperatorProtocol


@dataclass
class PiGDMConfig:
    """ΠGDM covariance closure, inverse accuracy and VE-DDIM schedule.

    covariance_std is s in the module equations, per real model coordinate;
    None selects model.sigma_data. eta is the DDIM stochasticity in [0, 1].
    cg_dtype changes only the physics solve precision; the operator must
    support it. No full denoiser Jacobian enters CG. nonlinear_mode defaults
    to rejecting operators without a linear adjoint; opt in to the separately
    labeled local approximation for differentiable nonlinear forward models.
    """

    num_steps: int = 100
    sigma_start: float | None = None
    rho: float = 7.0
    eta: float = 0.2
    covariance_std: float | None = None
    cg_max_iter: int = 100
    cg_rtol: float = 1e-5
    cg_dtype: torch.dtype | None = None
    nonlinear_mode: Literal["reject", "local_linearization"] = "reject"

    def __post_init__(self) -> None:
        if self.num_steps < 2 or self.cg_max_iter < 1:
            raise ValueError("Require >=2 diffusion steps and positive CG iterations")
        if not all(isfinite(v) and v > 0 for v in (self.rho, self.cg_rtol)):
            raise ValueError("rho and CG tolerance must be finite and positive")
        if not isfinite(self.eta) or not 0 <= self.eta <= 1:
            raise ValueError("DDIM eta must be finite and in [0, 1]")
        if self.covariance_std is not None and (not isfinite(self.covariance_std) or self.covariance_std <= 0):
            raise ValueError("covariance_std must be finite and positive in model coordinates")
        if self.cg_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("CG dtype must be float32, float64 or None (model precision)")
        if self.nonlinear_mode not in ("reject", "local_linearization"):
            raise ValueError("nonlinear_mode must be reject or local_linearization")


class PiGDMSampler:
    """Noisy linear ΠGDM reference, with an opt-in local nonlinear extension."""

    def __init__(self, model: DiffusionModel, cfg: PiGDMConfig) -> None:
        """
        Parameters
        ----------
        model
            VE/EDM prior; use evaluation mode and the desired checkpoint weights.
            denoise consumes model coordinates and returns its main clean head.
        cfg
            Schedule, scalar covariance closure and inverse approximation.
        """
        self.model, self.cfg = model, cfg
        self.covariance_std = model.sigma_data if cfg.covariance_std is None else cfg.covariance_std
        if not isfinite(self.covariance_std) or self.covariance_std <= 0:
            raise ValueError("The covariance closure requires a finite positive data SD")

    def _variance(self, sigma: float) -> float:
        return sigma**2 * self.covariance_std**2 / (sigma**2 + self.covariance_std**2)

    @torch.enable_grad()
    def _denoise_and_guidance(self, x: Tensor, problem: SamplingProblem, sigma: float) -> tuple[Tensor, Tensor]:
        if not problem.linear and self.cfg.nonlinear_mode != "local_linearization":
            raise ValueError("Nonlinear operators require explicit nonlinear_mode='local_linearization'")
        point = x.detach().requires_grad_(True)
        denoised = self.model.denoise(point, point.new_full((len(point),), sigma), problem.kind)
        center = denoised.detach().to(dtype=self.cfg.cg_dtype or denoised.dtype)
        residual = -problem.residual(center).detach()
        jacobian, adjoint = problem.linearize(center)
        variance = self._variance(sigma)

        def covariance(v: Tensor) -> Tensor:
            return problem.noise_std**2 * v + variance * jacobian(adjoint(v))

        solution = conjugate_gradient(covariance, residual, self.cfg.cg_max_iter, self.cfg.cg_rtol)
        pullback = adjoint(solution).detach().to(denoised)
        (guidance,) = torch.autograd.grad(denoised, point, pullback)
        return denoised.detach(), problem.constrain(guidance).detach()

    @torch.no_grad()
    def _step(self, x: Tensor, problem: SamplingProblem, sigma: float, sigma_next: float, noise: Tensor) -> Tensor:
        denoised, guidance = self._denoise_and_guidance(x, problem, sigma)
        # Paper Eqs. 18/20 in VE coordinates; eta=1 recovers the ancestral bridge.
        noise_std = self.cfg.eta * sigma_next * sqrt(max(0.0, 1 - (sigma_next / sigma) ** 2))
        residual_std = sqrt(max(0.0, sigma_next**2 - noise_std**2))
        result = denoised + residual_std / sigma * (x - denoised) + noise_std * noise + self._variance(sigma) * guidance
        result = problem.constrain(result)
        if not result.isfinite().all():
            raise RuntimeError("Nonfinite ΠGDM trajectory; check the operator, units and covariance approximation")
        return result

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
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor]:
        """Return one approximate physical reconstruction per template sample.

        Parameters
        ----------
        forward_op
            Tuple-returning physical forward operator acting independently per
            sample. Linear operators expose H; nonlinear compositions require
            cfg.nonlinear_mode="local_linearization". No sequence is hardcoded.
        measurement
            Physical observations, real or complex, with explicit sample axis.
            Complex noise must be circular with independent real/imaginary SDs.
        noise_std
            Positive physical Gaussian SD per real coordinate. Correlated noise
            must be whitened in both the supplied operator and measurements.
            Exact zero-noise pseudoinverses are outside this noisy reference.
        template
            Shape (B, ..., H, W), singleton nonspatial dimensions; values ignored.
            Dtype must match the real/complex kind of the unknown.
        kind
            Single DatasetKind for all unknowns, including physical map scaling.
        measurement_batch_dim
            Axis containing B in measurements/forward outputs (e.g. after time).
        noise
            Optional initial unit noise in model coordinates, in template layout.
        generator
            Explicit RNG for initial noise and every DDIM innovation. Supply a
            seeded generator to reproduce trajectories without the global RNG.
        """
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not self.model.sigma_min < start <= self.model.sigma_max:
            raise ValueError("Starting sigma must be within the model's supported range")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        if not problem.linear and cfg.nonlinear_mode != "local_linearization":
            raise ValueError("Nonlinear operators require explicit nonlinear_mode='local_linearization'")
        x = problem.start(start, generator, noise)
        grid = karras_sigmas(cfg.num_steps, sigma_min=self.model.sigma_min, sigma_max=start, rho=cfg.rho, device=x.device)
        grid = torch.cat([grid, grid.new_zeros(1)])
        for sigma, following in zip(grid[:-1], grid[1:], strict=True):
            x = self._step(x, problem, float(sigma), float(following), problem.noise(generator))
        return (problem.decode(x.detach()),)
