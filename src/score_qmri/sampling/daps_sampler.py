"""DAPS with the published probability-flow / Gaussian conditional approximation."""

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from ._problem import SamplingProblem
from .edm_sampler import karras_sigmas
from .protocols import OperatorProtocol


@dataclass
class DAPSConfig:
    """Gaussian DAPS schedule and Langevin settings.

    Langevin step is a step size in squared normalized diffusion units, decayed
    linearly over the outer iterations. Step size and minimum noise levels
    scale the InverseBench defaults from sigma_data=0.5 to sigma_data=0.2.
    The starting noise is limited to the model's supported maximum.
    Step size assumes a normalized operator; stronger operators need smaller steps.
    """

    num_steps: int = 200
    sigma_start: float | None = None
    sigma_min: float = 0.04
    reverse_sigma_min: float = 0.004
    reverse_steps: int = 5
    langevin_steps: int = 100
    langevin_step: float = 1.6e-5
    langevin_step_min_ratio: float = 0.01
    rho: float = 7.0

    def __post_init__(self) -> None:
        values = (self.sigma_min, self.reverse_sigma_min, self.langevin_step, self.rho)
        if not all(isfinite(x) and x > 0 for x in values):
            raise ValueError("Noise levels, step size and rho must be finite and positive")
        if not 0 < self.langevin_step_min_ratio <= 1:
            raise ValueError("Langevin step minimum ratio must be in (0, 1]")
        if self.num_steps < 2 or self.reverse_steps < 2 or self.langevin_steps < 1:
            raise ValueError("Require >=2 outer/PF steps and positive Langevin iterations")


class DAPSSampler:
    """DAPS Gaussian approximation (Zhang et al., Sec. 3.2, Eq. 7).

    Each outer step resets Langevin dynamics to an Euler probability-flow
    endpoint and approximates p(x0|x_sigma) by N(endpoint, sigma² I).
    Schedules include a final Euler step to zero, as in InverseBench.
    This is an approximate posterior sampler; sample spread is not calibrated.
    """

    def __init__(self, model: DiffusionModel, cfg: DAPSConfig) -> None:
        self.model, self.cfg = model, cfg

    @torch.no_grad()
    def _conditional(
        self,
        xt: Tensor,
        problem: SamplingProblem,
        sigma: float,
        generator: torch.Generator | None,
        *,
        ratio: float,
    ) -> Tensor:
        cfg = self.cfg
        grid = karras_sigmas(cfg.reverse_steps, sigma_min=cfg.reverse_sigma_min, sigma_max=sigma, rho=cfg.rho, device=xt.device)
        grid = torch.cat([grid, grid.new_zeros(1)])
        center = xt
        for current, following in zip(grid[:-1], grid[1:], strict=True):
            denoised = self.model.denoise(center, current.expand(len(center)), problem.kind)
            center = center + (following - current) * (center - denoised) / current
        x = center.clone()
        step = cfg.langevin_step * (1 + ratio * (cfg.langevin_step_min_ratio - 1))
        for _ in range(cfg.langevin_steps):
            score = (center - x) / sigma**2 - problem.gradient(x)
            x = x + step * score + sqrt(2 * step) * problem.noise(generator)
        if not x.isfinite().all():
            raise RuntimeError("Nonfinite DAPS trajectory; reduce langevin_step")
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
        measurement_batch_dim: int = 0,
        noise: Tensor | None = None,
        initial: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor]:
        """Draw physical images/maps; see DPSSampler.sample for coordinate conventions.

        initial is an optional physical warm start, independently re-noised.
        """
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        if not isfinite(start) or not self.model.sigma_min <= cfg.reverse_sigma_min < cfg.sigma_min < start <= self.model.sigma_max:
            raise ValueError("DAPS schedule must stay within the model's supported sigma range")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        xt = problem.start(start, generator, noise, initial)
        grid = karras_sigmas(cfg.num_steps, sigma_min=cfg.sigma_min, sigma_max=start, rho=cfg.rho, device=xt.device)
        for i, sigma in enumerate(grid):
            x = self._conditional(xt, problem, float(sigma), generator, ratio=i / cfg.num_steps)
            xt = x if i == cfg.num_steps - 1 else x + float(grid[i + 1]) * problem.noise(generator)
        return (problem.decode(xt),)
