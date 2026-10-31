"""RED-Diff point-mass variational reconstruction with the existing EDM prior.

References checked against primary sources on 2026-10-31:

* Mardani et al., ICLR 2024, arXiv:2305.04391v2, Algorithm 1,
  Proposition 2 and Eqs. (9)--(11): https://arxiv.org/html/2305.04391v2
* Official ``algos/reddiff.py`` and ``_configs/algo/reddiff.yaml`` at
  NVlabs/RED-diff commit 66482f23e242bb31166c3662002d0a6f9f065030:
  https://github.com/NVlabs/RED-diff/tree/66482f23e242bb31166c3662002d0a6f9f065030
* Karras et al., NeurIPS 2022, arXiv:2206.00364v2, Sec. 2 and Table 1;
  official VPPrecond/EDMPrecond: https://github.com/NVlabs/edm/blob/main/training/networks.py

With paper coefficients ``x_t = alpha_t*mu + sigma_t*epsilon``, set
``s = sigma_t/alpha_t`` and ``z = x_t/alpha_t = mu + s*epsilon``.
The existing preconditioned denoiser consumes z in MODEL coordinates, so
``epsilon_hat = (z - D(z,s))/s``. The paper defines amplitude SNR, alpha_t/sigma_t,
not its square. Its linear inverse-SNR weight is therefore lambda*s, giving
``g_prior = lambda*s*sg[epsilon_hat-epsilon] = lambda*sg[mu-D(z,s)]``.
No denoiser Jacobian, extra alpha, sigma_data factor or EDM training-loss weight
belongs in this update. The official default has zero variational dispersion.

Here the instantaneous surrogate is
``sum(|G(decode(mu))-y|**2)/(2*noise_std**2) + lambda*<sg[mu-D(z,s)],mu>``.
Both terms SUM coordinates and independent batch items. Official code instead
uses separate means and unwhitened half squared residuals; its lambda is not
numerically interchangeable when measurement/image sizes or noise units differ.
This surrogate supplies a gradient; its changing value is not the KL objective
or a global energy to compare across iterations. Linear Gaussian prior controls
have expected prior gradient lambda*s**2*(C+s**2*I)^-1*mu, generally not C^-1*mu.

Declared EDM adaptation: descend a finite Karras grid within checkpoint support
instead of the official uniformly spaced VP timesteps. This is annealed
optimization, not unbiased uniform-time integration of the variational bound.
Use explicit physical initial values or zero, without requiring a pseudoinverse.
General differentiable G, including nonlinear signals composed with acquisition,
uses the same likelihood gradient as the paper; no local linear approximation.
The state remains one kind/map per call. Callers compose known parameters into G;
joint multiple-map priors need a separate integration outside this method.

Outputs are reconstruction endpoints. Repeated randomized optimizations are not
posterior draws and must not enter posterior coverage/spread comparisons.
"""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

from ._problem import SamplingProblem
from .edm_sampler import karras_sigmas
from .protocols import OperatorProtocol


@dataclass
class REDDiffConfig:
    """Annealed reconstruction settings in normalized model coordinates.

    learning_rate is the Adam step size; prior_weight multiplies the summed
    stopped-gradient signal residual. Adam uses the reference betas (0.9, 0.99),
    epsilon 1e-8 and no weight decay. None bounds use checkpoint sigma limits.
    Defaults are reference starting choices, not tuned MRI settings.
    """

    num_steps: int = 1000
    learning_rate: float = 0.1
    prior_weight: float = 0.25
    sigma_start: float | None = None
    sigma_end: float | None = None
    rho: float = 7.0

    def __post_init__(self) -> None:
        if self.num_steps < 2:
            raise ValueError("Require >=2 optimization steps")
        if not all(isfinite(x) and x > 0 for x in (self.learning_rate, self.rho)):
            raise ValueError("learning_rate and rho must be finite and positive")
        if not isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        if any(x is not None and (not isfinite(x) or x <= 0) for x in (self.sigma_start, self.sigma_end)):
            raise ValueError("Sigma bounds must be finite and positive")


class REDDiffSampler:
    """Optimize a Dirac variational mean; ``sample`` returns a reconstruction."""

    def __init__(self, model: DiffusionModel, cfg: REDDiffConfig) -> None:
        """
        Parameters
        ----------
        model
            Existing EDM prior, in evaluation mode for reconstruction.
        cfg
            Optimization and finite noise schedule settings.
        """
        self.model, self.cfg = model, cfg

    def _objective(self, mu: Tensor, problem: SamplingProblem, sigma: float, epsilon: Tensor) -> Tensor:
        """Instantaneous surrogate with fixed epsilon, useful for gradient controls."""
        loss = problem.residual(mu).abs().square().sum() / (2 * problem.noise_std**2)
        if self.cfg.prior_weight:
            with torch.no_grad():
                z = mu + sigma * epsilon
                denoised = self.model.denoise(z, z.new_full((len(z),), sigma), problem.kind)
                # Algebraically lambda*s*(epsilon_hat-epsilon), avoiding cancellation.
                residual = problem.constrain(mu - denoised).detach()
            loss = loss + self.cfg.prior_weight * (residual * mu).sum()
        return loss

    @torch.enable_grad()
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
        """Return ``(reconstruction,)`` in physical coordinates, detached.

        template specifies (B, ..., H, W), with singleton nonspatial dimensions;
        its values are ignored. initial optionally gives a physical warm start,
        otherwise optimization starts at zero. Each step draws fresh unit noise
        on active real coordinates using generator. noise_std is the Gaussian
        measurement SD PER REAL coordinate, also for complex data. The operator
        returns a tensor tuple, accepts physical states, is differentiable and
        acts independently across B. measurement_batch_dim locates B in its output.
        Denoising noise sigma is in model units; its physical SD is decoded with
        the state. The caller provides an evaluation-mode model.
        """
        cfg = self.cfg
        start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        end = self.model.sigma_min if cfg.sigma_end is None else cfg.sigma_end
        if not isfinite(start) or not isfinite(end) or not self.model.sigma_min <= end < start <= self.model.sigma_max:
            raise ValueError("RED-Diff schedule must stay within the model's supported sigma range")
        problem = SamplingProblem(self.model, forward_op, measurement, template, kind, noise_std, measurement_batch_dim)
        zero = torch.zeros_like(problem.template)
        mu = problem.start(0.0, generator, zero, initial).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=cfg.learning_rate, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
        grid = karras_sigmas(cfg.num_steps, sigma_min=end, sigma_max=start, rho=cfg.rho, device=mu.device)
        for sigma in grid:
            optimizer.zero_grad(set_to_none=True)
            loss = self._objective(mu, problem, float(sigma), problem.noise(generator))
            if not loss.isfinite():
                raise RuntimeError("Nonfinite RED-Diff objective; check the signal model and likelihood units")
            # Only the variational mean is optimized; leave model/operator grads untouched.
            (mu.grad,) = torch.autograd.grad(loss, mu)
            if not mu.grad.isfinite().all():
                raise RuntimeError("Nonfinite RED-Diff gradient; check the signal model and likelihood units")
            optimizer.step()
            if not mu.isfinite().all():
                raise RuntimeError("Nonfinite RED-Diff endpoint; reduce learning_rate")
        return (problem.decode(mu.detach()),)
