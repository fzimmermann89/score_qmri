"""DAPS (Decoupled Annealing Posterior Sampling) for MRI reconstruction.

Implements the sampling procedure from Zhang et al., "Improving Diffusion Inverse
Problem Solving with Decoupled Noise Annealing" (arXiv:2407.01521).

Adapted for VE/EDM diffusion framework.

Algorithm 1 alternates:
1) Reverse diffusion: compute an estimator x0_hat(x_t).
   - "pf_ode": integrate PF-ODE from sigma_t to 0 (or sigma_min).
   - "one_step": use model's direct prediction D(x, sigma).

2) MCMC update: sample x0|y ~ p(x0 | x_t, y).
   Prior p(x0|x_t) is approximated as N(x0_hat, sigma_t^2 I).

3) Forward diffusion (re-noising): sample x_{t_next} ~ q(x_{t_next} | x0).
   x_{t_next} = x0 + sigma_{t_next} * eps
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel, _apply_kind_constraints, _create_noise
from score_qmri.sampling.edm_sampler import karras_sigmas

from .protocols import LinearOperatorProtocol

X0Estimator = Literal["pf_ode", "one_step"]
MCMCAlgorithm = Literal["langevin", "hmc", "mh"]


@dataclass
class DAPSConfig:
    """Configuration for DAPS sampling.

    Parameters
    ----------
    num_steps
        Number of annealing steps (N_A in the paper).
    ode_steps
        Number of steps for PF-ODE integration to compute x0_hat (if x0_estimator="pf_ode").
    x0_estimator
        How to compute x0_hat(x_t):
        - "pf_ode": PF-ODE integration (default).
        - "one_step": Single model evaluation (EDM denoise).
    mcmc_steps
        Number of MCMC iterations per annealing step.
    mcmc_algo
        Which MCMC algorithm to use: "langevin", "hmc", or "mh".
    step_size
        Base step size eta_0 for the MCMC sampler.
    delta
        Step size decay ratio (linear schedule).
    beta_y
        Measurement noise std in the Gaussian likelihood.
    hmc_momentum
        Momentum term for HMC.
    """

    num_steps: int = 50

    x0_estimator: X0Estimator = "pf_ode"
    ode_steps: int = 5

    mcmc_steps: int = 5
    mcmc_algo: MCMCAlgorithm = "langevin"

    step_size: float = 1e-4
    delta: float = 0.01

    beta_y: float = 1.0

    hmc_momentum: float = 0.9


def _to_complex(x_2ch: Tensor) -> Tensor:
    """Convert (B, 2, H, W) real to (B, 1, 1, H, W) complex for mrpro/operators."""
    # (B, 2, H, W) -> (B, H, W, 2) -> complex (B, H, W) -> (B, 1, 1, H, W)
    return torch.view_as_complex(torch.moveaxis(x_2ch, 1, -1).contiguous()).unsqueeze(1).unsqueeze(1)


class DAPSSampler:
    """DAPS sampler for MRI reconstruction with learned VE/EDM diffusion prior."""

    def __init__(self, model: DiffusionModel, config: DAPSConfig | None = None) -> None:
        self.model = model
        self.config = config or DAPSConfig()

    @torch.enable_grad()
    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Reconstruct image from k-space measurement using DAPS."""
        cfg = self.config
        device = self.model.device
        kind_idx = int(kind)

        if not is_complex_kind(kind_idx):
            raise ValueError(f"DAPS only supports complex kinds, got {DatasetKind(kind_idx).name}")

        # Infer image shape from adjoint
        x_template = forward_op.H(measurement)[0]
        B, _, _, H, W = x_template.shape

        kind_tensor = torch.full((B,), kind_idx, dtype=torch.long, device=device)

        # Initialize with prior noise (sigma_max)
        if initial_noise is None:
            x = _create_noise((B, 2, H, W), kind_tensor, device, torch.float32)
            x = x * self.model.sigma_max
        else:
            x = initial_noise.to(device)
            if x.ndim == 3:
                x = torch.stack([x, torch.zeros_like(x)], dim=1)
            # Scale provided unit noise by sigma_max (consistent with EDMSampler)
            x = x * self.model.sigma_max

        x = _apply_kind_constraints(x, kind_tensor)

        # Generate Karras sigma schedule
        sigmas = karras_sigmas(
            cfg.num_steps + 1,
            sigma_min=self.model.sigma_min,
            sigma_max=self.model.sigma_max,
            rho=self.model.rho,
            device=device,
        )

        x0_last = x
        for i in range(cfg.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # DAPS Step
            x, x0_last = self._step(x, sigma, sigma_next, kind_tensor, forward_op, measurement, ratio=i / max(cfg.num_steps - 1, 1))

        return _to_complex(x0_last)

    def _estimate_x0(self, x: Tensor, sigma: Tensor, kind: Tensor) -> Tensor:
        cfg = self.config

        if cfg.x0_estimator == "pf_ode":
            return self._estimate_x0_pf_ode(x, sigma, kind)
        elif cfg.x0_estimator == "one_step":
            x0_hat = self.model.denoise(x, sigma, kind)
            return self.model.clamp_x0(x0_hat, kind)
        else:
            raise ValueError(f"Unknown x0_estimator: {cfg.x0_estimator!r}")

    def _estimate_x0_pf_ode(self, x: Tensor, sigma: Tensor, kind: Tensor) -> Tensor:
        """Estimate x0_hat(x_sigma) by integrating the VE probability flow ODE.

        ODE: dx = (x - x0) / sigma * dsigma
        Integrate from sigma down to sigma_min (or 0).
        """
        cfg = self.config

        # Create a mini-schedule from sigma down to sigma_min
        # We'll use a simple linear or karras schedule for the integration
        # Since it's a small number of steps, linear is often fine, but let's stick to karras-like
        # strictly strictly decreasing.

        t0 = sigma[0].item() if sigma.ndim > 0 else sigma.item()

        # If we are already at low noise, just return one step
        if t0 <= self.model.sigma_min:
            return self.model.denoise(x, sigma, kind)

        ode_sigmas = karras_sigmas(
            cfg.ode_steps + 1,
            sigma_min=self.model.sigma_min,
            sigma_max=t0,
            rho=self.model.rho,
            device=x.device,
        )

        x_ode = x
        with torch.no_grad():
            for i in range(cfg.ode_steps):
                s_curr = ode_sigmas[i]
                s_next = ode_sigmas[i + 1]

                # EDM Euler step
                # x_next = x_curr + (s_next - s_curr) * d_curr
                # d_curr = (x_curr - x0) / s_curr

                # Expand sigmas for batch
                s_curr_b = s_curr.expand(x.shape[0])

                x0_curr = self.model.denoise(x_ode, s_curr_b, kind)

                d_curr = (x_ode - x0_curr) / s_curr
                dt = s_next - s_curr

                x_ode = x_ode + dt * d_curr

        return self.model.clamp_x0(x_ode, kind)

    def _get_step_size(self, ratio: float) -> float:
        cfg = self.config
        return cfg.step_size * (cfg.delta + ratio * (1.0 - cfg.delta))

    def _score_langevin(
        self,
        x0: Tensor,
        x0_hat: Tensor,
        r_t: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        beta_y: float,
    ) -> tuple[Tensor, Tensor]:
        """Compute score = grad_{x0} log p(x0 | x_t, y).

        Prior: p(x0 | x_t) approx N(x0_hat, r_t^2 I)
        Likelihood: p(y | x0) approx N(A(x0), beta_y^2 I)
        """
        B = x0.shape[0]

        # Prior score: grad_{x0} (-||x0 - x0_hat||^2 / 2r_t^2)
        # = -(x0 - x0_hat) / r_t^2 = (x0_hat - x0) / r_t^2
        if r_t.ndim == 0:
            r_t = r_t.expand(B)
        r_t_sq = (r_t.view(B, 1, 1, 1).square()).clamp(min=1e-8)

        prior_score = (x0_hat - x0) / r_t_sq

        # Data score: grad_{x0} (-||A(x0)-y||^2 / 2beta_y^2)
        x0_complex = _to_complex(x0)
        y_pred = forward_op(x0_complex)[0]
        residual = y_pred - measurement

        meas_sq = (residual.real**2 + residual.imag**2).reshape(B, -1).sum(dim=1)
        meas_loss = meas_sq.mean()

        data_energy = 0.5 * meas_sq.sum()
        (grad_data,) = torch.autograd.grad(data_energy, x0)

        # grad is w.r.t energy, so score is -grad
        data_score = -(grad_data / (beta_y**2))

        score = prior_score + data_score
        return score, meas_loss.detach()

    def _mh_accept(
        self,
        x_old: Tensor,
        x_new: Tensor,
        x0_hat: Tensor,
        r_t: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        beta_y: float,
    ) -> Tensor:
        """Compute MH accept mask."""
        B = x_old.shape[0]
        if r_t.ndim == 0:
            r_t = r_t.expand(B)
        r_t_sq = (r_t.view(B, 1, 1, 1).square()).clamp(min=1e-8)

        def meas_energy(x: Tensor) -> Tensor:
            x_complex = _to_complex(x)
            y_pred = forward_op(x_complex)[0]
            residual = y_pred - measurement
            sq = (residual.real**2 + residual.imag**2).reshape(B, -1).sum(dim=1)
            return sq

        meas_old = meas_energy(x_old)
        meas_new = meas_energy(x_new)
        # log p(y|x) ratio
        log_data_ratio = (meas_old - meas_new) / (2.0 * beta_y**2)

        # log p(x|x_t) ratio: -||x - x0hat||^2 / 2r_t^2
        prior_old = (x_old - x0_hat).pow(2).reshape(B, -1).sum(dim=1)
        prior_new = (x_new - x0_hat).pow(2).reshape(B, -1).sum(dim=1)

        log_prior_ratio = (prior_old - prior_new) / (2.0 * r_t_sq.flatten())

        log_accept = log_data_ratio + log_prior_ratio
        u = torch.rand_like(log_accept).log()
        return (u < log_accept).view(B, 1, 1, 1)

    def _step(
        self,
        x_t: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        ratio: float,
    ) -> tuple[Tensor, Tensor]:
        """One DAPS iteration: reverse (estimate x0) -> MCMC (sample x0) -> re-noise (sample x_{t-1})."""
        cfg = self.config

        B = x_t.shape[0]
        sigma_b = sigma.expand(B)

        # 1) Reverse diffusion: compute x0_hat(x_t)
        # This is our current best estimate of the clean image
        x0_hat = self._estimate_x0(x_t, sigma_b, kind)

        # 2) MCMC update: sample x0 from p(x0 | x_t, y)
        # We approximate prior p(x0|x_t) as N(x0_hat, sigma^2 I)
        # Scale step size by sigma^2 to normalize gradient magnitude across noise levels
        # (VE score scales as 1/sigma^2, so lr * sigma^2 keeps updates stable)
        lr = self._get_step_size(ratio) * (sigma.item() ** 2)

        # Initialize MCMC chain at x0_hat
        x0 = x0_hat.detach().clone().requires_grad_(cfg.mcmc_algo != "mh")
        velocity = torch.randn_like(x0) if cfg.mcmc_algo == "hmc" else None

        for _ in range(cfg.mcmc_steps):
            if cfg.mcmc_algo == "mh":
                # Symmetric Gaussian proposal
                x_prop = x0 + torch.randn_like(x0) * sqrt(lr)
                accept = self._mh_accept(x0, x_prop, x0_hat, sigma_b, forward_op, measurement, cfg.beta_y)
                x0 = torch.where(accept, x_prop, x0)
                continue

            # For Langevin/HMC, compute gradients
            score, _ = self._score_langevin(x0, x0_hat, sigma_b, forward_op, measurement, cfg.beta_y)

            with torch.no_grad():
                eps = torch.randn_like(x0)
                eps = _apply_kind_constraints(eps, kind)

                if cfg.mcmc_algo == "langevin":
                    # x_{k+1} = x_k + eps_k * score + sqrt(2 * eps_k) * z
                    x0 = x0 + lr * score + sqrt(2.0 * lr) * eps
                elif cfg.mcmc_algo == "hmc":
                    if velocity is None:
                        raise RuntimeError("HMC velocity state missing")
                    step_size = sqrt(lr)
                    velocity = cfg.hmc_momentum * velocity + step_size * score + sqrt(2.0 * (1.0 - cfg.hmc_momentum)) * eps
                    x0 = x0 + velocity * step_size
                else:
                    raise ValueError(f"Unknown mcmc_algo: {cfg.mcmc_algo!r}")

                x0 = self.model.clamp_x0(x0, kind)

            x0 = x0.detach().requires_grad_(True)

        x0 = x0.detach()

        # 3) Forward diffusion (re-noising) to next noise level sigma_next
        # x_{t-1} = x0 + sigma_{t-1} * eps

        sigma_next_b = sigma_next.expand(B)

        # q_sample in diffusion.py computes x0 + sigma * eps
        x_next = self.model.q_sample(x0, sigma_next_b, kind)

        return x_next, x0
