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
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel, _apply_kind_constraints
from score_qmri.sampling.edm_sampler import karras_sigmas

from .gaussian import CGInfo, sample_linear_gaussian
from .protocols import LinearOperatorProtocol


def to_complex(x_2ch: Tensor) -> Tensor:
    """Convert (B, 2, H, W) real to (B, 1, 1, H, W) complex."""
    return torch.complex(x_2ch[:, 0], x_2ch[:, 1]).unsqueeze(1).unsqueeze(1)


def to_2ch(x_complex: Tensor) -> Tensor:
    """Convert (B, 1, 1, H, W) complex to (B, 2, H, W) real."""
    z = x_complex.squeeze(1).squeeze(1)
    return torch.view_as_real(z).moveaxis(-1, 1).contiguous()


@dataclass
class DAPSConfig:
    """DAPS settings; tau is measurement noise std per real coordinate.

    Finite-step Langevin preserves the historical sampling behavior. An exact
    draw from the Gaussian approximation can overdisperse the outer trajectory.
    The optional Gaussian kernel requires a linear, batch-independent operator and its
    accurate adjoint. Langevin uses operator_norm as a supplied upper bound on
    the operator spectral norm, and remains a finite-step approximation.
    """

    inner_kernel: Literal["gaussian", "langevin"] = "langevin"
    cg_max_iter: int = 200
    cg_rtol: float = 1e-5
    anneal_steps: int = 200
    sigma_start: float = 5.0
    sigma_min: float = 0.1
    rho: float = 7.0

    reverse_steps: int = 5
    sigma_x0: float = 0.01

    langevin_steps: int = 100
    langevin_lr: float = 1.5e-5
    lr_min_ratio: float = 0.01
    operator_norm: float = 1.0

    tau: float = 0.0349

    clamp_x0hat: bool = False
    clamp_limit: float | None = 1.5

    debug_every: int = 25

    def __post_init__(self) -> None:
        if not all(
            isfinite(v)
            for v in (
                self.sigma_x0,
                self.sigma_min,
                self.sigma_start,
                self.rho,
                self.tau,
                self.langevin_lr,
                self.lr_min_ratio,
                self.operator_norm,
            )
        ):
            raise ValueError("DAPS settings must be finite")
        if self.inner_kernel not in ("gaussian", "langevin"):
            raise ValueError("inner_kernel must be gaussian or langevin")
        if self.anneal_steps < 2 or self.reverse_steps < 1 or self.langevin_steps < 1:
            raise ValueError("annealing needs >=2 steps; reverse and Langevin steps must be positive")
        if not 0 < self.sigma_x0 <= self.sigma_min < self.sigma_start:
            raise ValueError("require 0 < sigma_x0 <= sigma_min < sigma_start")
        if self.tau <= 0 or self.operator_norm < 0 or self.langevin_lr <= 0 or not 0 < self.lr_min_ratio <= 1:
            raise ValueError("invalid measurement noise or Langevin settings")


class DAPSSampler:
    def __init__(self, model: DiffusionModel, cfg: DAPSConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or DAPSConfig()
        self.cg_diagnostics: list[CGInfo] = []

    def _lr_at(self, ratio: float) -> float:
        cfg = self.cfg
        return cfg.langevin_lr * (1.0 + ratio * (cfg.lr_min_ratio - 1.0))

    @torch.no_grad()
    def _x0hat_pf_ode(self, x_t: Tensor, sigma: float, kind: Tensor) -> Tensor:
        cfg = self.cfg
        bsz = x_t.shape[0]

        if sigma <= cfg.sigma_x0:
            sigma_t = torch.full((bsz,), float(sigma), device=x_t.device)
            x0 = self.model.denoise(x_t, sigma_t, kind)
            if cfg.clamp_x0hat and cfg.clamp_limit is not None:
                x0 = self.model.clamp_x0(x0, kind, limit=cfg.clamp_limit)
            return x0

        sigmas = karras_sigmas(
            cfg.reverse_steps + 1,
            sigma_min=cfg.sigma_x0,
            sigma_max=float(sigma),
            rho=self.model.rho,
            device=x_t.device,
        )

        x = x_t
        for i in range(cfg.reverse_steps):
            s = float(sigmas[i].item())
            s_next = float(sigmas[i + 1].item())
            sigma_t = torch.full((bsz,), s, device=x_t.device)
            x0 = self.model.denoise(x, sigma_t, kind)
            d = (x - x0) / max(s, 1e-12)
            x = x + (s_next - s) * d

        sigma_t = torch.full((bsz,), cfg.sigma_x0, device=x_t.device)
        x0 = self.model.denoise(x, sigma_t, kind)
        if cfg.clamp_x0hat and cfg.clamp_limit is not None:
            x0 = self.model.clamp_x0(x0, kind, limit=cfg.clamp_limit)
        return x0

    @torch.enable_grad()
    def _langevin(
        self,
        x0hat: Tensor,
        sigma: float,
        ratio: float,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        cfg = self.cfg
        # Control the stiff prior term as sigma approaches zero. The data bound
        # is valid for a linear operator whose norm is <= operator_norm.
        curvature = sigma**-2 + cfg.operator_norm**2 / cfg.tau**2
        lr = min(self._lr_at(ratio), 0.1 / curvature)
        tau2 = float(cfg.tau**2)
        sig2 = float(sigma**2)

        x = x0hat.detach().clone().requires_grad_(True)

        for _ in range(cfg.langevin_steps):
            y_pred = forward_op(to_complex(x))[0]
            resid = y_pred - measurement
            data_energy = 0.5 * resid.abs().square().sum()
            (grad_data,) = torch.autograd.grad(data_energy, x, create_graph=False)

            grad_prior = (x - x0hat) / max(sig2, 1e-12)
            grad = grad_data / tau2 + grad_prior

            with torch.no_grad():
                eps = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                eps = _apply_kind_constraints(eps, kind)
                x = x - lr * grad + sqrt(2.0 * lr) * eps

            x = x.detach().requires_grad_(True)

            if not x.isfinite().all():
                raise RuntimeError("Nonfinite DAPS Langevin trajectory; check step size and operator/noise units")

        return x.detach()

    @torch.enable_grad()
    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        *,
        x0_init: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw a complex image using Gaussian-approximation DAPS.

        All image/measurement units must agree. Supplied initial noise is complex
        with unit variance in each real component. Warm starts and a finite outer
        schedule remain approximations to posterior sampling.
        """
        cfg = self.cfg
        self.cg_diagnostics.clear()
        device = self.model.device
        kind_idx = int(kind)

        if not is_complex_kind(kind_idx):
            raise NotImplementedError(f"DAPS only supports complex kinds, got {DatasetKind(kind_idx).name}")

        template = forward_op.H(measurement)[0]
        bsz, _, _, h, w = template.shape
        kind_t = torch.full((bsz,), kind_idx, dtype=torch.long, device=device)

        sigmas = karras_sigmas(
            cfg.anneal_steps,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_start,
            rho=cfg.rho,
            device=device,
        )
        sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])], dim=0)

        x0_init_2ch = None if x0_init is None else to_2ch(x0_init.to(device))

        if noise is None:
            eps0 = torch.randn((bsz, 2, h, w), device=device, dtype=template.real.dtype, generator=generator)
            eps0 = _apply_kind_constraints(eps0, kind_t)
        else:
            eps0 = to_2ch(noise.to(device))

        x_t = eps0 * float(cfg.sigma_start) if x0_init_2ch is None else x0_init_2ch + eps0 * float(cfg.sigma_start)
        x_t = _apply_kind_constraints(x_t, kind_t)

        if cfg.debug_every > 0:
            print(f"{'i':>4} {'sigma':>9} {'lr':>10} {'||r(x0hat)||':>12} {'||r(x0)||':>10}")

        x0_last = x_t
        for i in range(cfg.anneal_steps):
            sigma = float(sigmas[i].item())
            sigma_next = float(sigmas[i + 1].item())
            ratio = i / max(cfg.anneal_steps - 1, 1)

            x0hat = self._x0hat_pf_ode(x_t, sigma, kind_t)
            if cfg.inner_kernel == "gaussian":
                draw, info = sample_linear_gaussian(
                    forward_op,
                    measurement,
                    to_complex(x0hat),
                    sigma,
                    cfg.tau,
                    generator=generator,
                    max_iter=cfg.cg_max_iter,
                    rtol=cfg.cg_rtol,
                )
                self.cg_diagnostics.append(info)
                x0 = to_2ch(draw)
            else:
                x0 = self._langevin(x0hat, sigma, ratio, kind_t, forward_op, measurement, generator)

            if cfg.debug_every > 0 and (i % cfg.debug_every == 0 or i == cfg.anneal_steps - 1):
                with torch.no_grad():
                    r_hat = forward_op(to_complex(x0hat))[0] - measurement
                    r_x0 = forward_op(to_complex(x0))[0] - measurement
                    print(
                        f"{i:4d} {sigma:9.4g} {self._lr_at(ratio):10.3g} "
                        f"{float(r_hat.abs().square().mean().sqrt().item()):12.4g} "
                        f"{float(r_x0.abs().square().mean().sqrt().item()):10.4g}"
                    )

            with torch.no_grad():
                if sigma_next == 0.0:
                    x_t = x0
                else:
                    eps = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
                    eps = _apply_kind_constraints(eps, kind_t)
                    x_t = x0 + float(sigma_next) * eps

            x0_last = x0

        return to_complex(x0_last)
