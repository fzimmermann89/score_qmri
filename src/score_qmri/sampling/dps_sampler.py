"""DPS (Diffusion Posterior Sampling) for MRI reconstruction under VE/EDM.

Implements the core idea from Chung et al., "Diffusion Posterior Sampling for
General Noisy Inverse Problems" (ICLR 2023): combine diffusion sampling with
likelihood guidance via gradients through the denoiser prediction.

This implementation is sigma-native (continuous noise level) and uses an
EDM/Karras sigma schedule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel, _apply_kind_constraints, _create_noise

from .edm_sampler import karras_sigmas
from .protocols import LinearOperatorProtocol


def to_complex(x_2ch: Tensor) -> Tensor:
    """Convert (B, 2, H, W) real to (B, 1, 1, H, W) complex."""
    return torch.view_as_complex(torch.moveaxis(x_2ch, 1, -1).contiguous()).unsqueeze(1).unsqueeze(1)


def to_2ch(x_complex: Tensor) -> Tensor:
    """Convert (B, 1, 1, H, W) complex to (B, 2, H, W) real."""
    z = x_complex.squeeze(1).squeeze(1)
    return torch.view_as_real(z).moveaxis(-1, 1).contiguous()


class GuidanceWeighting(Enum):
    """Guidance weighting strategy for DPS."""

    CONSTANT = auto()
    """Fixed weight: scale * grad"""

    SIGMA_SCALED = auto()
    """Sigma-dependent: scale / (sigma + floor) * grad"""

    RESIDUAL_NORM = auto()
    """Paper-style: scale / ||y - A(x0)|| * grad"""


@dataclass
class DPSConfig:
    """Configuration for DPS sampling.

    Parameters
    ----------
    num_steps
        Number of diffusion steps.
    guidance_scale
        Data consistency gradient strength.
    weighting
        Guidance weighting strategy.
    sigma_start
        Starting noise level. None uses model's sigma_max.
    sigma_floor
        For SIGMA_SCALED: weight = scale / (sigma + floor).
    rho
        Karras schedule exponent.
    normalize_gradient
        Normalize gradients by RMS before weighting.
    clamp_x0_prior
        Clamp x0 estimate for prior score direction.
    clamp_x0_dc
        Clamp x0 estimate before forward model in data consistency.
    """

    num_steps: int = 1000
    guidance_scale: float = 0.1
    weighting: GuidanceWeighting = GuidanceWeighting.SIGMA_SCALED
    sigma_start: float | None = None
    sigma_floor: float = 0.05
    rho: float = 7.0
    normalize_gradient: bool = True
    clamp_x0_prior: bool = False
    clamp_x0_dc: bool = False


@dataclass
class DPSStepInfo:
    """Diagnostic info from a DPS step."""

    step: int
    sigma: float
    dc_loss: float
    residual_norm: float
    x0_absmax: float
    grad_norm: float


class DPSSampler:
    """DPS sampler for MRI reconstruction with VE/EDM diffusion prior.

    Implements Chung et al., "Diffusion Posterior Sampling for General Noisy
    Inverse Problems" (ICLR 2023), adapted for VE/EDM formulation.
    """

    def __init__(
        self,
        model: DiffusionModel,
        config: DPSConfig | None = None,
        callback: Callable[[DPSStepInfo], None] | None = None,
    ) -> None:
        self.model = model
        self.cfg = config or DPSConfig()
        self.callback = callback

    @torch.enable_grad()
    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        *,
        x0_init: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Reconstruct image from k-space measurement.

        Parameters
        ----------
        forward_op
            Forward operator A: image -> k-space with .H adjoint.
        measurement
            K-space data (B, Coils, 1, k2, k1) complex.
        kind
            Dataset kind (must be complex).
        x0_init
            Warm-start image (B, 1, 1, H, W) complex, e.g. from adjoint. Optional.
        noise
            Unit complex Gaussian noise (B, 1, 1, H, W). Sampled if None.

        Returns
        -------
        Tensor
            Reconstructed image (B, 1, 1, H, W) complex.
        """
        cfg = self.cfg
        device = self.model.device
        kind_idx = int(kind)

        if not is_complex_kind(kind_idx):
            raise NotImplementedError(f"DPS only implemented for complex kind, got {DatasetKind(kind_idx).name}")

        template = forward_op.H(measurement)[0]
        B, _, _, H, W = template.shape
        kind_t = torch.full((B,), kind_idx, dtype=torch.long, device=device)

        sigma_min = self.model.sigma_min
        sigma_start = cfg.sigma_start if cfg.sigma_start is not None else self.model.sigma_max

        sigmas = karras_sigmas(
            cfg.num_steps + 1,
            sigma_min=sigma_min,
            sigma_max=sigma_start,
            rho=cfg.rho,
            device=device,
        )

        noise_2ch = _create_noise((B, 2, H, W), kind_t, device, torch.float32) if noise is None else _to_2ch(noise.to(device))
        x0_init_2ch = _to_2ch(x0_init.to(device)) if x0_init is not None else None
        x = x0_init_2ch + noise_2ch * sigma_start if x0_init_2ch is not None else noise_2ch * sigma_start

        for i in range(cfg.num_steps):
            x = self._step(
                x,
                kind_t,
                forward_op,
                measurement,
                sigma=float(sigmas[i]),
                sigma_next=float(sigmas[i + 1]),
                step_idx=i,
            )

        sigma_final = torch.full((B,), sigma_min, device=device)
        x0 = self.model.denoise(x, sigma_final, kind_t)
        x0 = self.model.clamp_x0(x0, kind_t)
        return _to_complex(x0)

    def _compute_guidance_weight(
        self,
        sigma: float,
        residual_norm: Tensor,
    ) -> Tensor:
        """Compute guidance weight based on weighting strategy."""
        cfg = self.cfg
        B = residual_norm.shape[0]
        device = residual_norm.device

        match cfg.weighting:
            case GuidanceWeighting.CONSTANT:
                return torch.full((B, 1, 1, 1), cfg.guidance_scale, device=device)

            case GuidanceWeighting.SIGMA_SCALED:
                weight = cfg.guidance_scale / (sigma + cfg.sigma_floor)
                return torch.full((B, 1, 1, 1), weight, device=device)

            case GuidanceWeighting.RESIDUAL_NORM:
                w = cfg.guidance_scale / (residual_norm + 1e-8)
                return w.view(B, 1, 1, 1)

    def _step(
        self,
        x: Tensor,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        *,
        sigma: float,
        sigma_next: float,
        step_idx: int,
    ) -> Tensor:
        """Single DPS step: Euler on probability flow ODE + DC guidance."""
        cfg = self.cfg
        B = x.shape[0]
        device = x.device

        sigma_t = torch.full((B,), sigma, device=device)

        x_in = x.detach().requires_grad_(True)
        x0_hat = self.model.denoise(x_in, sigma_t, kind)

        x0_for_dc = self.model.clamp_x0(x0_hat, kind) if cfg.clamp_x0_dc else x0_hat
        y_pred = forward_op(_to_complex(x0_for_dc))[0]

        residual = (y_pred - measurement).flatten(start_dim=1)
        residual_norm = residual.abs().square().sum(dim=1).sqrt()
        loss = 0.5 * residual.abs().square().mean(dim=1).sum()

        (grad,) = torch.autograd.grad(loss, x_in)

        grad_norm_raw = grad.flatten(start_dim=1).norm(dim=1)

        if cfg.normalize_gradient:
            numel = float(grad[0].numel())
            rms = grad_norm_raw / (numel**0.5)
            grad = grad / (rms.view(B, 1, 1, 1) + 1e-12)

        grad = grad.detach()
        x0_hat = x0_hat.detach()
        x = x_in.detach()

        x0_for_prior = self.model.clamp_x0(x0_hat, kind) if cfg.clamp_x0_prior else x0_hat

        sigma_safe = max(sigma, 1e-8)
        d_prior = (x - x0_for_prior) / sigma_safe

        guidance_weight = self._compute_guidance_weight(sigma, residual_norm)
        d = d_prior + guidance_weight * grad

        dt = sigma_next - sigma
        x_next = x + dt * d

        if self.callback is not None:
            with torch.no_grad():
                info = DPSStepInfo(
                    step=step_idx,
                    sigma=sigma,
                    dc_loss=float(loss),
                    residual_norm=float(residual_norm.mean()),
                    x0_absmax=float(x0_hat.abs().max()),
                    grad_norm=float(grad_norm_raw.mean()),
                )
            self.callback(info)

        return _apply_kind_constraints(x_next, kind)
