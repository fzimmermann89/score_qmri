"""DPS (Diffusion Posterior Sampling) for MRI reconstruction under VE/EDM.

Implements the core idea from Chung et al., "Diffusion Posterior Sampling for
General Noisy Inverse Problems" (ICLR 2023): combine diffusion sampling with
likelihood guidance via gradients through the denoiser prediction.

This implementation is sigma-native (continuous noise level) and uses an
EDM/Karras sigma schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel, _apply_kind_constraints, _create_noise

from .edm_sampler import karras_sigmas
from .protocols import LinearOperatorProtocol


@dataclass
class DPSConfig:
    """Configuration for DPS sampling.

    Parameters
    ----------
    num_steps
        Number of sigma steps (default 50).
    guidance_scale
        Guidance strength ζ for likelihood gradient.
    gradient_clipping
        Clamp gradients elementwise to [-gradient_clipping, +gradient_clipping].
        None disables clipping.
    normalize_gradient
        If True, normalize gradient per-sample to unit norm before scaling.
    rho
        Karras schedule rho (poly-rho), typically 7.
    sigma_min, sigma_max
        Override sigma range. If None, uses model's values.
    clamp_denoised
        If True, clamp x0_hat used for the *prior direction*.
        The measurement gradient is computed through the unclamped x0_hat to
        avoid saturating gradients.
    """

    num_steps: int = 50
    guidance_scale: float = 1.0
    gradient_clipping: float | None = 1.0
    normalize_gradient: bool = True
    rho: float = 7.0
    sigma_min: float | None = None
    sigma_max: float | None = None
    clamp_denoised: bool = True


def _to_complex(x_2ch: Tensor) -> Tensor:
    """Convert (B, 2, H, W) real to (B, 1, 1, H, W) complex for mrpro."""
    return torch.view_as_complex(torch.moveaxis(x_2ch, 1, -1).contiguous()).unsqueeze(1).unsqueeze(1)


class DPSSampler:
    """DPS sampler for MRI reconstruction with a VE/EDM diffusion prior."""

    def __init__(self, model: DiffusionModel, config: DPSConfig | None = None) -> None:
        self.model = model
        self.config = config or DPSConfig()

    @torch.enable_grad()
    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Reconstruct image from k-space measurement using DPS.

        Parameters
        ----------
        forward_op
            Linear operator A: image -> k-space. Must have .H adjoint.
            Expects (B, 1, 1, H, W) complex input, returns tuple (k-data,).
        measurement
            K-space measurement y of shape (B, Coils, 1, k2, k1) complex64.
        kind
            Dataset kind (must be complex).
        initial_noise
            Optional unit Gaussian noise (B, 2, H, W). If None, sampled from N(0, 1).

        Returns
        -------
        Tensor
            Reconstructed image (B, 1, 1, H, W) complex64 in normalized scale.
        """
        cfg = self.config
        device = self.model.device
        kind_idx = int(kind)

        if not is_complex_kind(kind_idx):
            raise ValueError(f"DPS only supports complex kinds, got {DatasetKind(kind_idx).name}")

        x_template = forward_op.H(measurement)[0]
        B, _, _, H, W = x_template.shape
        kind_tensor = torch.full((B,), kind_idx, dtype=torch.long, device=device)

        sigma_min = float(self.model.sigma_min if cfg.sigma_min is None else cfg.sigma_min)
        sigma_max = float(self.model.sigma_max if cfg.sigma_max is None else cfg.sigma_max)

        x = initial_noise.to(device) if initial_noise is not None else _create_noise((B, 2, H, W), kind_tensor, device, torch.float32)
        x = _apply_kind_constraints(x, kind_tensor)
        x = x * sigma_max

        sigmas = karras_sigmas(
            cfg.num_steps + 1,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=cfg.rho,
            device=device,
        )

        for i in range(cfg.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            x = self._step(x, kind_tensor, forward_op, measurement, sigma=sigma, sigma_next=sigma_next)

        sigma_final = torch.full((B,), sigma_min, device=device)
        x0 = self.model.denoise(x, sigma_final, kind_tensor)
        x0 = self.model.clamp_x0(x0, kind_tensor)
        return _to_complex(x0)

    def _step(
        self,
        x: Tensor,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        *,
        sigma: Tensor,
        sigma_next: Tensor,
    ) -> Tensor:
        """Single DPS step: EDM prior update + likelihood guidance (Euler)."""
        cfg = self.config
        B = x.shape[0]
        device = x.device

        sigma_val = float(sigma.item())
        sigma_b = torch.full((B,), sigma_val, device=device)
        sigma_in = sigma_b.view(-1, 1, 1, 1)

        x = x.detach().requires_grad_(True)

        # Unclamped denoiser output for measurement gradient.
        x0_hat = self.model.denoise(x, sigma_b, kind)

        x0_complex = _to_complex(x0_hat)
        y_pred = forward_op(x0_complex)[0]

        residual = y_pred - measurement
        loss = torch.linalg.vector_norm(residual.view(B, -1), dim=1).sum()
        (grad,) = torch.autograd.grad(loss, x)

        if cfg.normalize_gradient:
            grad_norm = grad.reshape(B, -1).norm(dim=1).view(B, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

        if cfg.gradient_clipping is not None:
            grad = torch.clamp(grad, -cfg.gradient_clipping, cfg.gradient_clipping)

        # Prior direction optionally uses clamped x0_hat for stability.
        x0_for_prior = self.model.clamp_x0(x0_hat.detach(), kind) if cfg.clamp_denoised else x0_hat.detach()
        d_prior = (x.detach() - x0_for_prior) / sigma_in.clamp(min=1e-8)

        d_guided = d_prior + cfg.guidance_scale * sigma_in * grad.detach()
        dt = float(sigma_next.item() - sigma.item())
        x_next = x.detach() + dt * d_guided
        return _apply_kind_constraints(x_next, kind)
