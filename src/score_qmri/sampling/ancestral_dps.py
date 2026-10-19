"""Ancestral DPS with an EDM denoiser and explicit VP-coordinate conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from .daps_sampler import to_2ch, to_complex
from .edm_sampler import karras_sigmas
from .protocols import LinearOperatorProtocol


@dataclass
class AncestralDPSConfig:
    """DPS reference transition, evaluated using an EDM clean-image denoiser.

    ``residual_norm`` follows the official Gaussian conditioning implementation.
    ``gaussian_likelihood`` uses the summed Gaussian energy with noise standard
    deviation tau per real coordinate. Guidance is differentiated in VP state
    coordinates; there is no RMS normalization or inverse-sigma multiplier.
    The bridge variance is fixed-small (DDPM posterior variance), since an EDM
    denoiser does not supply the reference model's optional learned variance.
    """

    num_steps: int = 1000
    guidance_scale: float = 0.1
    conditioning: Literal["residual_norm", "gaussian_likelihood"] = "residual_norm"
    tau: float | None = None
    sigma_start: float | None = None
    sigma_min: float | None = None
    rho: float = 7.0
    clip_denoised: bool = False

    def __post_init__(self) -> None:
        if self.num_steps < 2 or not isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("DPS needs >=2 steps and a finite nonnegative guidance scale")
        if self.conditioning not in ("residual_norm", "gaussian_likelihood"):
            raise ValueError("Unknown DPS conditioning")
        if not isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive")
        if self.tau is not None and (not isfinite(self.tau) or self.tau <= 0):
            raise ValueError("tau must be finite and positive")
        if self.conditioning == "gaussian_likelihood" and self.tau is None:
            raise ValueError("Gaussian likelihood conditioning requires tau per real coordinate")


class AncestralDPSSampler:
    """Stochastic DPS, preserving the ancestral update and denoiser Jacobian.

    For VP state z, alpha_bar=1/(1+sigma²) and u=z/sqrt(alpha_bar).
    D_EDM(u,sigma) is the corresponding clean-image estimate. This change of
    coordinates requires neither a VP-trained network nor noise-head retraining.
    DPS's plug-in likelihood approximation remains part of the method.
    """

    def __init__(self, model: DiffusionModel, cfg: AncestralDPSConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg or AncestralDPSConfig()
        self.last_sigmas: Tensor | None = None

    @torch.enable_grad()
    def _step(
        self,
        z: Tensor,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        sigma: float,
        sigma_next: float,
        *,
        noise: Tensor,
    ) -> Tensor:
        """One DDPM bridge plus DPS correction, with both gradients in VP units."""
        if not 0 <= sigma_next < sigma:
            raise ValueError("Ancestral DPS requires decreasing positive noise levels")
        cfg = self.cfg
        scale = 1 / sqrt(1 + sigma**2)
        scale_next = 1 / sqrt(1 + sigma_next**2)
        z = z.detach().requires_grad_(True)
        sigmas = z.new_full((z.shape[0],), sigma)
        clean = self.model.denoise(z / scale, sigmas, kind)
        if cfg.clip_denoised:
            clean = self.model.clamp_x0(clean, kind)
        residual = (forward_op(to_complex(clean))[0] - measurement).flatten(1)
        if cfg.conditioning == "residual_norm":
            energy = torch.linalg.vector_norm(residual, dim=1).sum()
        else:
            assert cfg.tau is not None
            energy = residual.abs().square().sum() / (2 * cfg.tau**2)
        (gradient,) = torch.autograd.grad(energy, z)

        ratio = (sigma_next / sigma) ** 2
        mean = ratio * (z / scale) + (1 - ratio) * clean
        variance = sigma_next**2 * (1 - ratio)
        # Conjugating the DDPM bridge into EDM coordinates yields this mean and
        # variance. The reference correction is made after the reverse draw.
        proposal = scale_next * (mean + sqrt(variance) * noise)
        result = proposal - cfg.guidance_scale * gradient
        if not result.isfinite().all():
            raise RuntimeError("Nonfinite ancestral DPS trajectory; check units and guidance scale")
        return result.detach()

    def sample(
        self,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
        kind: DatasetKind | int,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw a complex image; initial noise is unit Gaussian per real part.

        Initial VP state is N(0,I), as in the reference algorithm. At a finite
        starting sigma this is an approximate initial marginal. All steps use
        the supplied generator. No final projection or extra denoising occurs.
        """
        if not is_complex_kind(int(kind)):
            raise NotImplementedError("Ancestral DPS currently supports complex image kinds")
        cfg = self.cfg
        measurement = measurement.to(self.model.device)
        template = forward_op.H(measurement)[0]
        batch, _, _, height, width = template.shape
        kinds = torch.full((batch,), int(kind), device=template.device, dtype=torch.long)
        sigma_start = self.model.sigma_max if cfg.sigma_start is None else cfg.sigma_start
        sigma_min = self.model.sigma_min if cfg.sigma_min is None else cfg.sigma_min
        if not isfinite(sigma_start) or not isfinite(sigma_min):
            raise ValueError("Noise limits must be finite")
        if sigma_start > self.model.sigma_max or sigma_min < self.model.sigma_min:
            raise ValueError("DPS noise levels must stay within the model's supported sigma range")
        sigmas = karras_sigmas(cfg.num_steps, sigma_min=sigma_min, sigma_max=sigma_start, rho=cfg.rho, device=template.device)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        self.last_sigmas = sigmas.detach().clone()
        shape = (batch, 2, height, width)
        z = (
            torch.randn(shape, dtype=template.real.dtype, device=template.device, generator=generator)
            if noise is None
            else to_2ch(noise.to(template.device))
        )
        if z.shape != shape:
            raise ValueError("Initial noise must match the image shape")
        for current, following in zip(sigmas[:-1], sigmas[1:], strict=True):
            epsilon = torch.randn(shape, dtype=z.dtype, device=z.device, generator=generator)
            z = self._step(z, kinds, forward_op, measurement, float(current), float(following), noise=epsilon)
        return to_complex(z)
