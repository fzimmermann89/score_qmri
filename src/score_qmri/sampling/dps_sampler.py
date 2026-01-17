"""DPS (Diffusion Posterior Sampling) for MRI reconstruction.

Implements the algorithm from Chung et al., "Diffusion Posterior Sampling
for General Noisy Inverse Problems" (ICLR 2023).

Core idea: combine unconditional diffusion sampling with likelihood guidance
via gradients through the predicted clean image x̂₀.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel, _apply_kind_constraints, _create_noise

from .protocols import LinearOperatorProtocol


@dataclass
class DPSConfig:
    """Configuration for DPS sampling.

    Parameters
    ----------
    num_steps
        Number of diffusion steps (default 100).
    eta
        DDIM stochasticity parameter (0 = deterministic, 1 = DDPM).
    guidance_scale
        Guidance strength ζ for likelihood gradient.
    gradient_clipping
        Max norm for gradient clipping (None = no clipping).
    normalize_gradient
        If True, normalize gradient to unit norm before scaling.
    """

    num_steps: int = 100
    eta: float = 0.0
    guidance_scale: float = 1.0
    gradient_clipping: float | None = 1.0
    normalize_gradient: bool = True


def _to_complex(x_2ch: Tensor) -> Tensor:
    """Convert (B, 2, H, W) real to (B, 1, 1, H, W) complex for mrpro."""
    return torch.view_as_complex(torch.moveaxis(x_2ch, 1, -1).contiguous()).unsqueeze(1).unsqueeze(1)


def _from_complex(x_complex: Tensor) -> Tensor:
    """Convert (B, 1, 1, H, W) complex to (B, 2, H, W) real."""
    return torch.moveaxis(torch.view_as_real(x_complex.squeeze(2).squeeze(1)), -1, 1)


class DPSSampler:
    """DPS sampler for MRI reconstruction with learned diffusion prior.

    Operates entirely in normalized space. The forward operator A maps
    images to k-space: k = A(x). Dataset scaling is assumed already applied
    to both images and measurements.
    """

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
            Linear operator A: image → k-space. Must have .H adjoint.
            Expects (B, 1, 1, H, W) complex input, returns tuple (k-data,).
            Both operator and measurements are assumed to be in dataset-scaled space.
        measurement
            K-space measurement y of shape (B, Coils, 1, k2, k1) complex64.
        kind
            Dataset kind (must be complex: T1W_COMPLEX, T2W_COMPLEX, etc.).
        initial_noise
            Optional starting noise (B, 2, H, W). If None, sampled from N(0,1).

        Returns
        -------
        Reconstructed image (B, 1, 1, H, W) complex64 in normalized scale.
        """
        cfg = self.config
        device = self.model.device
        kind_idx = int(kind)

        if not is_complex_kind(kind_idx):
            raise ValueError(f"DPS only supports complex kinds, got {DatasetKind(kind_idx).name}")

        # Infer image shape from adjoint
        x_template = forward_op.H(measurement)[0]  # (B, 1, 1, H, W)
        B, _, _, H, W = x_template.shape

        kind_tensor = torch.full((B,), kind_idx, dtype=torch.long, device=device)

        # Initialize in normalized space with standard Gaussian noise
        x = initial_noise.to(device) if initial_noise is not None else _create_noise((B, 2, H, W), kind_tensor, device, torch.float32)

        x = _apply_kind_constraints(x, kind_tensor)

        # Time schedule
        eps = 1e-3
        timesteps = torch.linspace(1.0 - eps, eps, cfg.num_steps + 1, device=device)

        x0_pred = x

        for i in range(cfg.num_steps):
            t = timesteps[i].expand(B)
            t_prev = timesteps[i + 1].expand(B)

            x, x0_pred = self._step(x, t, t_prev, kind_tensor, forward_op, measurement)

        return _to_complex(x0_pred)

    def _step(
        self,
        x: Tensor,
        t: Tensor,
        t_prev: Tensor,
        kind: Tensor,
        forward_op: LinearOperatorProtocol,
        measurement: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Single DPS step: DDIM update + likelihood guidance."""
        cfg = self.config
        B = x.shape[0]

        # Enable gradients for guidance
        x = x.detach().requires_grad_(True)

        # Get noise prediction and x0 estimate (all in normalized space)
        output = self.model(x, t, kind)
        noise_pred = output.noise
        x0_pred = self.model.predict_x0(x, t, noise_pred, kind)
        x0_pred = self.model.clamp_x0(x0_pred, kind)

        # Data fidelity gradient
        # For complex kinds, normalize/denormalize is identity, so we can use x0_pred directly
        x0_complex = _to_complex(x0_pred)  # (B, 1, 1, H, W) complex
        y_pred = forward_op(x0_complex)[0]

        # L2 loss in k-space
        # Chung et al. 2023 uses the norm (not squared norm) to stabilize step size
        residual = y_pred - measurement
        # Flatten to compute vector norm over all dimensions except batch
        loss = torch.linalg.vector_norm(residual.view(B, -1), dim=1).sum()

        # Gradient w.r.t. x (normalized state)
        (grad,) = torch.autograd.grad(loss, x)

        # Process gradient
        if cfg.normalize_gradient:
            grad_norm = grad.reshape(B, -1).norm(dim=1).view(B, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)

        if cfg.gradient_clipping is not None:
            grad = torch.clamp(grad, -cfg.gradient_clipping, cfg.gradient_clipping)

        # Detach for DDIM step
        x = x.detach()
        noise_pred = noise_pred.detach()
        x0_pred = x0_pred.detach()

        # DDIM step in normalized space
        ab_t = self.model.alpha_bar(t).view(-1, 1, 1, 1)
        ab_prev = self.model.alpha_bar(t_prev).view(-1, 1, 1, 1)

        # DDIM variance
        one_minus_ab_t = (1 - ab_t).clamp(min=1e-8)
        one_minus_ab_prev = (1 - ab_prev).clamp(min=0.0)
        sigma_sq = cfg.eta**2 * (one_minus_ab_prev / one_minus_ab_t * (1 - ab_t / ab_prev.clamp(min=1e-8)))
        sigma = sigma_sq.clamp(min=0.0).sqrt()

        # DDIM step
        pred_dir = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt() * noise_pred
        x_prev = ab_prev.sqrt() * x0_pred + pred_dir

        if cfg.eta > 0 and t_prev[0] > 1e-8:
            noise = _create_noise(x.shape, kind, x.device, x.dtype)
            x_prev = x_prev + sigma * noise

        # Apply DPS guidance correction (in normalized space)
        x_prev = x_prev - cfg.guidance_scale * grad

        x_prev = _apply_kind_constraints(x_prev, kind)

        return x_prev, x0_pred
