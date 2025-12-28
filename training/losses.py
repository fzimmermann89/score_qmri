"""Loss functions for diffusion training."""

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None


def _expand_mask(mask: Tensor) -> Tensor:
    """Expand 3D mask to 4D for broadcasting."""
    return rearrange(mask, "b h w -> b 1 h w") if mask.ndim == 3 else mask


def l_simple(noise_pred: Tensor, noise_target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Simple MSE loss on noise prediction (score matching loss).

    Args:
        noise_pred: Predicted noise of shape (B, C, H, W).
        noise_target: Target noise of shape (B, C, H, W).
        mask: Optional mask of shape (B, H, W) or (B, 1, H, W).

    Returns:
        Scalar MSE loss.
    """
    if mask is not None:
        mask_4d = _expand_mask(mask).float()
        error = (noise_pred - noise_target).pow(2) * mask_4d
        loss: Tensor = error.sum() / mask_4d.sum().clamp(min=1)
    else:
        loss = torch.nn.functional.mse_loss(noise_pred, noise_target)
    return loss


def l1_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """L1 loss with optional masking.

    Args:
        pred: Predicted tensor of shape (B, C, H, W).
        target: Target tensor of shape (B, C, H, W).
        mask: Optional mask of shape (B, H, W) or (B, 1, H, W).

    Returns:
        Scalar L1 loss.
    """
    if mask is not None:
        mask_4d = _expand_mask(mask).float()
        error = (pred - target).abs() * mask_4d
        loss: Tensor = error.sum() / mask_4d.sum().clamp(min=1)
    else:
        loss = torch.nn.functional.l1_loss(pred, target)
    return loss


def ssim_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """SSIM loss (1 - SSIM).

    Args:
        pred: Predicted tensor of shape (B, C, H, W).
        target: Target tensor of shape (B, C, H, W).
        mask: Optional mask of shape (B, H, W) or (B, 1, H, W).

    Returns:
        Scalar SSIM loss.
    """
    if ssim is None:
        raise ImportError("pytorch-msssim required for SSIM loss")

    if mask is not None:
        mask_4d = _expand_mask(mask).float()
        pred = pred * mask_4d
        target = target * mask_4d

    ssim_val: Tensor = ssim(pred, target, data_range=1.0, size_average=True)
    return 1.0 - ssim_val


def ssim_l1_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    ssim_weight: float = 0.5,
    l1_weight: float = 0.5,
) -> Tensor:
    """Combined SSIM + L1 loss.

    Args:
        pred: Predicted tensor of shape (B, C, H, W).
        target: Target tensor of shape (B, C, H, W).
        mask: Optional mask of shape (B, H, W) or (B, 1, H, W).
        ssim_weight: Weight for SSIM loss component.
        l1_weight: Weight for L1 loss component.

    Returns:
        Scalar combined loss.
    """
    loss: Tensor = ssim_weight * ssim_loss(pred, target, mask) + l1_weight * l1_loss(pred, target, mask)
    return loss


@dataclass
class DiffusionLossOutput:
    """Output from diffusion loss computation."""

    total: Tensor
    l_simple: Tensor
    l1: Tensor
    ssim: Tensor


def compute_diffusion_loss(
    x0_pred: Tensor,
    x0_target: Tensor,
    noise_pred: Tensor,
    noise_target: Tensor,
    mask: Tensor | None = None,
    noise_weight: float = 1.0,
    x0_weight: float = 0.1,
    ssim_weight: float = 0.5,
    l1_weight: float = 0.5,
) -> DiffusionLossOutput:
    """Compute combined diffusion training loss.

    Args:
        x0_pred: Predicted clean image (B, 2, H, W).
        x0_target: Target clean image (B, 2, H, W).
        noise_pred: Predicted noise (B, 2, H, W).
        noise_target: Target noise (B, 2, H, W).
        mask: Optional mask (B, H, W) or (B, 1, H, W).
        noise_weight: Weight for noise prediction loss.
        x0_weight: Weight for clean image prediction loss.
        ssim_weight: SSIM weight within x0 loss.
        l1_weight: L1 weight within x0 loss.

    Returns:
        DiffusionLossOutput with total loss and components.
    """
    loss_simple = l_simple(noise_pred, noise_target, mask)
    loss_l1 = l1_loss(x0_pred, x0_target, mask)
    loss_ssim = ssim_loss(x0_pred, x0_target, mask)

    x0_loss = ssim_weight * loss_ssim + l1_weight * loss_l1
    total = noise_weight * loss_simple + x0_weight * x0_loss

    return DiffusionLossOutput(
        total=total,
        l_simple=loss_simple,
        l1=loss_l1,
        ssim=loss_ssim,
    )
