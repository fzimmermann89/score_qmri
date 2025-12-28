"""Loss functions for diffusion training."""

from dataclasses import dataclass

import torch
from torch import Tensor

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None


@dataclass
class LossOutput:
    """Combined loss output."""

    total: Tensor
    noise_mse: Tensor
    x0_l1: Tensor
    x0_ssim: Tensor


def noise_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    MSE loss on noise prediction (L_simple).

    Parameters
    ----------
    pred
        Predicted noise of shape (B, C, H, W).
    target
        Target noise of shape (B, C, H, W).
    mask
        Optional mask of shape (B, 1, H, W).

    Returns
    -------
    Scalar loss value.
    """
    if mask is None:
        return torch.nn.functional.mse_loss(pred, target)
    mask = mask.float()
    return ((pred - target).pow(2) * mask).sum() / mask.sum().clamp(min=1)


def l1_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    L1 loss with optional masking.

    Parameters
    ----------
    pred
        Predicted tensor of shape (B, C, H, W).
    target
        Target tensor of shape (B, C, H, W).
    mask
        Optional mask of shape (B, 1, H, W).

    Returns
    -------
    Scalar loss value.
    """
    if mask is None:
        return torch.nn.functional.l1_loss(pred, target)
    mask = mask.float()
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1)


def ssim_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    SSIM loss (1 - SSIM).

    Parameters
    ----------
    pred
        Predicted tensor of shape (B, C, H, W).
    target
        Target tensor of shape (B, C, H, W).
    mask
        Optional mask of shape (B, 1, H, W).

    Returns
    -------
    Scalar loss value.

    Raises
    ------
    ImportError
        If pytorch-msssim is not installed.
    """
    if ssim is None:
        raise ImportError("pytorch-msssim required")
    if mask is not None:
        mask = mask.float()
        pred, target = pred * mask, target * mask
    ssim_val: Tensor = ssim(pred, target, data_range=1.0, size_average=True)
    return 1.0 - ssim_val


def compute_loss(
    x0_pred: Tensor,
    x0_target: Tensor,
    noise_pred: Tensor,
    noise_target: Tensor,
    mask: Tensor | None = None,
    noise_weight: float = 1.0,
    x0_weight: float = 0.1,
    ssim_weight: float = 0.5,
    l1_weight: float = 0.5,
) -> LossOutput:
    """
    Compute combined diffusion training loss.

    Parameters
    ----------
    x0_pred
        Predicted clean image of shape (B, 2, H, W).
    x0_target
        Target clean image of shape (B, 2, H, W).
    noise_pred
        Predicted noise of shape (B, 2, H, W).
    noise_target
        Target noise of shape (B, 2, H, W).
    mask
        Optional mask of shape (B, 1, H, W).
    noise_weight
        Weight for noise prediction loss.
    x0_weight
        Weight for clean image prediction loss.
    ssim_weight
        SSIM weight within x0 loss.
    l1_weight
        L1 weight within x0 loss.

    Returns
    -------
    Contains total loss and individual components.
    """
    loss_noise = noise_loss(noise_pred, noise_target, mask)
    loss_l1 = l1_loss(x0_pred, x0_target, mask)
    loss_ssim = ssim_loss(x0_pred, x0_target, mask)

    x0_loss = ssim_weight * loss_ssim + l1_weight * loss_l1
    total = noise_weight * loss_noise + x0_weight * x0_loss

    return LossOutput(total=total, noise_mse=loss_noise, x0_l1=loss_l1, x0_ssim=loss_ssim)
