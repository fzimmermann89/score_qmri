"""Loss functions for diffusion training."""

from collections.abc import Callable

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


class MSELoss:
    """Masked MSE loss for image space (x0) prediction."""

    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute masked MSE loss.

        Parameters
        ----------
        pred : Tensor
            Predicted x0 of shape (B, C, H, W).
        target : Tensor
            Target x0 of shape (B, C, H, W).
        mask : Tensor
            Mask of shape (B, H, W) or (B, 1, H, W). True for valid pixels.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        mask = _expand_mask(mask).float()
        error = (pred - target).pow(2) * mask
        return error.sum() / mask.sum().clamp(min=1)


class SSIML1Loss:
    """Combined SSIM + L1 loss with masking.

    Parameters
    ----------
    ssim_weight : float
        Weight for SSIM component (default: 0.5).
    l1_weight : float
        Weight for L1 component (default: 0.5).

    Raises
    ------
    ImportError
        If pytorch-msssim is not installed.
    """

    def __init__(self, ssim_weight: float = 0.5, l1_weight: float = 0.5) -> None:
        if ssim is None:
            raise ImportError(
                "pytorch-msssim required for SSIM loss. Install: pip install pytorch-msssim"
            )
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight

    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute SSIM + L1 loss.

        Parameters
        ----------
        pred : Tensor
            Predicted x0 of shape (B, C, H, W).
        target : Tensor
            Target x0 of shape (B, C, H, W).
        mask : Tensor
            Mask of shape (B, H, W) or (B, 1, H, W). True for valid pixels.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        mask = _expand_mask(mask).float()

        ssim_val = ssim(pred * mask, target * mask, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        l1_error = (pred - target).abs() * mask
        l1_loss = l1_error.sum() / mask.sum().clamp(min=1)

        return self.ssim_weight * ssim_loss + self.l1_weight * l1_loss


class LossScheduler:
    """Schedule-based interpolation between MSE and SSIM+L1 losses.

    Parameters
    ----------
    mse_loss : MSELoss
        MSE loss instance.
    ssim_l1_loss : SSIML1Loss
        SSIM+L1 loss instance.
    schedule_fn : Callable[[int], float]
        Function mapping step -> weight for SSIM+L1 (0.0 = MSE only, 1.0 = SSIM+L1 only).
    """

    def __init__(
        self,
        mse_loss: MSELoss,
        ssim_l1_loss: SSIML1Loss,
        schedule_fn: Callable[[int], float],
    ) -> None:
        self.mse_loss = mse_loss
        self.ssim_l1_loss = ssim_l1_loss
        self.schedule_fn = schedule_fn
        self.step = 0

    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute loss based on current schedule.

        Parameters
        ----------
        pred : Tensor
            Predicted x0.
        target : Tensor
            Target x0.
        mask : Tensor
            Mask tensor.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        w = self.schedule_fn(self.step)
        self.step += 1
        mse = self.mse_loss(pred, target, mask)
        ssim_l1 = self.ssim_l1_loss(pred, target, mask)
        return (1.0 - w) * mse + w * ssim_l1

    def reset(self) -> None:
        """Reset step counter."""
        self.step = 0
