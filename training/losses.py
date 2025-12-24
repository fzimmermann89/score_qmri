"""Loss functions for diffusion training."""

from collections.abc import Callable

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
    """Masked MSE loss for image space (x0) prediction.

    Computes mean squared error only over valid (masked) pixels,
    averaging over channels.
    """

    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute masked MSE loss.

        Args:
            pred: Predicted x0 of shape (B, C, H, W).
            target: Target x0 of shape (B, C, H, W).
            mask: Mask of shape (B, H, W) or (B, 1, H, W). True for valid pixels.

        Returns:
            Scalar loss value.
        """
        mask_4d = _expand_mask(mask).float()
        error = (pred - target).pow(2) * mask_4d
        # Average over channels, sum over spatial, divide by valid pixels
        loss: Tensor = error.mean(dim=1).sum() / mask_4d.sum().clamp(min=1)
        return loss


class SSIML1Loss:
    """Combined SSIM + L1 loss with masking.

    Computes weighted combination of structural similarity and L1 loss.
    """

    def __init__(self, ssim_weight: float = 0.5, l1_weight: float = 0.5) -> None:
        if ssim is None:
            raise ImportError("pytorch-msssim required for SSIM loss")
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight

    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute SSIM + L1 loss.

        Args:
            pred: Predicted x0 of shape (B, C, H, W).
            target: Target x0 of shape (B, C, H, W).
            mask: Mask of shape (B, H, W) or (B, 1, H, W). True for valid pixels.

        Returns:
            Scalar loss value.
        """
        mask_4d = _expand_mask(mask).float()

        ssim_val: Tensor = ssim(pred * mask_4d, target * mask_4d, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val

        l1_error = (pred - target).abs() * mask_4d
        l1_loss = l1_error.sum() / mask_4d.sum().clamp(min=1)

        loss: Tensor = self.ssim_weight * ssim_loss + self.l1_weight * l1_loss
        return loss


class LossScheduler:
    """Schedule-based interpolation between MSE and SSIM+L1 losses.

    Allows smooth transition from MSE-only training to SSIM+L1 based on step.
    """

    def __init__(
        self,
        mse_loss: MSELoss,
        ssim_l1_loss: SSIML1Loss,
        schedule_fn: Callable[[int], float],
    ) -> None:
        """Initialize loss scheduler.

        Args:
            mse_loss: MSE loss instance.
            ssim_l1_loss: SSIM+L1 loss instance.
            schedule_fn: Maps step -> weight for SSIM+L1 (0=MSE only, 1=SSIM+L1 only).
        """
        self.mse_loss = mse_loss
        self.ssim_l1_loss = ssim_l1_loss
        self.schedule_fn = schedule_fn
        self.step = 0

    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Compute interpolated loss based on current step."""
        w = self.schedule_fn(self.step)
        self.step += 1
        mse = self.mse_loss(pred, target, mask)
        ssim_l1 = self.ssim_l1_loss(pred, target, mask)
        loss: Tensor = (1.0 - w) * mse + w * ssim_l1
        return loss

    def reset(self) -> None:
        """Reset step counter."""
        self.step = 0
