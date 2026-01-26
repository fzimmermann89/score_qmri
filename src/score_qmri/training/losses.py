"""Loss functions for diffusion training."""

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from score_qmri.datasets.kind import is_complex_kind


def create_channel_mask(kind: Tensor) -> Tensor:
    """Create (B, 2, 1, 1) channel mask: [1,1] for complex kinds, [1,0] for real kinds."""
    complex_mask = is_complex_kind(kind)
    mask = torch.ones(kind.shape[0], 2, 1, 1, device=kind.device)
    mask[:, 1] = complex_mask.to(dtype=mask.dtype).view(-1, 1, 1)
    return mask


def _sliding_window(
    x: Tensor,
    window_shape: int | Sequence[int],
    dim: int | Sequence[int] | None = None,
    stride: int | Sequence[int] = 1,
) -> Tensor:
    """Create a sliding window view into a tensor.

    Parameters
    ----------
    x
        Input tensor to create sliding windows from.
    window_shape
        Size of window over each dimension.
    dim
        Dimension(s) to apply to. If None, applies to all dimensions.
    stride
        Stride of the sliding window.

    Returns
    -------
    A view of the tensor with window dimensions at the front and
    original sliding dim replaced by kernel dimensions.
    """
    ndim = x.ndim

    if dim is None:
        dim = tuple(range(ndim))
    elif isinstance(dim, int):
        dim = (dim % ndim,)
    else:
        dim = tuple(ax % ndim for ax in dim)
        if len(set(dim)) != len(dim):
            msg = "Duplicate values in axis are not allowed"
            raise ValueError(msg)

    n_dim = len(dim)
    window_shape_ = (window_shape,) * n_dim if isinstance(window_shape, int) else tuple(window_shape)
    stride_ = (stride,) * n_dim if isinstance(stride, int) else tuple(stride)

    if any(len(param) != n_dim for param in [window_shape_, stride_]):
        msg = f"Length mismatch: window_shape and stride must all have length {n_dim}"
        raise ValueError(msg)

    if any(w <= 0 for w in window_shape_):
        msg = "window_shape must be positive"
        raise ValueError(msg)
    if any(s <= 0 for s in stride_):
        msg = "stride must be positive"
        raise ValueError(msg)

    axis_to_idx = {ax: i for i, ax in enumerate(dim)}
    out_shape = []
    for i, ax in enumerate(dim):
        n_win = (x.shape[ax] - window_shape_[i]) // stride_[i] + 1
        if n_win <= 0:
            axis_size = x.shape[ax]
            msg = f"Dimension {ax} with size {axis_size} is too small for window_size={window_shape_[i]}, stride={stride_[i]}"
            raise ValueError(msg)
        out_shape.append(n_win)

    x_stride = x.stride()
    out_stride = [x_stride[ax] * st for ax, st in zip(dim, stride_, strict=False)]
    for i in range(len(x.shape)):
        if i in axis_to_idx:
            idx = axis_to_idx[i]
            out_shape.append(window_shape_[idx])
            out_stride.append(x_stride[i])
        else:
            out_shape.append(x.shape[i])
            out_stride.append(x_stride[i])

    return x.as_strided(size=out_shape, stride=out_stride)


@torch.compile
def noise_loss(
    pred: Tensor,
    target: Tensor,
    spatial_mask: Tensor | None = None,
    channel_mask: Tensor | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """MSE loss on noise prediction for (B, 2, H, W) tensors.

    Parameters
    ----------
    pred, target
        Prediction and target tensors (B, 2, H, W).
    spatial_mask
        Spatial weight mask (B, H, W) or (B, 1, H, W).
    channel_mask
        Channel mask (B, 2, 1, 1) to ignore imag channel for real kinds.
    """
    diff_sq = (pred - target).square()

    if channel_mask is not None:
        diff_sq = diff_sq * channel_mask

    if spatial_mask is not None:
        mask_f = spatial_mask.float()
        if diff_sq.ndim == 4 and mask_f.ndim == 3:
            mask_f = mask_f.unsqueeze(1)
        diff_sq = diff_sq * mask_f
        n_elements = mask_f.sum(dim=(-2, -1))
        if channel_mask is not None:
            n_elements = n_elements * channel_mask.squeeze(-1).squeeze(-1).sum(dim=1, keepdim=True)
        loss = diff_sq.sum(dim=(1, 2, 3)) / n_elements.sum(dim=1).clamp(min=1)
    else:
        if channel_mask is not None:
            n_channels = channel_mask.squeeze(-1).squeeze(-1).sum(dim=1)
            loss = diff_sq.sum(dim=(1, 2, 3)) / (diff_sq.shape[2] * diff_sq.shape[3] * n_channels).clamp(min=1)
        else:
            loss = diff_sq.mean(dim=(1, 2, 3))

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")


@torch.compile
def l1_loss(
    pred: Tensor,
    target: Tensor,
    spatial_mask: Tensor | None = None,
    channel_mask: Tensor | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """L1 loss for (B, 2, H, W) tensors.

    Parameters
    ----------
    pred, target
        Prediction and target tensors (B, 2, H, W).
    spatial_mask
        Spatial weight mask (B, H, W) or (B, 1, H, W).
    channel_mask
        Channel mask (B, 2, 1, 1) to ignore imag channel for real kinds.
    """
    diff = (pred - target).abs()

    if channel_mask is not None:
        diff = diff * channel_mask

    if spatial_mask is not None:
        mask_f = spatial_mask.float()
        if diff.ndim == 4 and mask_f.ndim == 3:
            mask_f = mask_f.unsqueeze(1)
        diff = diff * mask_f
        n_elements = mask_f.sum(dim=(-2, -1))
        if channel_mask is not None:
            n_elements = n_elements * channel_mask.squeeze(-1).squeeze(-1).sum(dim=1, keepdim=True)
        loss = diff.sum(dim=(1, 2, 3)) / n_elements.sum(dim=1).clamp(min=1)
    else:
        if channel_mask is not None:
            n_channels = channel_mask.squeeze(-1).squeeze(-1).sum(dim=1)
            loss = diff.sum(dim=(1, 2, 3)) / (diff.shape[2] * diff.shape[3] * n_channels).clamp(min=1)
        else:
            loss = diff.mean(dim=(1, 2, 3))

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss.mean(dim=1) if loss.ndim > 1 else loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")


@torch.compile
def _ssim_loss_real(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    data_range: Tensor | float | None = None,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """SSIM loss on real 4D tensors (B, C, H, W)."""
    if pred.shape != target.shape:
        msg = f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        raise ValueError(msg)

    if mask is not None:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1).float()
        elif mask.ndim == 4:
            mask = mask.float()
        else:
            msg = f"Mask must be 3D or 4D, got {mask.ndim}D"
            raise ValueError(msg)

        if (mask < 0).any():
            msg = "Mask contains negative values"
            raise ValueError(msg)

    if mask is not None:
        target, pred, mask = torch.broadcast_tensors(target, pred, mask)
    else:
        target, pred = torch.broadcast_tensors(target, pred)

    window = (window_size, window_size)
    target_window = _sliding_window(target, window_shape=window, dim=(-2, -1)).movedim((0, 1, 2, 3), (2, 3, 0, 1))
    pred_window = _sliding_window(pred, window_shape=window, dim=(-2, -1)).movedim((0, 1, 2, 3), (2, 3, 0, 1))

    mean_target = target_window.mean(dim=(-2, -1))
    mean_pred = pred_window.mean(dim=(-2, -1))

    mean_target_sq = (target_window**2).mean(dim=(-2, -1))
    mean_pred_sq = (pred_window**2).mean(dim=(-2, -1))
    mean_target_pred = (target_window * pred_window).mean(dim=(-2, -1))

    n = target_window.shape[-2:].numel()
    cov_norm = n / (n - 1)
    var_target = (cov_norm * (mean_target_sq - mean_target * mean_target)).clamp(min=0.0)
    var_pred = (cov_norm * (mean_pred_sq - mean_pred * mean_pred)).clamp(min=0.0)
    var_target_pred = cov_norm * (mean_target_pred - mean_target * mean_pred)

    if data_range is None:
        target_max = target_window.amax(dim=(-2, -1)).amax(dim=(-2, -1), keepdim=True)
        target_min = target_window.amin(dim=(-2, -1)).amin(dim=(-2, -1), keepdim=True)
        data_range_ = (target_max - target_min).clamp(min=1e-3)
    else:
        data_range_ = data_range if isinstance(data_range, Tensor) else torch.tensor(data_range)

    c1 = (k1 * data_range_) ** 2
    c2 = (k2 * data_range_) ** 2

    numerator = (2 * mean_target * mean_pred + c1) * (2 * var_target_pred + c2)
    denominator = (mean_target**2 + mean_pred**2 + c1) * (var_target + var_pred + c2)
    ssim_map = numerator / denominator

    if mask is not None:
        mask_window = _sliding_window(mask, window_shape=window, dim=(-2, -1)).movedim((0, 1, 2, 3), (2, 3, 0, 1))
        valid_mask = ~torch.isclose(mask_window, torch.tensor(0.0, device=mask_window.device)).any(dim=(-2, -1), keepdim=True)
        weight = mask_window * valid_mask.float()
        weight = weight.mean(dim=(-2, -1))
        weight /= weight.sum(dim=(-2, -1), keepdim=True) + 1e-8
        ssim_weighted = (ssim_map * weight).sum(dim=(-2, -1))
    else:
        ssim_weighted = ssim_map.mean(dim=(-2, -1))

    if reduction == "mean":
        return 1.0 - ssim_weighted.mean()
    elif reduction == "none":
        loss_per_channel = 1.0 - ssim_weighted
        return loss_per_channel.mean(dim=1) if loss_per_channel.ndim > 1 else loss_per_channel
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")


def ssim_loss(
    pred: Tensor,
    target: Tensor,
    spatial_mask: Tensor | None = None,
    channel_mask: Tensor | None = None,
    data_range: Tensor | float | None = None,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: Literal["mean", "none"] = "mean",
) -> Tensor:
    """SSIM loss for (B, 2, H, W) tensors. Computes per-channel then combines."""
    loss_ch0 = _ssim_loss_real(pred[:, 0:1], target[:, 0:1], spatial_mask, data_range, window_size, k1, k2, "none")
    loss_ch1 = _ssim_loss_real(pred[:, 1:2], target[:, 1:2], spatial_mask, data_range, window_size, k1, k2, "none")

    if channel_mask is not None:
        ch_weights = channel_mask[:, :, 0, 0]
        loss = (loss_ch0 * ch_weights[:, 0:1] + loss_ch1 * ch_weights[:, 1:2]) / ch_weights.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        loss = (loss_ch0 + loss_ch1) / 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean' or 'none'")
