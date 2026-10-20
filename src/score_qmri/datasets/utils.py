"""Dataset utilities."""

import torch


def _pad_or_crop_centered(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    replicate: bool = False,
) -> torch.Tensor:
    """Pad (positive) or crop (negative) to target size, centered.

    Parameters
    ----------
    tensor
        Input tensor of shape (H, W), (C, H, W), or (B, C, H, W).
    target_h
        Target height.
    target_w
        Target width.
    replicate
        If True, use replicate padding instead of zero padding.

    Returns
    -------
    torch.Tensor
        Tensor with spatial dimensions (target_h, target_w).
    """
    h, w = tensor.shape[-2:]
    delta_h = target_h - h
    delta_w = target_w - w
    pad_top = delta_h // 2
    pad_bot = delta_h - pad_top
    pad_left = delta_w // 2
    pad_right = delta_w - pad_left

    match tensor.ndim:
        case 2:
            x = tensor[None, None]
        case 3:
            x = tensor[:, None]
        case 4:
            x = tensor
        case _:
            raise ValueError(f"Expected 2D/3D/4D tensor, got {tensor.ndim}D")

    y = torch.nn.functional.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bot),
        mode="replicate" if replicate else "constant",
    )

    if tensor.ndim == 2:
        return y[0, 0]
    if tensor.ndim == 3:
        return y[:, 0]
    return y
