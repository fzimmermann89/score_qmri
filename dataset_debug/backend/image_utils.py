"""Image rendering helpers for dataset debug viewer."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


if TYPE_CHECKING:
    from matplotlib.figure import Figure


def render_images(
    image: torch.Tensor,
    display_mode: str,
    is_complex: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[str, str | None, str, str | None, dict[str, tuple[float, float]], list[float], list[float] | None]:
    """Render images for display.

    Returns
    -------
    tuple
        (left_base64, right_base64, left_label, right_label, colorbar_ranges, raw_data_left, raw_data_right)
    """
    if is_complex:
        if display_mode == "mag_phase":
            magnitude = image.abs().float()
            phase = torch.angle(image).float()
            left = magnitude
            right = phase
            left_label = "Magnitude"
            right_label = "Phase"
            left_range = (0.0, float(magnitude.max().item()))
            right_range = (-float(np.pi), float(np.pi))
            left_cmap = "gray"
            right_cmap = "twilight"
        else:
            real = image.real.float()
            imag = image.imag.float()
            vabs = max(real.abs().max().item(), imag.abs().max().item())
            left = real
            right = imag
            left_label = "Real"
            right_label = "Imaginary"
            left_range = (-vabs, vabs)
            right_range = (-vabs, vabs)
            left_cmap = "gray"
            right_cmap = "gray"

        render_left_min = vmin if vmin is not None else left_range[0]
        render_left_max = vmax if vmax is not None else left_range[1]
        render_right_min = right_range[0]
        render_right_max = right_range[1]

        left_img = _render_image(left.cpu().numpy(), cmap=left_cmap, vmin=render_left_min, vmax=render_left_max)
        right_img = _render_image(right.cpu().numpy(), cmap=right_cmap, vmin=render_right_min, vmax=render_right_max)
        raw_left = left.cpu().numpy().ravel().tolist()
        raw_right = right.cpu().numpy().ravel().tolist()

        return (
            left_img,
            right_img,
            left_label,
            right_label,
            {"left": (render_left_min, render_left_max), "right": (render_right_min, render_right_max)},
            raw_left,
            raw_right,
        )
    else:
        values = image.float()
        value_min = float(values.min().item())
        value_max = float(values.max().item())
        render_min = vmin if vmin is not None else value_min
        render_max = vmax if vmax is not None else value_max

        left_img = _render_image(values.cpu().numpy(), cmap="gray", vmin=render_min, vmax=render_max)
        raw_left = values.cpu().numpy().ravel().tolist()

        return (
            left_img,
            None,
            "Image",
            None,
            {"left": (render_min, render_max), "right": (render_min, render_max)},
            raw_left,
            None,
        )


def render_mask_overlay(mask: torch.Tensor, color: tuple[int, int, int, int]) -> str:
    """Render a mask overlay with transparency.

    Parameters
    ----------
    mask
        Boolean tensor mask (H, W).
    color
        RGBA color tuple (0-255).
    """
    mask_np = mask.cpu().numpy().astype(bool)
    rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    rgba[mask_np] = np.array(color, dtype=np.uint8)
    
    return _render_rgba(rgba)


def _render_image(data: np.ndarray, cmap: str, vmin: float, vmax: float) -> str:
    """Render a single image using PIL (fast)."""
    range_val = vmax - vmin
    if range_val > 1e-8:
        norm_data = (data - vmin) / range_val
    else:
        norm_data = np.zeros_like(data)
    
    norm_data = np.clip(norm_data, 0.0, 1.0)

    colormap = plt.get_cmap(cmap)
    rgba_data = colormap(norm_data) 
    
    img_uint8 = (rgba_data * 255).astype(np.uint8)
    
    img = Image.fromarray(img_uint8)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _render_rgba(rgba: np.ndarray) -> str:
    """Render RGBA image using PIL (fast)."""
    img = Image.fromarray(rgba, mode="RGBA")
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.0, dpi=150)
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
