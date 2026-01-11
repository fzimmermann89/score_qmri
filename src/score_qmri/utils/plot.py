"""Plotting utilities for MRI visualization."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from torch import Tensor


def plot_samples_grid(
    samples: Tensor,
    output_path: Path | str | None = None,
    title: str | None = None,
) -> None:
    """Plot samples in an auto-square grid.

    Parameters
    ----------
    samples
        Tensor of shape (N, H, W), complex or real.
    output_path
        If provided, save to PNG. Otherwise show interactive window.
    title
        Optional title for the figure.
    """
    n = samples.shape[0]
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    samples_np = samples.abs().cpu().numpy() if samples.is_complex() else samples.cpu().numpy()

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < n:
            ax.imshow(samples_np[i], cmap="gray")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to: {output_path}")
    else:
        plt.show()
