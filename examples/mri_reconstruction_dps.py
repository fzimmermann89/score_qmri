# %% [markdown]
# # DPS MRI Reconstruction Example
#
# This example demonstrates Diffusion Posterior Sampling (DPS) for MRI reconstruction
# using mrpro operators and a pretrained diffusion model.
#
# DPS combines an unconditional diffusion prior with likelihood guidance to solve
# the inverse problem: k-space → image.

# ruff: noqa: E402
# %%
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel
from score_qmri.models.unet import UNet
from score_qmri.sampling import DPSConfig, DPSSampler

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# %%
# Configuration
CHECKPOINT_PATH = Path("checkpoints/diffusion_model.ckpt")  # Update with your checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DPS parameters (VE/EDM sampler)
NUM_STEPS = 100
GUIDANCE_SCALE = 0.5

# %%
# Load the diffusion model (VE/EDM framework)
if CHECKPOINT_PATH.exists():
    model = DiffusionModel.from_checkpoint(CHECKPOINT_PATH)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
else:
    diffusion = DiffusionConfig()
    unet = UNet(
        base_channels=64,
        channel_mult=(1, 2, 4, 8, 16),
        cond_dim=diffusion.cond_dim,
        attention_levels=(3, 4),
    )
    model = DiffusionModel(unet=unet, diffusion=diffusion)
    print("No checkpoint found - using randomly initialized model (for demonstration)")

model = model.to(DEVICE).eval()

# %%
# Create synthetic data for demonstration
# In practice, load real k-space data using mrpro

H, W = 256, 256
B = 1

# Create a simple phantom image
x, y = torch.meshgrid(
    torch.linspace(-1, 1, H),
    torch.linspace(-1, 1, W),
    indexing="ij",
)
phantom = torch.zeros(H, W, dtype=torch.complex64)
phantom[(x**2 + y**2) < 0.5] = torch.tensor(1.0 + 0.0j)
phantom[(x**2 + y**2) < 0.2] = torch.tensor(0.5 + 0.0j)
phantom = phantom.to(DEVICE)

# Reshape to mrpro format: (B, 1, 1, H, W)
phantom_mrpro = phantom.unsqueeze(0).unsqueeze(0).unsqueeze(0)

# %%
# Create undersampling mask (Cartesian, random)
acceleration = 4
mask = torch.zeros(1, 1, 1, H, W, device=DEVICE)

# Keep center lines (ACS region)
acs_width = W // 8
center_start = W // 2 - acs_width // 2
center_end = W // 2 + acs_width // 2
mask[..., center_start:center_end] = 1.0

# Random undersampling for remaining lines
remaining_lines = W - acs_width
target_lines = W // acceleration - acs_width
if target_lines > 0:
    indices = torch.randperm(remaining_lines)[:target_lines]
    outer_indices = torch.cat(
        [
            torch.arange(0, center_start),
            torch.arange(center_end, W),
        ]
    )
    selected = outer_indices[indices % len(outer_indices)]
    mask[..., selected] = 1.0

print(f"Acceleration: {acceleration}x, Sampled {mask.sum().item() / mask.numel() * 100:.1f}% of k-space")

# %%
# Create forward operator using mrpro (if available)
# Falls back to simple FFT if mrpro not installed

try:
    from mrpro.operators import FourierOp

    # mrpro FourierOp for single-coil
    fourier_op = FourierOp.from_kdata(...)  # Configure based on your kdata
    print("Using mrpro FourierOp")
    forward_op = fourier_op

except ImportError:
    # Fallback: simple masked FFT operator
    class SimpleMaskedFourierOp:
        """Simple masked Fourier operator for demonstration."""

        def __init__(self, mask: torch.Tensor) -> None:
            self.mask = mask

        def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            k = torch.fft.fft2(x, norm="ortho")
            return (k * self.mask,)

        @property
        def H(self) -> Any:
            return _SimpleMaskedFourierOpAdjoint(self.mask)

    class _SimpleMaskedFourierOpAdjoint:
        def __init__(self, mask: torch.Tensor) -> None:
            self.mask = mask

        def __call__(self, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return (torch.fft.ifft2(y * self.mask, norm="ortho"),)

        @property
        def H(self) -> Any:
            return SimpleMaskedFourierOp(self.mask)

    forward_op: Any = SimpleMaskedFourierOp(mask)  # type: ignore[no-redef]
    print("Using simple masked FFT operator (mrpro not installed)")

# %%
# Generate undersampled k-space measurement
kspace_full = torch.fft.fft2(phantom_mrpro, norm="ortho")
kspace_undersampled = kspace_full * mask

# Zero-filled reconstruction (baseline)
zero_filled = torch.fft.ifft2(kspace_undersampled, norm="ortho")

# %%
# Run DPS reconstruction (VE/EDM-based DPS with Karras sigma schedule)
config = DPSConfig(
    num_steps=NUM_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    normalize_gradient=True,
    gradient_clipping=1.0,
)

sampler = DPSSampler(model, config)

print(f"Running DPS with {NUM_STEPS} steps, guidance_scale={GUIDANCE_SCALE}...")
with torch.no_grad():
    reconstruction = sampler.sample(
        forward_op=forward_op,
        measurement=kspace_undersampled,
        kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
    )

print("DPS reconstruction complete!")

# %%
# Visualize results


def show_complex(img: torch.Tensor, title: str, ax: Axes) -> None:
    """Display magnitude of complex image."""
    mag = img.squeeze().abs().cpu().numpy()
    ax.imshow(mag, cmap="gray")
    ax.set_title(title)
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(16, 4))

show_complex(phantom_mrpro, "Ground Truth", axes[0])
show_complex(zero_filled, "Zero-Filled", axes[1])
show_complex(reconstruction, "DPS Reconstruction", axes[2])

# Error map
error = (reconstruction.squeeze() - phantom_mrpro.squeeze()).abs()
axes[3].imshow(error.cpu().numpy(), cmap="hot", vmax=0.3)
axes[3].set_title("Error Map")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("dps_reconstruction.png", dpi=150)
plt.show()

# %%
# Compute metrics


def psnr(x: Tensor, y: Tensor) -> float:
    mse = ((x - y).abs() ** 2).mean()
    max_val = y.abs().max()
    return (10 * torch.log10(max_val**2 / mse)).item()


def nmse(x: Tensor, y: Tensor) -> float:
    return (((x - y).abs() ** 2).sum() / (y.abs() ** 2).sum()).item()


print("\nReconstruction Metrics:")
print(f"Zero-filled PSNR: {psnr(zero_filled, phantom_mrpro):.2f} dB")
print(f"DPS PSNR: {psnr(reconstruction, phantom_mrpro):.2f} dB")
print(f"Zero-filled NMSE: {nmse(zero_filled, phantom_mrpro):.4f}")
print(f"DPS NMSE: {nmse(reconstruction, phantom_mrpro):.4f}")
