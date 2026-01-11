# %%
"""
Debug script to visualize normalization logic for MRI datasets.
This script is designed to be run as a cell-based script (e.g. in VSCode) or as a standalone script.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from score_qmri.datasets.config import AugmentationConfig
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import ComplexDatasetConfig
from score_qmri.models.normalization import RunningNormalizer

# %% Setup parameters
# Default parameters (can be overridden by args if run as script)
CONFIG_PATH = "config.yaml"
H5_PATH = "./datasets/fastmri_t1.h5"  # Update this to match your local/server path if needed
KIND = DatasetKind.COMPLEX_T1W_COIL_COMBINED
MODE = "combined"
NUM_WARMUP_SAMPLES = 100
NUM_VIS_SAMPLES = 10
SEED = 42

# %% Argument parsing (if run as script)
if __name__ == "__main__" and not hasattr(np, "ps"):  # hasattr hack to check if interactive
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--h5_path", type=str, default=H5_PATH)
        parser.add_argument("--mode", type=str, default=MODE)
        parser.add_argument("--warmup", type=int, default=NUM_WARMUP_SAMPLES)
        parser.add_argument("--vis", type=int, default=NUM_VIS_SAMPLES)
        args, _ = parser.parse_known_args()
        
        H5_PATH = args.h5_path
        MODE = args.mode
        NUM_WARMUP_SAMPLES = args.warmup
        NUM_VIS_SAMPLES = args.vis
        print(f"Running with: h5={H5_PATH}, mode={MODE}, warmup={NUM_WARMUP_SAMPLES}")
    except Exception:
        pass

# %% Initialize components
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Initializing dataset from {H5_PATH}...")
dataset_config = ComplexDatasetConfig(
    h5_path=H5_PATH,
    mode=MODE,
    kind=KIND,
    # augmentation_config=AugmentationConfig(scale_range=(0.8, 1.2)) # Optional: Enable to test augs
    augmentation_config=None
)

try:
    dataset = dataset_config.create()
    print(f"Dataset loaded. Length: {len(dataset)}")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    print("Please ensure H5_PATH points to a valid file.")
    exit(1)

normalizer = RunningNormalizer(num_kinds=len(DatasetKind))
normalizer.train()

# %% Warmup normalizer
print("Warming up normalizer stats...")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

samples_seen = 0
for batch in dataloader:
    img = batch["image"]
    mask = batch["mask"]
    kind = batch["kind"]
    
    # Prepare input (B, C, H, W)
    if img.is_complex():
        x_in = torch.moveaxis(torch.view_as_real(img), -1, 1)
    else:
        x_in = torch.stack([img, torch.zeros_like(img)], dim=1)
        
    # Update stats
    normalizer.normalize(x_in, mask, kind)
    
    samples_seen += 1
    if samples_seen >= NUM_WARMUP_SAMPLES:
        break

# Print stats for the relevant kind
k_idx = int(KIND)
mean, std = normalizer._get_stats(k_idx, 2)
print(f"\nStats for kind {KIND.name}:")
print(f"Mean: {mean.tolist()}")
print(f"Std:  {std.tolist()}")
print(f"N:    {normalizer._get_buffers(k_idx)[2].item()}")

# %% Process visualization samples
print(f"\nProcessing {NUM_VIS_SAMPLES} samples for visualization...")
normalizer.eval()

vis_results = {
    "original": [],
    "normalized": [],
    "denormalized": [],
    "mask": []
}

idxs = np.random.choice(len(dataset), NUM_VIS_SAMPLES, replace=False)

for idx in idxs:
    sample = dataset[int(idx)]
    img = sample["image"].unsqueeze(0)  # (1, H, W)
    mask = sample["mask"].unsqueeze(0)  # (1, H, W)
    kind = sample["kind"].unsqueeze(0)  # (1,)
    
    # Prepare
    if img.is_complex():
        x_in = torch.moveaxis(torch.view_as_real(img), -1, 1)
    else:
        x_in = torch.stack([img, torch.zeros_like(img)], dim=1)
        
    # Normalize
    x_norm = normalizer.normalize(x_in, mask, kind)
    
    # Denormalize
    x_denorm = normalizer.denormalize(x_norm, kind)
    
    # Convert back to complex for storage/plotting
    orig_c = img[0]
    
    if img.is_complex():
        norm_c = torch.view_as_complex(torch.moveaxis(x_norm, 1, -1).contiguous())[0]
        denorm_c = torch.view_as_complex(torch.moveaxis(x_denorm, 1, -1).contiguous())[0]
    else:
        # Should not happen for ComplexDataset, but for completeness
        norm_c = x_norm[0, 0] + 1j * x_norm[0, 1]
        denorm_c = x_denorm[0, 0] + 1j * x_denorm[0, 1]

    vis_results["original"].append(orig_c)
    vis_results["normalized"].append(norm_c)
    vis_results["denormalized"].append(denorm_c)
    vis_results["mask"].append(mask[0])

# Stack tensors
for k in vis_results:
    vis_results[k] = torch.stack(vis_results[k])

# %% Plotting
print("Generating visualization plot...")
fig, axes = plt.subplots(NUM_VIS_SAMPLES, 7, figsize=(20, 3 * NUM_VIS_SAMPLES))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

cols = ["Orig Abs", "Orig Real", "Orig Imag", "Norm Abs", "Norm Real", "Norm Imag", "Denorm Abs"]

for i in range(NUM_VIS_SAMPLES):
    orig = vis_results["original"][i]
    norm = vis_results["normalized"][i]
    denorm = vis_results["denormalized"][i]
    mask = vis_results["mask"][i]
    
    # Data for columns
    data = [
        orig.abs(), orig.real, orig.imag,
        norm.abs(), norm.real, norm.imag,
        denorm.abs()
    ]
    
    for j, d in enumerate(data):
        ax = axes[i, j] if NUM_VIS_SAMPLES > 1 else axes[j]
        
        # Use simple gray map
        im = ax.imshow(d.numpy(), cmap="gray")
        ax.axis("off")
        
        # Titles on first row
        if i == 0:
            ax.set_title(cols[j], fontsize=10, fontweight="bold")
            
        # Add min/max text
        d_min, d_max = d.min().item(), d.max().item()
        ax.text(0.05, 0.05, f"[{d_min:.2f}, {d_max:.2f}]", 
                transform=ax.transAxes, color="yellow", fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))

# Save plot
plt.tight_layout()
plt.savefig("normalization_debug.png", dpi=150)
print("Saved visualization to normalization_debug.png")

# %% Save tensors
print("Saving tensors to debug_norm_tensors.pt...")
torch.save({
    "vis_results": vis_results,
    "stats_mean": mean,
    "stats_std": std,
    "samples_seen": samples_seen
}, "debug_norm_tensors.pt")
print("Done.")
