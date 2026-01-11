# %%
"""
Debug script to visualize normalization logic for MRI datasets.

Loads ALL datasets from config.yaml matching a given DatasetKind,
warms up RunningNormalizer on shuffled samples from the concatenated dataset,
then visualizes random samples showing original vs normalized vs denormalized.

This script is designed to be run as a cell-based script (e.g. in VSCode) or as a standalone script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from score_qmri.datasets.config import AugmentationConfig
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import (
    ComplexDatasetConfig,
    ConstantSchedule,
    DatasetConfig,
    MagnitudeDatasetConfig,
    MappingDatasetConfig,
    PhaseInSchedule,
    PhaseOutSchedule,
)
from score_qmri.models.normalization import RunningNormalizer

# %% Configuration
# Default parameters (can be overridden by args if run as script)
CONFIG_PATH = "config.yaml"
KIND_NAME = "M0_MAGNITUDE"
NUM_WARMUP_SAMPLES = 200
NUM_VIS_SAMPLES = 10
SEED = 42


# %% Helper functions
def _as_tuple2(x: Any) -> tuple[Any, Any]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (x[0], x[1])
    raise ValueError(f"Expected a 2-element list/tuple, got: {x!r}")


def _parse_schedule(raw: dict[str, Any] | None) -> ConstantSchedule | PhaseInSchedule | PhaseOutSchedule:
    if raw is None:
        return ConstantSchedule()
    schedule_type = raw.get("type")
    if schedule_type == "constant":
        return ConstantSchedule(weight=float(raw.get("weight", 1.0)))
    if schedule_type == "phase_in":
        return PhaseInSchedule(steps=int(raw["steps"]), weight=float(raw.get("weight", 1.0)))
    if schedule_type == "phase_out":
        return PhaseOutSchedule(steps=int(raw["steps"]), weight=float(raw.get("weight", 1.0)))
    raise ValueError(f"Unknown schedule type: {schedule_type!r}")


def _parse_aug(raw: dict[str, Any] | None) -> AugmentationConfig | None:
    if raw is None:
        return None
    raw = dict(raw)
    for key in ["shift_range", "scale_range", "rotation_range", "shear_range", "poly_coeff_range"]:
        if key in raw:
            raw[key] = _as_tuple2(raw[key])
    return AugmentationConfig(**raw)


def _resolve_h5_path(h5_path: str, config_dir: Path) -> str:
    p = Path(h5_path)
    if p.is_absolute():
        return str(p)
    return str((config_dir / p).resolve())


def _parse_dataset_configs(raw: list[dict[str, Any]], config_dir: Path) -> list[DatasetConfig]:
    configs: list[DatasetConfig] = []
    for entry in raw:
        ds_type = entry["type"]
        kind_raw = entry["kind"]
        kind = DatasetKind[kind_raw] if isinstance(kind_raw, str) else DatasetKind(int(kind_raw))

        augmentation_config = _parse_aug(entry.get("augmentation_config"))
        schedule = _parse_schedule(entry.get("schedule"))
        h5_path = _resolve_h5_path(str(entry["h5_path"]), config_dir)
        scaling_factor = float(entry.get("scaling_factor", 1.0))

        if ds_type == "complex":
            mode = entry.get("mode", "flat")
            configs.append(
                ComplexDatasetConfig(
                    h5_path=h5_path,
                    mode=mode,
                    kind=kind,
                    scaling_factor=scaling_factor,
                    augmentation_config=augmentation_config,
                    schedule=schedule,
                )
            )
        elif ds_type == "magnitude":
            configs.append(
                MagnitudeDatasetConfig(
                    h5_path=h5_path,
                    kind=kind,
                    scaling_factor=scaling_factor,
                    augmentation_config=augmentation_config,
                    schedule=schedule,
                )
            )
        elif ds_type == "mapping":
            configs.append(
                MappingDatasetConfig(
                    h5_path=h5_path,
                    kind=kind,
                    scaling_factor=scaling_factor,
                    augmentation_config=augmentation_config,
                    schedule=schedule,
                )
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type!r}")
    return configs


def load_datasets_for_kind(config_path: Path, kind: DatasetKind) -> list[Any]:
    """Load all datasets from config.yaml matching the given kind."""
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_block = cfg.get("data", {})
    init_args = data_block.get("init_args", {})
    raw_ds_cfgs = init_args.get("dataset_configs", [])

    config_dir = config_path.parent
    ds_configs = _parse_dataset_configs(raw_ds_cfgs, config_dir)

    matching = [d for d in ds_configs if d.kind == kind]
    print(f"Found {len(matching)} dataset config(s) for kind {kind.name}")

    datasets = []
    for i, cfg in enumerate(matching):
        try:
            ds = cfg.create()
            print(f"  [{i}] {cfg.h5_path} -> {len(ds)} samples")
            datasets.append(ds)
        except Exception as e:
            print(f"  [{i}] {cfg.h5_path} -> FAILED: {e}")

    return datasets


def prepare_input(img: torch.Tensor) -> torch.Tensor:
    """Convert (B, H, W) image to (B, 2, H, W) for normalizer."""
    if img.is_complex():
        return torch.moveaxis(torch.view_as_real(img), -1, 1)
    return torch.stack([img, torch.zeros_like(img)], dim=1)


def to_display(x: torch.Tensor, is_complex: bool) -> torch.Tensor:
    """Convert (B, 2, H, W) back to (B, H, W) complex or real."""
    if is_complex:
        return torch.view_as_complex(torch.moveaxis(x, 1, -1).contiguous())
    return x[:, 0]


def center_crop(x: torch.Tensor, target_h: int = 256, target_w: int = 256) -> torch.Tensor:
    """Center crop a tensor to target size. Works with (H, W) or (B, H, W) or (B, C, H, W)."""
    h, w = x.shape[-2], x.shape[-1]
    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)
    end_h = start_h + min(target_h, h)
    end_w = start_w + min(target_w, w)
    return x[..., start_h:end_h, start_w:end_w]


# %% Argument parsing (if run as script)
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Visualize normalization for a given DatasetKind")
        parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to config.yaml")
        parser.add_argument("--kind", type=str, default=KIND_NAME, help="DatasetKind name (e.g. M0_MAGNITUDE)")
        parser.add_argument("--warmup", type=int, default=NUM_WARMUP_SAMPLES, help="Number of warmup samples")
        parser.add_argument("--vis", type=int, default=NUM_VIS_SAMPLES, help="Number of visualization samples")
        parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
        args, _ = parser.parse_known_args()

        CONFIG_PATH = args.config
        KIND_NAME = args.kind
        NUM_WARMUP_SAMPLES = args.warmup
        NUM_VIS_SAMPLES = args.vis
        SEED = args.seed
    except Exception:
        pass

# %% Initialize
torch.manual_seed(SEED)
np.random.seed(SEED)

KIND = DatasetKind[KIND_NAME]
config_path = Path(CONFIG_PATH).resolve()

print(f"Config: {config_path}")
print(f"Kind: {KIND.name} (index {int(KIND)}, is_complex={KIND.is_complex})")
print(f"Warmup samples: {NUM_WARMUP_SAMPLES}")
print(f"Visualization samples: {NUM_VIS_SAMPLES}")
print()

# %% Load all datasets for this kind
datasets = load_datasets_for_kind(config_path, KIND)

if not datasets:
    raise RuntimeError(f"No valid datasets found for kind {KIND.name}")

concat_ds = ConcatDataset(datasets)
total_samples = len(concat_ds)
print(f"\nTotal samples across all datasets: {total_samples}")

# %% Initialize normalizer and warm it up
normalizer = RunningNormalizer(num_kinds=len(DatasetKind))
normalizer.train()

print(f"\nWarming up normalizer with {NUM_WARMUP_SAMPLES} samples...")
dataloader = DataLoader(concat_ds, batch_size=1, shuffle=True)

samples_seen = 0
for batch in dataloader:
    img = batch["image"]
    mask = batch["mask"]
    kind_t = batch["kind"]

    x_in = prepare_input(img)
    normalizer.normalize(x_in, mask, kind_t)

    samples_seen += 1
    if samples_seen >= NUM_WARMUP_SAMPLES:
        break

# Print stats for the relevant kind
k_idx = int(KIND)
mean, std = normalizer._get_stats(k_idx, 2)
n_updates = normalizer._get_buffers(k_idx)[2].item()

print(f"\nStats for kind {KIND.name} after {samples_seen} samples:")
print(f"  Mean: {mean.tolist()}")
print(f"  Std:  {std.tolist()}")
print(f"  N:    {n_updates}")

# %% Process visualization samples
print(f"\nProcessing {NUM_VIS_SAMPLES} samples for visualization...")
normalizer.eval()

original_list: list[torch.Tensor] = []
normalized_list: list[torch.Tensor] = []
denormalized_list: list[torch.Tensor] = []
mask_list: list[torch.Tensor] = []

# Sample random indices from the concatenated dataset
vis_idxs = np.random.choice(total_samples, min(NUM_VIS_SAMPLES, total_samples), replace=False)

for idx in vis_idxs:
    sample = concat_ds[int(idx)]
    img: torch.Tensor = sample["image"].unsqueeze(0)  # type: ignore[union-attr]
    mask_t: torch.Tensor = sample["mask"].unsqueeze(0)  # type: ignore[union-attr]
    kind_t: torch.Tensor = sample["kind"].unsqueeze(0)  # type: ignore[union-attr]

    img = center_crop(img)
    mask_t = center_crop(mask_t)

    is_complex = img.is_complex()
    x_in = prepare_input(img)

    # Normalize (eval mode, no stats update)
    x_norm = normalizer.normalize(x_in, mask_t, kind_t)

    # Denormalize
    x_denorm = normalizer.denormalize(x_norm, kind_t)

    # Convert back for visualization
    orig_disp = img[0]
    norm_disp = to_display(x_norm, is_complex)[0]
    denorm_disp = to_display(x_denorm, is_complex)[0]

    original_list.append(orig_disp)
    normalized_list.append(norm_disp)
    denormalized_list.append(denorm_disp)
    mask_list.append(mask_t[0])

# Stack tensors
vis_results = {
    "original": torch.stack(original_list),
    "normalized": torch.stack(normalized_list),
    "denormalized": torch.stack(denormalized_list),
    "mask": torch.stack(mask_list),
}

# %% Plotting
print("Generating visualization plot...")

n_vis = len(vis_results["original"])
is_complex = KIND.is_complex

if is_complex:
    cols = ["Orig Abs", "Orig Real", "Orig Imag", "Norm Abs", "Norm Real", "Norm Imag", "Denorm Abs", "Mask"]
    n_cols = 8
else:
    cols = ["Original", "Normalized", "Denormalized", "Mask"]
    n_cols = 4

fig, axes = plt.subplots(n_vis, n_cols, figsize=(2.5 * n_cols, 2.5 * n_vis), squeeze=False)
plt.subplots_adjust(wspace=0.05, hspace=0.15)

for i in range(n_vis):
    orig = vis_results["original"][i]
    norm = vis_results["normalized"][i]
    denorm = vis_results["denormalized"][i]
    mask = vis_results["mask"][i]

    if is_complex:
        data = [
            orig.abs(),
            orig.real,
            orig.imag,
            norm.abs(),
            norm.real,
            norm.imag,
            denorm.abs(),
            mask.float(),
        ]
    else:
        data = [orig, norm, denorm, mask.float()]

    for j, d in enumerate(data):
        ax = axes[i, j]
        d_np = d.detach().cpu().numpy()
        ax.imshow(d_np, cmap="gray")
        ax.axis("off")

        if i == 0:
            ax.set_title(cols[j], fontsize=9, fontweight="bold")

        # Show min/max
        d_min, d_max = d_np.min(), d_np.max()
        ax.text(
            0.02,
            0.02,
            f"[{d_min:.2g}, {d_max:.2g}]",
            transform=ax.transAxes,
            color="yellow",
            fontsize=7,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

fig.suptitle(f"Normalization debug: {KIND.name} (warmup={samples_seen}, n_updates={n_updates})", fontsize=11)
plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

output_png = f"normalization_debug_{KIND.name}.png"
plt.savefig(output_png, dpi=150)
print(f"Saved visualization to {output_png}")

# %% Save tensors for further debugging
output_pt = f"debug_norm_tensors_{KIND.name}.pt"
torch.save(
    {
        "kind": KIND.name,
        "vis_results": vis_results,
        "stats_mean": mean,
        "stats_std": std,
        "n_updates": n_updates,
        "samples_seen": samples_seen,
    },
    output_pt,
)
print(f"Saved tensors to {output_pt}")
print("Done.")
