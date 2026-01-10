#!/usr/bin/env python
"""Check all datasets defined in config for data quality issues."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from score_qmri.datasets.complex import ComplexImageDataset
from score_qmri.datasets.magnitude import MagnitudeImageDataset
from score_qmri.datasets.mapping import MappingDataset
from score_qmri.datasets.kind import DatasetKind


def print_histogram(values: np.ndarray, bins: int = 50, width: int = 60) -> None:
    """Print ASCII histogram."""
    if len(values) == 0:
        print("  [No data]")
        return

    values = values[np.isfinite(values)]
    if len(values) == 0:
        print("  [All values are NaN/Inf]")
        return

    hist, edges = np.histogram(values, bins=bins)
    max_count = hist.max() if hist.max() > 0 else 1

    for i, count in enumerate(hist):
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len
        lo, hi = edges[i], edges[i + 1]
        print(f"  [{lo:10.4f}, {hi:10.4f}): {bar} ({count})")


def check_h5_raw(h5_path: str, mode: str = "flat") -> dict:
    """Check raw H5 file without dataset wrapper."""
    stats = {
        "path": h5_path,
        "exists": False,
        "subjects": 0,
        "total_slices": 0,
        "nan_slices": 0,
        "inf_slices": 0,
        "zero_slices": 0,
        "all_values": [],
        "bad_samples": [],
    }

    if not Path(h5_path).exists():
        print(f"  ❌ File not found: {h5_path}")
        return stats

    stats["exists"] = True

    with h5py.File(h5_path, "r") as f:
        if mode in ("combined", "coilwise") and mode in f:
            container = f[mode]
        else:
            container = f

        subject_keys = [k for k in container if isinstance(container[k], h5py.Dataset)]
        stats["subjects"] = len(subject_keys)

        for subj_key in subject_keys:
            data = container[subj_key][:]
            
            if data.ndim == 3:
                for z in range(data.shape[0]):
                    slice_data = data[z]
                    check_slice(slice_data, subj_key, z, None, stats)
            elif data.ndim == 4:
                for z in range(data.shape[0]):
                    for c in range(data.shape[1]):
                        slice_data = data[z, c]
                        check_slice(slice_data, subj_key, z, c, stats)

    return stats


def check_slice(slice_data: np.ndarray, subj: str, z: int, coil: int | None, stats: dict) -> None:
    """Check a single slice for issues."""
    stats["total_slices"] += 1

    if np.iscomplexobj(slice_data):
        mag = np.abs(slice_data)
    else:
        mag = slice_data

    mask = mag > 1e-6
    masked_vals = mag[mask]

    if len(masked_vals) == 0:
        stats["zero_slices"] += 1
        stats["bad_samples"].append((subj, z, coil, "all_zero"))
        return

    stats["all_values"].extend(masked_vals.flatten().tolist())

    has_nan = np.isnan(masked_vals).any()
    has_inf = np.isinf(masked_vals).any()

    if has_nan:
        stats["nan_slices"] += 1
        stats["bad_samples"].append((subj, z, coil, "has_nan"))

    if has_inf:
        stats["inf_slices"] += 1
        stats["bad_samples"].append((subj, z, coil, "has_inf"))

    vmin, vmax = masked_vals.min(), masked_vals.max()
    vmean, vstd = masked_vals.mean(), masked_vals.std()

    if vstd < 1e-10:
        stats["bad_samples"].append((subj, z, coil, f"constant_value={vmean:.4f}"))

    if vmax > 1e6:
        stats["bad_samples"].append((subj, z, coil, f"extreme_max={vmax:.2e}"))

    if vmean > 1e4:
        stats["bad_samples"].append((subj, z, coil, f"high_mean={vmean:.2e}"))


def check_dataset_wrapper(config: dict, base_path: Path) -> dict:
    """Check dataset using the actual dataset wrapper."""
    ds_type = config.get("type", "complex")
    h5_path = str(base_path / config["h5_path"])
    kind = DatasetKind[config.get("kind", "T1W_MAGNITUDE")]

    if ds_type == "complex":
        mode = config.get("mode", "flat")
        ds = ComplexImageDataset(h5_path, mode=mode, kind=kind, augmentation_config=None)
    elif ds_type == "magnitude":
        ds = MagnitudeImageDataset(h5_path, kind=kind, augmentation_config=None)
    elif ds_type == "mapping":
        scaling = config.get("scaling_factor", 1.0)
        ds = MappingDataset(h5_path, scaling_factor=scaling, kind=kind, augmentation_config=None)
    else:
        return {"error": f"Unknown type: {ds_type}"}

    stats = {
        "type": ds_type,
        "kind": kind.name,
        "num_samples": len(ds),
        "nan_samples": 0,
        "inf_samples": 0,
        "zero_samples": 0,
        "all_values": [],
        "bad_samples": [],
    }

    print(f"  Checking {len(ds)} samples...")

    for i in range(len(ds)):
        sample = ds[i]
        img = sample["image"]
        mask = sample["mask"]

        if img.is_complex():
            mag = img.abs()
        else:
            mag = img

        masked_vals = mag[mask].numpy()

        if len(masked_vals) == 0:
            stats["zero_samples"] += 1
            stats["bad_samples"].append((i, "all_masked_out"))
            continue

        stats["all_values"].extend(masked_vals.tolist())

        if np.isnan(masked_vals).any():
            stats["nan_samples"] += 1
            stats["bad_samples"].append((i, "has_nan"))

        if np.isinf(masked_vals).any():
            stats["inf_samples"] += 1
            stats["bad_samples"].append((i, "has_inf"))

        vmin, vmax = masked_vals.min(), masked_vals.max()
        vmean, vstd = masked_vals.mean(), masked_vals.std()

        if vstd < 1e-10:
            stats["bad_samples"].append((i, f"constant={vmean:.4f}"))

        if vmax > 100:
            stats["bad_samples"].append((i, f"max={vmax:.2e}"))

        if vmean > 10:
            stats["bad_samples"].append((i, f"mean={vmean:.2e}"))

        if i % 1000 == 0 and i > 0:
            print(f"    ... processed {i}/{len(ds)}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Check dataset quality")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--raw", action="store_true", help="Check raw H5 files without dataset wrapper")
    parser.add_argument("--hist-bins", type=int, default=30, help="Histogram bins")
    parser.add_argument("--max-bad", type=int, default=20, help="Max bad samples to print per dataset")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_path = config_path.parent
    dataset_configs = config.get("data", {}).get("init_args", {}).get("dataset_configs", [])

    if not dataset_configs:
        print("No datasets found in config")
        sys.exit(1)

    print(f"Found {len(dataset_configs)} dataset configs\n")
    print("=" * 80)

    all_issues = []

    for i, ds_config in enumerate(dataset_configs):
        h5_path = ds_config.get("h5_path", "")
        ds_type = ds_config.get("type", "complex")
        kind = ds_config.get("kind", "UNKNOWN")
        mode = ds_config.get("mode", "flat")
        scaling = ds_config.get("scaling_factor", 1.0)

        print(f"\n[{i+1}/{len(dataset_configs)}] {kind}")
        print(f"  Type: {ds_type}, Mode: {mode}, Scaling: {scaling}")
        print(f"  Path: {h5_path}")

        full_path = base_path / h5_path
        if not full_path.exists():
            print(f"  ❌ FILE NOT FOUND")
            all_issues.append((kind, h5_path, "file_not_found"))
            continue

        if args.raw:
            stats = check_h5_raw(str(full_path), mode=mode)
        else:
            try:
                stats = check_dataset_wrapper(ds_config, base_path)
            except Exception as e:
                print(f"  ❌ Error loading dataset: {e}")
                all_issues.append((kind, h5_path, str(e)))
                continue

        if not stats.get("all_values"):
            print("  ⚠️  No valid data found")
            continue

        values = np.array(stats["all_values"])
        finite_vals = values[np.isfinite(values)]

        print(f"\n  📊 Statistics (n={len(finite_vals):,} pixels):")
        print(f"     Min:    {finite_vals.min():.6f}")
        print(f"     Max:    {finite_vals.max():.6f}")
        print(f"     Mean:   {finite_vals.mean():.6f}")
        print(f"     Std:    {finite_vals.std():.6f}")
        print(f"     Median: {np.median(finite_vals):.6f}")

        percentiles = [1, 5, 25, 75, 95, 99]
        pct_vals = np.percentile(finite_vals, percentiles)
        print(f"     Percentiles: " + ", ".join(f"p{p}={v:.4f}" for p, v in zip(percentiles, pct_vals)))

        n_nan = stats.get("nan_samples", stats.get("nan_slices", 0))
        n_inf = stats.get("inf_samples", stats.get("inf_slices", 0))
        n_zero = stats.get("zero_samples", stats.get("zero_slices", 0))
        n_total = stats.get("num_samples", stats.get("total_slices", 0))

        if n_nan > 0 or n_inf > 0 or n_zero > 0:
            print(f"\n  ⚠️  Issues:")
            if n_nan > 0:
                print(f"     NaN samples: {n_nan}/{n_total}")
            if n_inf > 0:
                print(f"     Inf samples: {n_inf}/{n_total}")
            if n_zero > 0:
                print(f"     Zero/empty samples: {n_zero}/{n_total}")

        bad_samples = stats.get("bad_samples", [])
        if bad_samples:
            print(f"\n  🔴 Bad samples ({len(bad_samples)} total):")
            for sample in bad_samples[:args.max_bad]:
                print(f"     {sample}")
            if len(bad_samples) > args.max_bad:
                print(f"     ... and {len(bad_samples) - args.max_bad} more")

            all_issues.append((kind, h5_path, f"{len(bad_samples)} bad samples"))

        print(f"\n  📈 Histogram:")
        print_histogram(finite_vals, bins=args.hist_bins, width=50)

        print("-" * 80)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_issues:
        print(f"\n⚠️  Found issues in {len(all_issues)} datasets:")
        for kind, path, issue in all_issues:
            print(f"  - {kind}: {issue}")
    else:
        print("\n✅ All datasets look clean!")


if __name__ == "__main__":
    main()
