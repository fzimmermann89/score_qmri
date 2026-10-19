"""Slice filter from scripts/cleanup_h5.py; stream to a new file, dropping attributes."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from dataset_preparation.common import axial_chunks, exclusive_h5

ACTIVE_THRESHOLD = 1e-6
MIN_ACTIVE_PIXELS = 1000


def retained_indices(
    dataset: h5py.Dataset | np.ndarray, threshold: float = ACTIVE_THRESHOLD, min_pixels: int = MIN_ACTIVE_PIXELS
) -> list[int]:
    """Return the exact original >=1000 active-pixel selection."""
    if dataset.ndim not in (3, 4):
        raise ValueError(f"Expected (Z,H,W) or (Z,C,H,W): {dataset.shape}")
    valid = []
    for index in range(dataset.shape[0]):
        active = np.abs(dataset[index]) > threshold
        count = int(np.max(np.sum(active, axis=(-2, -1))) if active.ndim == 3 else np.sum(active))
        if count >= min_pixels:
            valid.append(index)
    return valid


def _copy_selected(source: h5py.Dataset, target: h5py.Group, name: str, indices: list[int]) -> None:
    shape = (len(indices), *source.shape[1:])
    output = target.create_dataset(name, shape=shape, dtype=source.dtype, chunks=axial_chunks(shape), compression="lzf", shuffle=True)
    for destination, original in enumerate(indices):
        output[destination] = source[original]


def clean_file(
    input_path: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, dict[str, int]]:
    """Create a new filtered file, optionally selecting a bounded subject subset."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    with h5py.File(input_path, "r") as source:
        groups = [name for name in ("coilwise", "combined") if name in source and isinstance(source[name], h5py.Group)]
        reference = source["combined"] if "combined" in groups else source[groups[0]] if groups else source
        names = sorted(name for name in reference if isinstance(reference[name], h5py.Dataset))
        if not groups and any(isinstance(source[name], h5py.Group) for name in source):
            raise ValueError("Unsupported nested schema; use the Brainweb packing recipe for class volumes")
        if subjects is not None:
            missing = set(subjects) - set(names)
            if missing:
                raise ValueError(f"Unknown subjects: {sorted(missing)}")
            names = [name for name in names if name in subjects]
        if limit is not None:
            names = names[:limit]
        if not names:
            raise ValueError("No subjects selected")
        report = {}
        with exclusive_h5(output_path) as target:
            for name in groups:
                target.create_group(name)
            for name in names:
                indices = retained_indices(reference[name])
                report[name] = {"input_slices": reference[name].shape[0], "output_slices": len(indices)}
                if not indices:
                    continue
                if groups:
                    for group in groups:
                        if name in source[group]:
                            _copy_selected(source[group][name], target[group], name, indices)
                else:
                    _copy_selected(source[name], target, name, indices)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output file; existing paths are refused")
    parser.add_argument("--subjects", nargs="+", help="Exact HDF5 subject keys")
    parser.add_argument("--limit", type=int, help="Only the first N selected subjects")
    args = parser.parse_args()
    for name, counts in clean_file(args.input, args.output, args.subjects, args.limit).items():
        print(f"{name}: {counts['input_slices']} -> {counts['output_slices']} slices")


if __name__ == "__main__":
    main()
