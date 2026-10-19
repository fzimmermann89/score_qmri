"""Shapes and subject exclusions from scripts/preprocess_h5.py; write a new file."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset_preparation.common import axial_chunks, exclusive_h5


@dataclass
class NormalizeConfig:
    filename: str
    target_shape: tuple[int, int]
    skip_subjects: list[str] | None = None


CONFIGS = [
    NormalizeConfig("fastmri_t1.h5", (320, 320)),
    NormalizeConfig("fastmri_t2.h5", (320, 320)),
    NormalizeConfig("fastmri_flair.h5", (320, 320)),
    NormalizeConfig("ds004611_mprage.h5", (320, 320)),
    NormalizeConfig("ds004611_t1map.h5", (320, 320)),
    NormalizeConfig("ds004611_t1w.h5", (320, 320)),
    NormalizeConfig("ds004611_t2map.h5", (320, 320)),
    NormalizeConfig("ds002685_flair.h5", (256, 192)),
    NormalizeConfig(
        "ds002685_t2w.h5",
        (256, 176),
        skip_subjects=["sub-01_ses-00_acq-tse_T2w.nii", "sub-02_ses-00_acq-tse_T2w.nii"],
    ),
    NormalizeConfig("ibc_relaxo_s0map.h5", (256, 160), skip_subjects=["sub-12_ses-17"]),
    NormalizeConfig("ibc_relaxo_t2starmap.h5", (256, 160), skip_subjects=["sub-12_ses-17"]),
]


def pad_or_crop_centered(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad (positive) or crop (negative) array to target shape, centered."""
    h, w = arr.shape[-2:]

    delta_h = target_h - h
    delta_w = target_w - w

    pad_top = delta_h // 2
    pad_bot = delta_h - pad_top
    pad_left = delta_w // 2
    pad_right = delta_w - pad_left

    tensor = torch.from_numpy(arr)
    result = torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bot))
    return result.numpy()


def normalize_file(
    source_path: str | Path,
    output_path: str | Path,
    shape: tuple[int, int] | None = None,
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Create a new shape-normalized file, retaining historical subject skips."""
    source_path = Path(source_path)
    recipe = next((cfg for cfg in CONFIGS if cfg.filename == source_path.name), None)
    if shape is None:
        if recipe is None:
            raise ValueError("Unknown filename: specify --shape explicitly")
        shape = recipe.target_shape
    if min(shape) <= 0 or (limit is not None and limit <= 0):
        raise ValueError("shape and limit must be positive")
    skip = set(recipe.skip_subjects or []) if recipe else set()
    with h5py.File(source_path, "r") as source:
        groups = [name for name in ("coilwise", "combined") if name in source and isinstance(source[name], h5py.Group)]
        names = sorted(set().union(*(set(source[group]) for group in groups))) if groups else sorted(source)
        if subjects is not None:
            missing = set(subjects) - set(names)
            if missing:
                raise ValueError(f"Unknown subjects: {sorted(missing)}")
            names = [name for name in names if name in subjects]
        names = [name for name in names if name not in skip]
        if limit is not None:
            names = names[:limit]
        if not names:
            raise ValueError("No subjects remain after selection and historical skips")
        with exclusive_h5(output_path) as target:
            for group in groups:
                target.create_group(group)
            group_names: list[str | None] = [*groups] if groups else [None]
            for group_name in group_names:
                origin = source[group_name] if group_name else source
                destination = target[group_name] if group_name else target
                for name in names:
                    if name not in origin:
                        continue
                    ds = origin[name]
                    if not isinstance(ds, h5py.Dataset) or ds.ndim not in (3, 4):
                        raise ValueError(f"Unsupported dataset schema: {ds.name}")
                    dimensions = (*ds.shape[:-2], *shape)
                    output = destination.create_dataset(
                        name, shape=dimensions, dtype=ds.dtype, chunks=axial_chunks(dimensions), compression="lzf", shuffle=True
                    )
                    for index in range(ds.shape[0]):
                        output[index] = pad_or_crop_centered(ds[index], *shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    normalize_file(args.input, args.output, tuple(args.shape) if args.shape else None, args.subjects, args.limit)


if __name__ == "__main__":
    main()
