"""Build ds003967_t1map_cropped.h5 from the downloaded ds003967 NIfTIs.
Get https://openneuro.org/datasets/ds003967 with its git-annex file contents;
Pass the downloaded clone root as --input-root, retaining git-annex object names
used as historical H5 keys. Original masks, units and crops are retained.
Run: python ds003967.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from einops import rearrange


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def selected_paths(paths: Iterable[Path], subjects: list[str] | None = None, limit: int | None = None) -> list[Path]:
    """Match BIDS subject ancestors, preserving stable deterministic ordering."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    result = sorted(set(paths))
    if subjects:
        result = [p for p in result if set(p.parts) & set(subjects) or p.stem in subjects]
    return result[:limit] if limit is not None else result


def volume(path: Path) -> torch.Tensor:
    image = torch.as_tensor(nib.load(path).get_fdata())
    image = rearrange(image, "x y z -> y x z")
    image = torch.nn.functional.interpolate(image[None, None], scale_factor=0.6, mode="trilinear")[0, 0]
    return (image / 1000).float()


def prepare(input_root: Path, output_dir: Path, subjects: list[str] | None = None, limit: int | None = None) -> None:
    paths = selected_paths((p for p in input_root.rglob("*.nii.gz") if p.is_file()), subjects, limit)
    if not paths:
        raise ValueError("No NIfTI inputs found")
    with exclusive_h5(output_dir / "ds003967_t1map_cropped.h5") as output:
        for path in paths:
            write_dataset(output, path.stem, volume(path).numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(**vars(args))


if __name__ == "__main__":
    main()
