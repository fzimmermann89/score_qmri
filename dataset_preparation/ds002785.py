"""Build ds002785_t1w.h5 from the downloaded ds002785 NIfTIs.
Get https://openneuro.org/datasets/ds002785 with its git-annex file contents;
Pass the downloaded clone root as --input-root, retaining git-annex object names
used as historical H5 keys. The final active-slice filter is included.
Run: python ds002785.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
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
        result = [p for p in result if set(p.parts) & set(subjects) or p.stem in subjects or p.resolve().stem in subjects]
    return result[:limit] if limit is not None else result


def clean_mask_3d(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    st = ndi.generate_binary_structure(3, 1)
    st = ndi.iterate_structure(st, 2)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=5)
    result = ndi.binary_dilation(result, structure=st, iterations=3)
    result = np.array([ndi.binary_fill_holes(s) for s in result])
    labels, n = ndi.label(result, structure=ndi.generate_binary_structure(3, 1))
    if n:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def volume(path: Path) -> torch.Tensor:
    source = nib.load(path)
    if len(source.shape) != 3:
        raise ValueError(f"Expected a 3D T1w image: {path}")
    image = torch.as_tensor(source.get_fdata())
    image = rearrange(image, "x y z -> z y x").flip(1)
    image = image / image.max() * clean_mask_3d(image > 20).float()
    image = image[80:-20].float()
    return image[(image.abs() > 1e-6).sum((-2, -1)) >= 1000]


def prepare(input_root: Path, output_dir: Path, subjects: list[str] | None = None, limit: int | None = None) -> None:
    paths = selected_paths((p for p in input_root.rglob("*_T1w.nii.gz") if ".git" not in p.parts and p.is_file()), subjects, limit)
    if not paths:
        raise ValueError("No NIfTI inputs found")
    with exclusive_h5(output_dir / "ds002785_t1w.h5") as output:
        for path in paths:
            image = volume(path).numpy()
            # The training files contain both BIDS and annex-object keys for each scan.
            for name in dict.fromkeys((path.stem, path.resolve().stem)):
                write_dataset(output, name, image)


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
