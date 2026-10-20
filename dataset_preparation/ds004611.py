"""Build ds004611_*.h5 from the downloaded ds004611 NIfTIs.
Get https://openneuro.org/datasets/ds004611 with its git-annex file contents;
pass the dataset root as --input-root. Original masks, units and crops are retained.
Run: python ds004611.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import rearrange

MODALITIES = ("t1map", "t2map", "t1w", "mprage")


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


def load_nifti(path: Path) -> torch.Tensor:
    return rearrange(torch.as_tensor(nib.load(path).get_fdata()), "x y z -> z y x").flip(1)


def first(root: Path, pattern: str) -> Path:
    """Require exactly one matching scan."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {root}; download annex contents first")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {pattern} in {root}: {matches}")
    return matches[0]


def clean_mask_2d(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Original opening, closing, dilation, filling and largest component."""
    st = ndi.generate_binary_structure(2, 1)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=1)
    result = ndi.binary_dilation(result, structure=st, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n = ndi.label(result, structure=st)
    if n:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def subject_4611(root: Path, modality: str) -> Iterator[tuple[str, torch.Tensor]]:
    t1_path = first(root, "*T1map*.nii.gz")
    t1 = load_nifti(t1_path)
    inv1 = load_nifti(first(root, "*inv-1*.nii.gz"))
    inv2 = load_nifti(first(root, "*inv-2*.nii.gz"))
    mask = torch.stack([clean_mask_2d(m) for m in inv1 + inv2 > 40]).float()
    name = t1_path.stem.split("_run")[0]
    if modality == "t1map":
        yield name, (t1 * mask)[5:-1].float() / 1000
    if modality == "t2map":
        t2 = load_nifti(first(root, "*T2map*.nii.gz"))
        yield name, (t2 * mask)[5:-1].float() / 1000
    if modality == "t1w":
        image = load_nifti(first(root, "*UNIT*.nii.gz"))
        image = (image * mask)[5:-1].float()
        yield name, image / image.max()
    if modality == "mprage":
        for i, image in enumerate((inv1, inv2), 1):
            image = (image * mask)[5:-1].float()
            yield f"{name}_{i}", image / image.max()


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    requirements = {
        "t1map": ("*T1map*.nii.gz", "*inv-1*.nii.gz", "*inv-2*.nii.gz"),
        "t2map": ("*T1map*.nii.gz", "*inv-1*.nii.gz", "*inv-2*.nii.gz", "*T2map*.nii.gz"),
        "t1w": ("*T1map*.nii.gz", "*inv-1*.nii.gz", "*inv-2*.nii.gz", "*UNIT*.nii.gz"),
        "mprage": ("*T1map*.nii.gz", "*inv-1*.nii.gz", "*inv-2*.nii.gz"),
    }
    folders = list(input_root.rglob("anat"))
    for modality in modalities or MODALITIES:
        eligible = [root for root in folders if all(any(p.is_file() for p in root.rglob(pattern)) for pattern in requirements[modality])]
        selected = selected_paths(eligible, subjects, limit)
        if not selected:
            raise ValueError(f"No {modality} inputs for selected subjects")
        with exclusive_h5(output_dir / f"ds004611_{modality}.h5") as output:
            for root in selected:
                for name, image in subject_4611(root, modality):
                    write_dataset(output, name, image.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(**vars(args))


if __name__ == "__main__":
    main()
