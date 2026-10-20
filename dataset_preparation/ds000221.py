"""Build ds000221_*.h5 from the downloaded ds000221 NIfTIs.
Get https://openneuro.org/datasets/ds000221 with its git-annex file contents;
pass the dataset root as --input-root. Original masks, units and crops are retained.
Run: python ds000221.py --input-root RAW --output-dir NEW --limit 1.
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
from einops import rearrange, reduce

MODALITIES = ("gre", "t1w", "t1map", "mprage", "t2w", "flair")


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


def load_nifti(path: Path, *, oriented: bool = True) -> tuple[torch.Tensor, nib.Nifti1Header]:
    """Load with original float64 get_fdata precision and optional ZYX flip."""
    image = nib.load(path)
    data = torch.as_tensor(image.get_fdata())
    if oriented:
        data = rearrange(data, "x y z -> z y x").flip(1)
    return data, image.header


def first(root: Path, pattern: str) -> Path:
    """Require exactly one matching scan."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {root}; download annex contents first")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {pattern} in {root}: {matches}")
    return matches[0]


def clean_mask_3d(mask: torch.Tensor | np.ndarray, *, large: bool = True) -> torch.Tensor:
    """Original v3-v7 radius2 mask or v8-v9 radius1 mask."""
    st = ndi.generate_binary_structure(3, 1)
    if large:
        st = ndi.iterate_structure(st, 2)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=5 if large else 3)
    result = ndi.binary_dilation(result, structure=st, iterations=3 if large else 1)
    result = np.array([ndi.binary_fill_holes(s) for s in result])
    labels, n = ndi.label(result, structure=ndi.generate_binary_structure(3, 1))
    if n:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def clean_mask_221_t1(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    """T1 producer's helper before its later redefinition (IPython205:18)."""
    st = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 2)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=5)
    result = ndi.binary_dilation(result, structure=st, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n = ndi.label(result, structure=ndi.generate_binary_structure(3, 1))
    if n:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def subject_221(root: Path, modalities: set[str]) -> Iterator[tuple[str, str, torch.Tensor]]:
    name = root.stem
    if "gre" in modalities:
        mag, _ = load_nifti(first(root, "*mag_GRE*.nii.gz"), oriented=False)
        phase, _ = load_nifti(first(root, "*phase_GRE*.nii.gz"), oriented=False)
        phase -= phase.min()
        phase /= phase.max()
        phase *= 2 * np.pi
        phase -= np.pi
        image = torch.polar(mag, phase)
        image /= image.abs().max()
        yield "gre", name, rearrange(image, "x y z -> z y x").flip(1).to(torch.complex64)
    if modalities & {"t1w", "t1map", "mprage"}:
        inv1, _ = load_nifti(first(root, "*inv-1_mp2*.nii.gz"))
        inv2, _ = load_nifti(first(root, "*inv-2_mp2*.nii.gz"))
        mask = clean_mask_221_t1(inv1 + inv2 > 40).float()
        if "t1w" in modalities:
            image, _ = load_nifti(first(root, "*T1w*.nii.gz"))
            yield "t1w", name, ((image / image.max()) * mask).float()[70:220]
        if "t1map" in modalities:
            image, _ = load_nifti(first(root, "*T1map*.nii.gz"))
            yield "t1map", name, ((image / 1000) * mask).float()[70:220]
        if "mprage" in modalities:
            for i, image in enumerate((inv1, inv2), 1):
                image -= image.min()
                image /= image.max()
                yield "mprage", f"{name}_{i}", (image * mask).float()[70:220]
    if "t2w" in modalities:
        image, _ = load_nifti(first(root, "*T2w*.nii.gz"))
        yield "t2w", name, (clean_mask_3d(image > 10).float() * image / image.max())[70:210]
    if "flair" in modalities:
        path = first(root, "*FLAIR*.nii.gz")
        image, _ = load_nifti(path)
        if "highres" in path.name:
            image = reduce(image, "(h ph) (w pw) d -> h w d", "mean", ph=2, pw=2)
        else:
            image = reduce(image, "h (w pw) (d pd) -> h w d", "mean", pd=2, pw=2)
        image = image / image.max() * clean_mask_3d(image > 20).float()
        yield "flair", name, image[100:200:5] if "highres" in path.name else image[4:-4]


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    requirements = {
        "gre": ("*mag_GRE*.nii.gz", "*phase_GRE*.nii.gz"),
        "t1w": ("*inv-1_mp2*.nii.gz", "*inv-2_mp2*.nii.gz", "*T1w*.nii.gz"),
        "t1map": ("*inv-1_mp2*.nii.gz", "*inv-2_mp2*.nii.gz", "*T1map*.nii.gz"),
        "mprage": ("*inv-1_mp2*.nii.gz", "*inv-2_mp2*.nii.gz"),
        "t2w": ("*T2w*.nii.gz",),
        "flair": ("*FLAIR*.nii.gz",),
    }
    folders = list(input_root.glob("sub*"))
    for modality in modalities or MODALITIES:
        eligible = [root for root in folders if all(any(p.is_file() for p in root.rglob(pattern)) for pattern in requirements[modality])]
        selected = selected_paths(eligible, subjects, limit)
        if not selected:
            raise ValueError(f"No {modality} inputs for selected subjects")
        with exclusive_h5(output_dir / f"ds000221_{modality}.h5") as output:
            for root in selected:
                for _, name, image in subject_221(root, {modality}):
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
