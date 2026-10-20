"""Build ds004889_*.h5 from the downloaded ds004889 NIfTIs.
Get https://openneuro.org/datasets/ds004889 with its git-annex file contents;
pass the dataset root as --input-root. Original masks, units and crops are retained.
Run: python ds004889.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import rearrange

MODALITIES = ("t1w", "flair")


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


def centered(data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Match mrpro.utils.pad_or_crop centered PyTorch pad semantics."""
    pads: list[int] = []
    for current, target in zip(reversed(data.shape), reversed(shape), strict=True):
        delta = target - current
        left = math.trunc(delta / 2)
        pads.extend((left, delta - left))
    return torch.nn.functional.pad(data, pads)


def volume(modality: str, path: Path) -> torch.Tensor:
    image, header = load_nifti(path, oriented=False)
    image = torch.nn.functional.interpolate(image[None, None], scale_factor=header["pixdim"][1:4].tolist(), mode="trilinear")[0, 0]
    quaternion = header.get_qform_quaternion()
    if modality == "flair":
        if torch.tensor(image.shape).argmin() == 0:
            image = rearrange(image, "x y z -> z y x").flip(1)
        else:
            image = rearrange(image, "x y z -> y z x")
    elif quaternion[0] > 0.9:
        image = rearrange(image, "x y z -> z y x").flip(1)
    elif quaternion[2] > 0.9:
        image = rearrange(image, "x y z -> z y x")
    else:
        raise ValueError(f"Original v9 rejected quaternion {quaternion}: {path}")
    mask = clean_mask_3d(image > 20, large=False)
    image = image / np.percentile(image[mask].numpy(), 99.5)
    image[~mask] = 0
    keep = mask.float().mean((-1, -2)) > 0.3
    keep[: 3 * len(keep) // 8] = False
    keep[-len(keep) // 8 :] = False
    return centered(image[keep], (90, 256, 256)).float()


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"ds004889_{m}.h5")) for m in modalities or MODALITIES}
        for modality, output in outputs.items():
            pattern = f"*{'T1' if modality == 't1w' else 'FLAIR'}*.nii.gz"
            paths = selected_paths((p for p in input_root.rglob(pattern) if p.is_file()), subjects, limit)
            if not paths:
                raise ValueError(f"No matching {pattern} inputs")
            for path in paths:
                write_dataset(output, path.stem, volume(modality, path).numpy())


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
