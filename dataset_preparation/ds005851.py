"""Build ds005851_{t1map,t2map,m0_abs}.h5 from MNI R1/R2/PD derivatives.
Download https://openneuro.org/datasets/ds005851, including annex contents,
and pass the dataset root as --input-root. Preserve the original rate conversions.
Run: python ds005851.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch

MODALITIES = ("t1map", "t2map", "m0_abs")


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def select(paths: list[Path], subjects: list[str] | None, limit: int | None) -> list[Path]:
    """Filter by BIDS ancestor or filename without changing volume contents."""
    paths = sorted(set(paths))
    if subjects:
        paths = [p for p in paths if any(s in p.parts or p.name.startswith(s + "_") or p.stem == s for s in subjects)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError("No matching input files; check input root, subjects and downloaded annex content")
    return paths


def clean_mask_2d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Original opening, closing, dilation, hole fill and largest component."""
    structure = ndi.generate_binary_structure(2, 1)
    result = ndi.binary_opening(np.asarray(mask).astype(bool), structure=structure)
    result = ndi.binary_closing(result, structure=structure, iterations=1)
    result = ndi.binary_dilation(result, structure=structure, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n_labels = ndi.label(result, structure=structure)
    if n_labels:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def ds005851_volumes(folder: Path) -> dict[str, np.ndarray]:
    """Original rate conventions: T1=1000/R1 and T2=1e-3/R2."""
    arrays = [nib.load(next(folder.rglob(pattern))).get_fdata() for pattern in ("*MNI_R1*.nii.gz", "*MNI_R2*.nii.gz", "*MNI_PD*.nii.gz")]
    r1, r2, pd = [torch.tensor(d).moveaxis((0, 1, 2), (2, 1, 0)).flip(1) for d in arrays]
    pd /= pd.max()
    pd /= np.percentile(pd[pd > 0.01], 99)
    pd.clamp_max_(1.5)
    mask = torch.stack([clean_mask_2d(m > 0.05) for m in pd.abs()])
    good = mask.float().mean((1, 2)) > 0.2
    t1 = torch.nan_to_num(1000 / r1, nan=0, posinf=0, neginf=0).clamp(min=0, max=4)
    t2 = torch.nan_to_num(1e-3 / r2, nan=0, posinf=0, neginf=0).clamp(min=0, max=1)
    for tensor in (t1, t2, pd):
        tensor[~mask] = 0
    # Historical _m0_abs is stored as complex64 with zero imaginary channel.
    return {"t1map": t1[good].float().numpy(), "t2map": t2[good].float().numpy(), "m0_abs": pd[good].numpy().astype(np.complex64)}


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"ds005851_{m}.h5")) for m in modalities or MODALITIES}
        folders = select([p.parent for p in input_root.rglob("*MNI_PD*.*")], subjects, limit)
        for folder in folders:
            for modality, array in ds005851_volumes(folder).items():
                if modality in outputs:
                    write_dataset(outputs[modality], folder.parent.stem, array)


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
