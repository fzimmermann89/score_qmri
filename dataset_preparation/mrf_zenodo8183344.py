"""Build mrf_zenodo8183344_{t1map,t2map,t1w,t2w}.h5.
Download and extract https://zenodo.org/records/8183344; --input-root must
contain MR? sessions with original/reposition MPRAGE, T1, T2 and TSE NIfTIs.
Run: python mrf_zenodo8183344.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import rearrange

MODALITIES = ("t1map", "t2map", "t1w", "t2w")
ACTIVE_THRESHOLD = 1e-6
MIN_ACTIVE_PIXELS = 1000


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


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


def select_roots(paths: Iterator[Path], subjects: list[str] | None, limit: int | None) -> list[Path]:
    """Select subject ancestors as well as series/session names."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    roots = sorted(set(paths))
    if subjects:
        roots = [p for p in roots if set(p.parts) & set(subjects) or p.stem in subjects]
    return roots[:limit] if limit is not None else roots


def first(root: Path, pattern: str) -> Path:
    """Require exactly one matching scan."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {root}; download annex contents first")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {pattern} in {root}: {matches}")
    return matches[0]


def load_nifti(path: Path, *, oriented: bool = True) -> tuple[torch.Tensor, nib.Nifti1Header]:
    """Load with original float64 get_fdata precision and optional ZYX flip."""
    image = nib.load(path)
    data = torch.as_tensor(image.get_fdata())
    if oriented:
        data = rearrange(data, "x y z -> z y x").flip(1)
    return data, image.header


def zenodo_subject(root: Path, which: str, modalities: set[str]) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Original v10 quantitative seconds and independent image masks/crops."""
    name = "_".join([*root.parts[-2:], which])
    mprage, _ = load_nifti(first(root, f"*{which}*mprage.nii"))
    mask = torch.stack([clean_mask_2d(m > 10) for m in mprage])
    for modality in ("t1map", "t2map"):
        if modality in modalities:
            image, _ = load_nifti(first(root, f"*{which}*{modality[:2]}.nii"))
            yield modality, name, (image * mask.float() / 1000)[7:-5].float()
    if "t1w" in modalities:
        yield "t1w", name, (mprage * mask.float() / mprage.max())[3:-5].float()
    if "t2w" in modalities:
        image, _ = load_nifti(first(root, f"*{which}*tse.nii"))
        mask = torch.stack([clean_mask_2d(m > 50) for m in image])
        yield "t2w", name, (image * mask.float() / image.max())[3:-5].float()


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


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    wanted = set(modalities or MODALITIES)
    roots = select_roots(input_root.rglob("MR?"), subjects, limit)
    if not roots:
        raise ValueError("No MR? sessions found")
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"mrf_zenodo8183344_{m}.h5")) for m in sorted(wanted)}
        for root in roots:
            for which in ("original", "reposition"):
                for modality, key, image in zenodo_subject(root, which, wanted):
                    array = image.numpy()
                    write_dataset(outputs[modality], key, array[retained_indices(array)])


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
