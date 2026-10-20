"""Build mrf_nist_{t1map,t2map,m0map}.h5 from NIST MRF DICOM series.
Download MRF_invivo_anon.zip from https://data.nist.gov/od/id/mds2-3379
and extract it beneath --input-root. The historical first-five-file selection
uses the native filesystem order, as in the original script.
Run: python mrf_nist.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch

MODALITIES = ("t1map", "t2map", "m0map")


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


def centered(data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Match mrpro.utils.pad_or_crop centered PyTorch pad semantics."""
    pads: list[int] = []
    for current, target in zip(reversed(data.shape), reversed(shape), strict=True):
        delta = target - current
        left = math.trunc(delta / 2)
        pads.extend((left, delta - left))
    return torch.nn.functional.pad(data, pads)


def select_roots(paths: Iterator[Path], subjects: list[str] | None, limit: int | None) -> list[Path]:
    """Select subject ancestors as well as series/session names."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    roots = sorted(set(paths))
    if subjects:
        roots = [p for p in roots if set(p.parts) & set(subjects) or p.stem in subjects]
    return roots[:limit] if limit is not None else roots


def nist_series(root: Path, modality: str) -> torch.Tensor:
    """Read the original five DICOM candidates and sort by SliceLocation."""
    import pydicom

    files = sorted(list(root.rglob("*.IMA"))[:5])
    images = []
    for path in files:
        dcm = pydicom.dcmread(path)
        image = torch.tensor(dcm.pixel_array, dtype=torch.float32)
        mask = clean_mask_2d(image > 0) * (image > 0)
        if mask.sum() < 2000:
            continue
        images.append((float(dcm.SliceLocation), image * mask / 1000))
    if not images:
        raise ValueError(f"No retained slices in {root}")
    image = torch.stack([x[1] for x in sorted(images, key=lambda x: x[0])])
    image = centered(image, (image.shape[0], 224, 224))
    if modality == "m0map":
        image /= image.max()
    return image


def prepare(
    input_root: Path,
    output_dir: Path,
    modalities: list[str] | None = None,
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> None:
    with ExitStack() as stack:
        for modality in modalities or MODALITIES:
            output = stack.enter_context(exclusive_h5(output_dir / f"mrf_nist_{modality}.h5"))
            tag = {"t1map": "T1", "t2map": "T2", "m0map": "M0"}[modality]
            roots = select_roots((p for p in input_root.rglob(f"*{tag}*") if p.is_dir()), subjects, limit)
            if not roots:
                raise ValueError(f"No {tag} DICOM series found")
            for root in roots:
                write_dataset(output, "_".join(root.parts[-2:]), nist_series(root, modality).numpy())


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
