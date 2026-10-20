"""Build quiero_{t1map,t2map,m0map}.h5 from the three-channel 4Pub maps.
Download 3DMRFReproducibility.zip from https://zenodo.org/records/3989799
and pass its extracted 4Pub directory as --input-root.
Run: python quiero.py --input-root RAW --output-dir NEW --limit 1.
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

MODALITIES = ("t1map", "t2map", "m0map")
ACTIVE_THRESHOLD = 1e-6
MIN_ACTIVE_PIXELS = 1000


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "x")


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


def clean_mask_3d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Original Quiero volume mask; no dilation."""
    structure = ndi.generate_binary_structure(3, 1)
    result = ndi.binary_opening(np.asarray(mask).astype(bool), structure=structure)
    result = ndi.binary_closing(result, structure=structure, iterations=3)
    result = np.array([ndi.binary_fill_holes(s) for s in result])
    labels, n_labels = ndi.label(result, structure=structure)
    if n_labels:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


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


def quiero_volumes(path: Path) -> dict[str, np.ndarray]:
    """Original 4Pub T1/T2/M0 channels and corner-derived foreground mask."""
    data = torch.tensor(nib.load(path).get_fdata()).moveaxis((0, 1, 2, 3), (2, 3, 1, 0))
    if data.shape[0] != 3:
        raise ValueError(f"Expected three parameter channels: {path}")
    edge = torch.cat(
        [
            data[-1, 20:-20, :10, -10:].ravel(),
            data[-1, 20:-20, -10:, :10].ravel(),
            data[-1, 20:-20, :10, :10].ravel(),
            data[-1, 20:-20, -10:, -10:].ravel(),
        ]
    )
    threshold = edge[edge > 0].quantile(0.8)
    mask = clean_mask_3d(data[-1] > threshold)
    t1, t2, m0 = data.float() * mask.float()[None]
    m0 /= m0[m0 > 0].quantile(0.8)
    # The original m0.clamp_max(2.0) did not assign its result.
    return {"t1map": t1.numpy(), "t2map": t2.numpy(), "m0map": m0.numpy()}


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"quiero_{m}.h5")) for m in modalities or MODALITIES}
        for path in select(list(input_root.rglob("*test.nii")), subjects, limit):
            if nib.load(path).shape[-1] != 3:
                continue
            name = path.parent.parent.stem + "_" + path.stem
            for modality, array in quiero_volumes(path).items():
                if modality in outputs:
                    write_dataset(outputs[modality], name, array[retained_indices(array)])


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
