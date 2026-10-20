"""Build ixi_{t1w,t2w,pd}.h5 from IXI NIfTI scans.
Download T1/T2/PD archives from https://brain-development.org/ixi-dataset/
and extract them beneath --input-root. Retain the original filenames.
Run: python ixi.py --input-root RAW --output-dir NEW --limit 1.
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

MODALITIES = ("t1w", "t2w", "pd")


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


def ixi_volume(path: Path, modality: str) -> np.ndarray:
    """Original latest T1 resampling, and recovered native T2/PD branches."""
    image = nib.load(path)
    data = np.array(image.get_fdata())
    data -= np.percentile(data[0, :50, :50], 1)
    data /= np.percentile(data[data > 0], 99.5)
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if modality == "t1w":
        tensor = torch.nn.functional.interpolate(tensor[None, None], scale_factor=image.header["pixdim"][1:4].tolist(), mode="trilinear")[
            0, 0
        ].moveaxis((0, 1, 2), (1, 0, 2))
    else:
        tensor = tensor.moveaxis((0, 1, 2), (2, 1, 0)).flip(1)
    mask = torch.stack([clean_mask_2d(m) for m in tensor > 0.05])
    good = mask.float().mean((-1, -2)) > 0.2
    tensor[~mask] = 0
    return tensor[good].numpy()


def prepare(
    input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None, limit: int | None = None
) -> None:
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"ixi_{m}.h5")) for m in modalities or MODALITIES}
        tags = {"t1w": "T1", "t2w": "T2", "pd": "PD"}
        for modality, output in outputs.items():
            paths = select(list(input_root.rglob(f"*{tags[modality]}*.nii.gz")), subjects, limit)
            for path in paths:
                name = path.name.split(".")[0]
                array = ixi_volume(path, modality)
                write_dataset(output, name, array)


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
