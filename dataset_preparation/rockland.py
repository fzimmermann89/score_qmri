"""Build rockland_{t1w,t2w,flair}.h5 from NKI-Rockland BIDS anatomy.
Get T1w/T2w/FLAIR NIfTIs via https://fcon_1000.projects.nitrc.org/indi/enhanced/
or the original aws_links.csv; retain the BIDS filenames under --input-root.
Run: python rockland.py --input-root RAW --output-dir NEW.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch

MODALITIES = ("t1w", "t2w", "flair")


@contextmanager
def exclusive_h5(path: str | Path) -> Iterator[h5py.File]:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(str(path), "w-")
    try:
        with handle:
            yield handle
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def select(paths: list[Path], subjects: list[str] | None) -> list[Path]:
    paths = sorted(set(paths))
    if subjects:
        paths = [p for p in paths if any(s in p.parts or p.name.startswith(s + "_") or p.stem == s for s in subjects)]
    if not paths:
        raise ValueError("No matching input files")
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


def rockland_volume(path: Path) -> np.ndarray:
    """Executed IPython session343 revision, before later exploratory edits."""
    image = nib.load(path)
    tensor = torch.tensor(image.get_fdata())
    if (torch.tensor(image.header["pixdim"][1:4]) - 1).abs().max() > 0.01:
        tensor = torch.nn.functional.interpolate(tensor[None, None], scale_factor=image.header["pixdim"][1:4].tolist(), mode="trilinear")[
            0, 0
        ]
    tensor = tensor.moveaxis((0, 1, 2), (2, 1, 0)).flip(1)
    tensor /= tensor.max()
    tensor /= float(np.percentile(tensor[tensor > 0.01], 99))
    tensor.clamp_max_(1.4)
    mask = torch.stack([clean_mask_2d(m > 0.05) for m in tensor.abs()])
    good = mask.float().mean((1, 2)) > 0.2
    tensor[~mask] = 0
    return tensor[good][30:].float().numpy()


def prepare(input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None) -> None:
    tags = {"t1w": "T1w", "t2w": "T2w", "flair": "FLAIR"}
    selected = {m: select(list(input_root.rglob(f"*{tags[m]}*.nii.gz")), subjects) for m in modalities or MODALITIES}
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"rockland_{m}.h5")) for m in selected}
        for modality, paths in selected.items():
            output = outputs[modality]
            for path in paths:
                name = path.stem
                array = rockland_volume(path)
                write_dataset(output, name, array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(**vars(args))


if __name__ == "__main__":
    main()
