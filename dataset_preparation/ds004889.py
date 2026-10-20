"""Build ds004889_*.h5 from the downloaded ds004889 NIfTIs.
Get https://openneuro.org/datasets/ds004889 with its git-annex file contents;
pass the dataset root as --input-root. Original masks, units and crops are retained.
Run: python ds004889.py --input-root RAW --output-dir NEW.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import rearrange

MODALITIES = ("t1w", "flair")


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


def selected_paths(paths: Iterable[Path], subjects: list[str] | None = None) -> list[Path]:
    """Match BIDS subject ancestors, preserving stable deterministic ordering."""
    result = sorted(set(paths))
    if subjects:
        result = [p for p in result if set(p.parts) & set(subjects) or p.stem in subjects]
    return result


def load_nifti(path: Path) -> tuple[torch.Tensor, nib.Nifti1Header]:
    image = nib.load(path)
    data = torch.as_tensor(image.get_fdata())
    return data, image.header


def clean_mask_3d(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    st = ndi.generate_binary_structure(3, 1)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=3)
    result = ndi.binary_dilation(result, structure=st, iterations=1)
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


def volume(modality: str, path: Path) -> torch.Tensor | None:
    try:
        image, header = load_nifti(path)
    except Exception as error:
        print(f"Skipping {path}: {error}")
        return None
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
        print(f"Skipping {path}: unsupported quaternion {quaternion}")
        return None
    mask = clean_mask_3d(image > 20)
    image = image / np.percentile(image[mask].numpy(), 99.5)
    image[~mask] = 0
    keep = mask.float().mean((-1, -2)) > 0.3
    keep[: 3 * len(keep) // 8] = False
    keep[-len(keep) // 8 :] = False
    return centered(image[keep], (90, 256, 256)).float()


def prepare(input_root: Path, output_dir: Path, modalities: list[str] | None = None, subjects: list[str] | None = None) -> None:
    selected = {}
    for modality in modalities or MODALITIES:
        pattern = f"*{'T1' if modality == 't1w' else 'FLAIR'}*.nii.gz"
        paths = selected_paths((p for p in input_root.rglob(pattern) if p.is_file()), subjects)
        if not paths:
            raise ValueError(f"No matching {pattern} inputs")
        selected[modality] = paths
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"ds004889_{m}.h5")) for m in selected}
        for modality, paths in selected.items():
            output = outputs[modality]
            for path in paths:
                image = volume(modality, path)
                if image is None:
                    continue
                write_dataset(output, path.stem, image.numpy())


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
