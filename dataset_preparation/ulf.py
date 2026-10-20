"""Build ulf_{t1,t2,flair}.h5 from the ULF-EnC 3T reference scans.
Request/download the training data through
https://github.com/khtohidulislam/ULF-EnC-Challenge and extract it;
--input-root must contain subject/3T folders (not enhanced 64mT predictions).
Run: python ulf.py --input-root TRAIN3 --output-dir NEW --limit 1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch


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


def prepare_volume(volume: np.ndarray) -> np.ndarray:
    """Reproduce the original orientation, 16-pixel padding and mask selection."""
    data = np.transpose(volume, (2, 1, 0))
    data = np.flip(data, axis=1)
    data = np.pad(data, ((0, 0), (16, 16), (16, 16)))
    data /= np.percentile(data[data > 0.05 * data.max()], 99.5)
    mask = torch.stack([clean_mask_2d(m) for m in data > 0.05])
    slicemask = mask.float().mean((-1, -2)) > 0.3
    mask = mask[slicemask, ...]
    data = data[slicemask, ...]
    data[~mask] = 0
    return data.astype(np.float32)


def prepare(
    input_root: Path,
    output_dir: Path,
    modalities: Sequence[str] = ("t1", "t2", "flair"),
    subjects: Sequence[str] | None = None,
    limit: int | None = None,
) -> None:
    """Prepare subjects from local 3T references into new exclusive H5 files."""

    if len(set(modalities)) != len(modalities):
        raise ValueError("Select distinct modalities.")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive.")
    folders = sorted(input_root.rglob("3T"))
    if subjects:
        folders = [folder for folder in folders if folder.parent.stem in subjects]
    folders = folders[:limit]
    if not folders:
        raise ValueError("No selected subject/3T input directories.")
    pairs: dict[str, list[tuple[str, Path]]] = {}
    for modality in modalities:
        inputs = [(folder.parent.stem, file) for folder in folders for file in sorted(folder.rglob(f"*{modality.upper()}*.nii.gz"))]
        if not inputs:
            raise ValueError(f"No {modality} NIfTI inputs in selected 3T folders.")
        if len({subject for subject, _ in inputs}) != len(inputs):
            raise ValueError(f"Ambiguous multiple {modality} files per subject.")
        pairs[modality] = inputs
        output = output_dir / f"ulf_{modality}.h5"
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        targets = {modality: stack.enter_context(h5py.File(str(output_dir / f"ulf_{modality}.h5"), "w-")) for modality in modalities}
        for modality, inputs in pairs.items():
            for subject, file in inputs:
                data = prepare_volume(nib.load(file).get_fdata())
                if not len(data):
                    raise ValueError(f"{subject}/{modality}: no slices pass the original foreground threshold.")
                targets[modality].create_dataset(subject, data=data, compression="lzf", shuffle=True, chunks=(1, *data.shape[1:]))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True, type=Path, help="Local ulfenc/train3 directory, containing subject/3T.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modalities", nargs="+", default=["t1", "t2", "flair"], choices=("t1", "t2", "flair"))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    torch.set_num_threads(1)
    prepare(args.input_root, args.output_dir, args.modalities, args.subjects, args.limit)


if __name__ == "__main__":
    main()
