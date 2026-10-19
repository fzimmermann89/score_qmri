"""Prepare ULF-labelled 3T reference images from the recovered HPC recipe.

Original HPC IPython session49 cells60/61/62 wrote ulf_t1/ulf_flair/ulf_t2.h5.
T1 cell SHA256: e3f7b2aa09def5257f646143e35d656b0b2548bc3ecffd3ef6f8c6aa9289b0cf.
Inputs are subject/3T/*T1*.nii.gz, *T2*.nii.gz and *FLAIR*.nii.gz beneath
ulfenc/train3, NOT the separate enhanced low-field predictions. Their historical
'ulf' output label is retained for compatibility. This recipe reads local NIfTI
files only; no download or reconstruction is launched. The complete original
cells and cleaning helper are recorded in the preparation audit.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np


def prepare_volume(volume: np.ndarray) -> np.ndarray:
    """Reproduce the original orientation, 16-pixel padding and mask selection."""
    import torch

    from dataset_preparation.raw_mri import clean_mask_2d

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
    import nibabel as nib

    if not modalities or not set(modalities) <= {"t1", "t2", "flair"} or len(set(modalities)) != len(modalities):
        raise ValueError("Select distinct modalities from t1, t2, flair.")
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
        targets = {modality: stack.enter_context(h5py.File(output_dir / f"ulf_{modality}.h5", "x")) for modality in modalities}
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
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.limit is not None and args.limit < 1:
        parser.error("--threads and --limit must be positive.")
    import torch

    torch.set_num_threads(args.threads)
    prepare(args.input_root, args.output_dir, args.modalities, args.subjects, args.limit)


if __name__ == "__main__":
    main()
