"""Prepare IXI, Quiero, Rockland and ds005851 from existing NIfTI downloads.

Extracted from datacheck/ixi.py, v12.py, rockland.py and ds5851.py.
Preparation preserves the original numerical operations, including unusual
rate conversions and ineffective non-inplace clamps. Run clean_h5 separately
for the curated slice-filtering stage. See provenance.json for source hashes.
"""

from __future__ import annotations

import argparse
import math
from contextlib import ExitStack
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch

from .common import exclusive_h5, write_dataset


def pad_or_crop(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Historical mrpro center convention, including odd negative padding."""
    pads = []
    for old, new in reversed(list(zip(array.shape[-2:], shape, strict=True))):
        delta = new - old
        left = math.trunc(delta / 2)
        pads.extend([left, delta - left])
    return torch.nn.functional.pad(torch.from_numpy(array), pads).numpy()


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
    tensor /= np.percentile(tensor[tensor > 0.01], 99)
    tensor.clamp_max_(1.4)
    mask = torch.stack([clean_mask_2d(m > 0.05) for m in tensor.abs()])
    good = mask.float().mean((1, 2)) > 0.2
    tensor[~mask] = 0
    return tensor[good][30:].float().numpy()


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
    dataset: str,
    input_root: Path,
    output_dir: Path,
    modalities: list[str],
    subjects: list[str] | None = None,
    limit: int | None = None,
    stage: str = "curated",
) -> None:
    """Write each requested modality into a new HDF5 file."""
    valid = {
        "ixi": {"t1w", "t2w", "pd"},
        "quiero": {"t1map", "t2map", "m0map"},
        "rockland": {"t1w", "t2w", "flair"},
        "ds005851": {"t1map", "t2map", "m0_abs"},
    }
    if not modalities or set(modalities) - valid[dataset]:
        raise ValueError(f"Choose modalities from {sorted(valid[dataset])}")
    if stage not in ("prepared", "curated"):
        raise ValueError("stage must be prepared or curated")
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles = {m: stack.enter_context(exclusive_h5(output_dir / f"{dataset}_{m}.h5")) for m in modalities}
        if dataset == "quiero":
            paths = select(list(input_root.rglob("*test.nii")), subjects, limit)
            for path in paths:
                if nib.load(path).shape[-1] != 3:
                    continue
                name = path.parent.parent.stem + "_" + path.stem
                for modality, array in quiero_volumes(path).items():
                    if modality in handles:
                        if stage == "curated":
                            from .clean_h5 import retained_indices

                            array = array[retained_indices(array)]
                        write_dataset(handles[modality], name, array)
        elif dataset == "ds005851":
            paths = select([p.parent for p in input_root.rglob("*MNI_PD*.*")], subjects, limit)
            for folder in paths:
                for modality, array in ds005851_volumes(folder).items():
                    if modality in handles:
                        write_dataset(handles[modality], folder.parent.stem, array)
        else:
            tags = {"t1w": "T1" if dataset == "ixi" else "T1w", "t2w": "T2" if dataset == "ixi" else "T2w", "pd": "PD", "flair": "FLAIR"}
            for modality in modalities:
                paths = select(list(input_root.rglob(f"*{tags[modality]}*.nii.gz")), subjects, limit)
                for path in paths:
                    array = ixi_volume(path, modality) if dataset == "ixi" else rockland_volume(path)
                    name = path.name.split(".")[0] if dataset == "ixi" else path.stem
                    write_dataset(handles[modality], name, array)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ixi", "quiero", "rockland", "ds005851"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--stage",
        choices=["prepared", "curated"],
        default="curated",
        help="Apply the verified Quiero slice filter before writing; other families are unchanged",
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    prepare(args.dataset, args.input_root, args.output_dir, args.modalities, args.subjects, args.limit, args.stage)


if __name__ == "__main__":
    main()
