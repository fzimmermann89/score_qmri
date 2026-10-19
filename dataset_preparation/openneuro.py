"""OpenNeuro recipes from datacheck/v3.py–v9.py and executed IPython history.
Preserve masks, units, interpolation precision and dataset-specific crops."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import rearrange, reduce

from .common import exclusive_h5, write_dataset

MODALITIES = {
    "ds000221": ("gre", "t1w", "t1map", "mprage", "t2w", "flair"),
    "ds002685": ("t1w", "t2w", "flair"),
    "ds002785": ("t1w",),
    "ds002790": ("t1w",),
    "ds003967": ("t1map_cropped",),
    "ds004611": ("t1map", "t2map", "t1w", "mprage"),
    "ds004889": ("t1w", "flair"),
}


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


def clean_mask_3d(mask: torch.Tensor | np.ndarray, *, large: bool = True) -> torch.Tensor:
    """Original v3-v7 radius2 mask or v8-v9 radius1 mask."""
    st = ndi.generate_binary_structure(3, 1)
    if large:
        st = ndi.iterate_structure(st, 2)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=5 if large else 3)
    result = ndi.binary_dilation(result, structure=st, iterations=3 if large else 1)
    result = np.array([ndi.binary_fill_holes(s) for s in result])
    labels, n = ndi.label(result, structure=ndi.generate_binary_structure(3, 1))
    if n:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def clean_mask_221_t1(mask: torch.Tensor | np.ndarray) -> torch.Tensor:
    """T1 producer's helper before its later redefinition (IPython205:18)."""
    st = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 2)
    result = ndi.binary_opening(np.asarray(mask, dtype=bool), structure=st)
    result = ndi.binary_closing(result, structure=st, iterations=5)
    result = ndi.binary_dilation(result, structure=st, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n = ndi.label(result, structure=ndi.generate_binary_structure(3, 1))
    if n:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def load_nifti(path: Path, *, oriented: bool = True) -> tuple[torch.Tensor, nib.Nifti1Header]:
    """Load with original float64 get_fdata precision and optional ZYX flip."""
    image = nib.load(path)
    data = torch.as_tensor(image.get_fdata())
    if oriented:
        data = rearrange(data, "x y z -> z y x").flip(1)
    return data, image.header


def first(root: Path, pattern: str) -> Path:
    """Reject absent/ambiguous matches rather than silently select another scan."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern} in {root}; download annex contents first")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {pattern} in {root}: {matches}")
    return matches[0]


def centered(data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Match mrpro.utils.pad_or_crop centered PyTorch pad semantics."""
    pads: list[int] = []
    for current, target in zip(reversed(data.shape), reversed(shape), strict=True):
        delta = target - current
        left = math.trunc(delta / 2)
        pads.extend((left, delta - left))
    return torch.nn.functional.pad(data, pads)


def subject_221(root: Path, modalities: set[str]) -> Iterator[tuple[str, str, torch.Tensor]]:
    name = root.stem
    if "gre" in modalities:
        mag, _ = load_nifti(first(root, "*mag_GRE*.nii.gz"), oriented=False)
        phase, _ = load_nifti(first(root, "*phase_GRE*.nii.gz"), oriented=False)
        phase -= phase.min()
        phase /= phase.max()
        phase *= 2 * np.pi
        phase -= np.pi
        image = torch.polar(mag, phase)
        image /= image.abs().max()
        yield "gre", name, rearrange(image, "x y z -> z y x").flip(1).to(torch.complex64)
    if modalities & {"t1w", "t1map", "mprage"}:
        inv1, _ = load_nifti(first(root, "*inv-1_mp2*.nii.gz"))
        inv2, _ = load_nifti(first(root, "*inv-2_mp2*.nii.gz"))
        mask = clean_mask_221_t1(inv1 + inv2 > 40).float()
        if "t1w" in modalities:
            image, _ = load_nifti(first(root, "*T1w*.nii.gz"))
            yield "t1w", name, ((image / image.max()) * mask).float()[70:220]
        if "t1map" in modalities:
            image, _ = load_nifti(first(root, "*T1map*.nii.gz"))
            yield "t1map", name, ((image / 1000) * mask).float()[70:220]
        if "mprage" in modalities:
            for i, image in enumerate((inv1, inv2), 1):
                image -= image.min()
                image /= image.max()
                yield "mprage", f"{name}_{i}", (image * mask).float()[70:220]
    if "t2w" in modalities:
        image, _ = load_nifti(first(root, "*T2w*.nii.gz"))
        yield "t2w", name, (clean_mask_3d(image > 10).float() * image / image.max())[70:210]
    if "flair" in modalities:
        path = first(root, "*FLAIR*.nii.gz")
        image, _ = load_nifti(path)
        if "highres" in path.name:
            image = reduce(image, "(h ph) (w pw) d -> h w d", "mean", ph=2, pw=2)
        else:
            image = reduce(image, "h (w pw) (d pd) -> h w d", "mean", pd=2, pw=2)
        image = image / image.max() * clean_mask_3d(image > 20).float()
        yield "flair", name, image[100:200:5] if "highres" in path.name else image[4:-4]


def subject_4611(root: Path, modalities: set[str]) -> Iterator[tuple[str, str, torch.Tensor]]:
    t1_path = first(root, "*T1map*.nii.gz")
    t1, _ = load_nifti(t1_path)
    inv1, _ = load_nifti(first(root, "*inv-1*.nii.gz"))
    inv2, _ = load_nifti(first(root, "*inv-2*.nii.gz"))
    mask = torch.stack([clean_mask_2d(m) for m in inv1 + inv2 > 40]).float()
    name = t1_path.stem.split("_run")[0]
    if "t1map" in modalities:
        yield "t1map", name, (t1 * mask)[5:-1].float() / 1000
    if "t2map" in modalities:
        paths = sorted(root.rglob("*T2map*.nii.gz"))
        if paths:
            t2, _ = load_nifti(first(root, "*T2map*.nii.gz"))
            yield "t2map", name, (t2 * mask)[5:-1].float() / 1000
    if "t1w" in modalities:
        image, _ = load_nifti(first(root, "*UNIT*.nii.gz"))
        image = (image * mask)[5:-1].float()
        yield "t1w", name, image / image.max()
    if "mprage" in modalities:
        for i, image in enumerate((inv1, inv2), 1):
            image = (image * mask)[5:-1].float()
            yield "mprage", f"{name}_{i}", image / image.max()


def file_recipe(dataset: str, modality: str, path: Path) -> torch.Tensor:
    """Apply an individual v4/v5/v6/v9 transform without writing files."""
    image, header = load_nifti(path, oriented=False)
    if dataset == "ds002685":
        image = rearrange(image, "x y z -> y x z")
        if modality == "flair":
            image = reduce(image, "z (y two) x -> z y x", "mean", two=2)
        mask = clean_mask_3d(image > 20)
        if modality == "flair":
            image /= np.percentile(image[mask].numpy(), 99.5)
            image = image.float()
            image[~mask] = 0
            # Original clamp_max was out of place and discarded.
            keep = mask.float().mean((-1, -2)) > 0.2
            keep[:50] = False
            return image[keep]
        image = image * mask.float()
        return (image / image.max())[100:220].float()
    if dataset in {"ds002785", "ds002790"}:
        image = rearrange(image, "x y z -> z y x").flip(1)
        image = image / image.max() * clean_mask_3d(image > 20).float()
        return image[80:-20].float() if dataset == "ds002785" else image[40:-60].float()
    if dataset == "ds003967":
        image = rearrange(image, "x y z -> y x z")
        image = torch.nn.functional.interpolate(image[None, None], scale_factor=0.6, mode="trilinear")[0, 0]
        return (image / 1000).float()
    if dataset == "ds004889":
        # Saved v9 has '.to + list()'; v8 confirms intended .tolist().
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
            raise ValueError(f"Original v9 rejected quaternion {quaternion}: {path}")
        mask = clean_mask_3d(image > 20, large=False)
        image = image / np.percentile(image[mask].numpy(), 99.5)
        image[~mask] = 0
        keep = mask.float().mean((-1, -2)) > 0.3
        keep[: 3 * len(keep) // 8] = False
        keep[-len(keep) // 8 :] = False
        return centered(image[keep], (90, 256, 256)).float()
    raise ValueError(dataset)


def selected_paths(paths: Iterable[Path], subjects: list[str] | None = None, limit: int | None = None) -> list[Path]:
    """Match BIDS subject ancestors, preserving stable deterministic ordering."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    result = sorted(set(paths))
    if subjects:
        result = [p for p in result if set(p.parts) & set(subjects) or p.stem in subjects]
    return result[:limit] if limit is not None else result


def prepare(
    dataset: str,
    input_root: Path,
    output_dir: Path,
    *,
    subjects: list[str] | None = None,
    limit: int | None = None,
    modalities: list[str] | None = None,
) -> None:
    """Write selected recovered modalities to new HDF5 files only."""
    if dataset not in MODALITIES:
        raise ValueError(f"No recovered recipe for {dataset}")
    wanted = set(modalities or MODALITIES[dataset])
    if not wanted or wanted - set(MODALITIES[dataset]):
        raise ValueError(f"Unavailable modalities {wanted}; recovered: {MODALITIES[dataset]}")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"{dataset}_{m}.h5")) for m in sorted(wanted)}
        if dataset in {"ds000221", "ds004611"}:
            paths: Iterable[Path] = input_root.glob("sub*") if dataset == "ds000221" else input_root.rglob("anat")
            roots = selected_paths(paths, subjects=subjects, limit=limit)
            if not roots:
                raise ValueError(f"No selected subjects in {input_root}")
            for root in roots:
                recipe = subject_221 if dataset == "ds000221" else subject_4611
                for modality, name, image in recipe(root, wanted):
                    write_dataset(outputs[modality], name, image.numpy())
        else:
            for modality in sorted(wanted):
                pattern = (
                    {"t1w": "*ses-00_T1w*.nii.gz", "t2w": "*ses-00*T2w*.nii.gz", "flair": "*FLAIR*.nii.gz"}.get(modality, "*.nii.gz")
                    if dataset == "ds002685"
                    else (f"*{'T1' if modality == 't1w' else 'FLAIR'}*.nii.gz" if dataset == "ds004889" else "*.nii.gz")
                )
                paths = selected_paths((p for p in input_root.rglob(pattern) if p.is_file()), subjects=subjects, limit=limit)
                if not paths:
                    raise ValueError(f"No selected {pattern} in {input_root}")
                for path in paths:
                    image = file_recipe(dataset, modality, path)
                    write_dataset(outputs[modality], path.stem, image.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(MODALITIES))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--modalities", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(**vars(args))


if __name__ == "__main__":
    main()
