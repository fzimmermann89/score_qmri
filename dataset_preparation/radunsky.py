"""Build radunsky_{t1,m0mag,t2,t2star}.h5 from the subject MATLAB maps.
Get the Session1/Session2 maps via the data links in
https://doi.org/10.1371/journal.pone.0297244 and extract beneath --input-root;
each subject needs mrQ_maps.mat, T2map_EMC.mat and T2_star_map.mat.
The paper’s repository link returned 404; reuse the saved session archives
or request the maps from the authors if that link is still unavailable.
Run: python radunsky.py --input-root RAW --output-dir NEW --limit 1."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
from scipy.io import loadmat


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "x")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


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


def volumes(folder: Path, modalities: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if {"t1", "m0mag"} & set(modalities):
        maps = loadmat(folder / "mrQ_maps.mat")["mrQ_maps"][0, 0]
        m0 = np.flip(np.swapaxes(np.moveaxis(maps["M0"], -1, 0), 1, 2), 1)
        m0 /= np.percentile(m0[m0 > 0], 99.5)
        m0 = np.clip(m0, 0, 1.2)
        t1 = np.flip(np.swapaxes(np.moveaxis(maps["T1"], -1, 0), 1, 2), 1)
        t1 = np.clip(t1, 0, 4)
        mask = torch.stack([clean_mask_2d(m) for m in m0 > 1e-6]).numpy()
        good = mask.mean((-1, -2)) > 0.4
        m0, t1, mask = (a[good][10:] for a in (m0, t1, mask))
        t1[~mask] = 0
        result.update(t1=t1, m0mag=m0)
    if "t2" in modalities:
        result["t2"] = (np.moveaxis(loadmat(folder / "T2map_EMC.mat")["T2map_EMC"], -1, 0)[:-10] / 1000).astype(np.float32)
    if "t2star" in modalities:
        t2 = np.moveaxis(loadmat(folder / "T2_star_map.mat")["T2s_map"], -1, 0)[30:-40] / 1000
        mask = torch.stack([clean_mask_2d(m) for m in t2 > 1e-5]).numpy()
        t2[~mask] = 0
        result["t2star"] = t2.astype(np.float32)
    return {m: result[m] for m in modalities}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=["t1", "m0mag", "t2", "t2star"], default=["t1", "m0mag", "t2", "t2star"])
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folders = select([p.parent for p in args.input_root.rglob("mrQ_maps.mat")], args.subjects, args.limit)
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(args.output_dir / f"radunsky_{m}.h5")) for m in args.modalities}
        for folder in folders:
            for modality, array in volumes(folder, args.modalities).items():
                write_dataset(outputs[modality], folder.stem, array)


if __name__ == "__main__":
    main()
