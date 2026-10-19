"""Convert the original Radunsky MATLAB parameter maps to training HDF5.

Source: datacheck/radunsky.py. T1 is in seconds; T2/T2* are converted from ms.
Use clean_h5 afterwards for the final curated slice filter.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from .common import exclusive_h5, write_dataset
from .nifti import clean_mask_2d, select


def volumes(folder: Path, modalities: list[str]) -> dict[str, np.ndarray]:
    result = {}
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
