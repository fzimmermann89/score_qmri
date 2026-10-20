"""Build ibc_relaxo_{t2starmap,s0map}.h5 from IBC relaxometry sourcedata.
Download the relaxometry NIfTIs and JSON sidecars using the IBC data access
instructions at https://individual-brain-charting.github.io/docs/ibc_api.html;
pass the sourcedata root as --input-root. The recovered fit matched the checked historical subject exactly.
Run: python ibc_t2star.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from threadpoolctl import threadpool_limits


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "x")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def load_t2star_data(base_path: Path, sub: str, ses: str) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    mag_path = base_path / f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_part-mag_T2starmap.nii.gz"
    json_path = mag_path.with_suffix("").with_suffix(".json")

    img = nib.load(mag_path)
    mag = img.get_fdata(dtype=np.float32)

    with open(json_path) as f:
        meta = json.load(f)

    if "EchoTimes" in meta:
        te = np.array(meta["EchoTimes"], dtype=np.float32)
    elif "EchoTime1" in meta:
        te = np.array(
            [meta[f"EchoTime{i}"] for i in range(1, mag.shape[-1] + 1)],
            dtype=np.float32,
        )
    else:
        te0 = meta["EchoTime"]
        n_echo = mag.shape[-1]
        delta = meta.get("EchoSpacing", (meta.get("EchoTime2", te0 * 2) - te0))
        te = te0 + np.arange(n_echo) * delta

    return mag, te, img


def historical_t2star_mask(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Executed session224 line6 mask used before its line40 T2star writer."""
    st = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(np.asarray(mask).astype(bool), structure=st)
    mask = ndi.binary_closing(mask, structure=st, iterations=3)
    mask = np.array([ndi.binary_fill_holes(item) for item in mask])
    labels, count = ndi.label(mask, structure=st)
    if count:
        mask = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(mask, dtype=torch.bool)


def create_mask_t2star(mag: np.ndarray | torch.Tensor) -> torch.Tensor:
    mask = mag.mean(-1) > 200
    mask = historical_t2star_mask(mask)
    return mask


def fit_t2star(mag: np.ndarray | torch.Tensor, te: np.ndarray, mask: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-linear T2* fit on (X,Y,Z,echo), with echo times in seconds; return T2* and normalized S0."""
    log_s = np.log(np.clip(mag, 1, None))

    t2star = np.zeros(mag.shape[:-1], dtype=np.float32)
    s0 = np.zeros_like(t2star)

    idx = np.where(mask)
    y = log_s[idx]

    te_mean = te.mean()
    te_c = te - te_mean
    y_mean = y.mean(axis=1, keepdims=True)
    y_c = y - y_mean

    slope = (y_c @ te_c) / (te_c @ te_c + 1e-10)
    intercept = y_mean[:, 0] - slope * te_mean

    r2star = np.clip(-slope, 1e-6, 500)
    t2star[idx] = 1.0 / r2star
    s0[idx] = np.exp(intercept)
    s0_tensor = torch.tensor(s0)
    mask_tensor = torch.stack([torch.tensor(ndi.binary_erosion(m)) for m in mask])
    s0_tensor = s0_tensor.clamp_max(s0_tensor[mask_tensor].quantile(0.95))
    s0_tensor = s0_tensor / s0_tensor[mask_tensor].quantile(0.8)
    s0_tensor = s0_tensor * mask_tensor
    t2star_tensor = torch.tensor(t2star) * mask_tensor
    t2star_tensor = t2star_tensor.clamp_max(0.1)
    s0_tensor = s0_tensor[20:-20]
    t2star_tensor = t2star_tensor[20:-20]

    return t2star_tensor, s0_tensor


def compute_t2star_map(base_path: Path, sub: str, ses: str) -> tuple[torch.Tensor, torch.Tensor]:
    mag, te, _ = load_t2star_data(base_path, sub, ses)
    mag_tensor = torch.tensor(mag).movedim(0, 2)
    mag_tensor = mag_tensor[80:-20]

    mask = create_mask_t2star(mag_tensor)
    t2star, s0 = fit_t2star(mag_tensor, te, mask)
    return t2star, s0


def prepare(input_path: Path, output_dir: Path, subjects: list[str] | None = None, limit: int | None = None) -> list[str]:
    """Write the paired parameter/S0 files for selected subject/session keys."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    entries = {}
    for path in sorted(input_path.rglob("*mag_T2starmap.nii.gz")):
        parts = path.name.split("_")
        sub, ses = (parts[0].split("-")[1], parts[1].split("-")[1])
        name = f"sub-{sub}_ses-{ses}"
        if sub in ("09", "19"):
            continue
        if name in entries:
            raise ValueError(f"Ambiguous duplicate source for {name}")
        entries[name] = (sub, ses)
    names = sorted(entries)
    if subjects is not None:
        missing = set(subjects) - set(names)
        if missing:
            raise ValueError(f"Unknown subject/session keys: {sorted(missing)}")
        names = [name for name in names if name in subjects]
    if limit is not None:
        names = names[:limit]
    if not names:
        raise ValueError("No matching subjects")
    kinds = ("t2starmap", "s0map")
    outputs = [output_dir / f"ibc_relaxo_{kind}.h5" for kind in kinds]
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
    with ExitStack() as stack:
        handles = [stack.enter_context(exclusive_h5(path)) for path in outputs]
        for name in names:
            sub, ses = entries[name]
            arrays = compute_t2star_map(input_path, sub, ses)
            for handle, array in zip(handles, arrays, strict=True):
                write_dataset(handle, name, np.asarray(array))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        prepare(args.input_root, args.output_dir, args.subjects, None if args.all else args.limit)


if __name__ == "__main__":
    main()
