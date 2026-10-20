"""Build ibc_relaxo_{t1map,m0map}.h5 from IBC relaxometry sourcedata.
Download the relaxometry NIfTIs and JSON sidecars using the IBC data access
instructions at https://individual-brain-charting.github.io/docs/ibc_api.html;
pass the sourcedata root as --input-root. The recovered fit still differs slightly from the historical T1/M0 H5.
Run: python ibc_t1.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from nilearn.image import resample_to_img
from threadpoolctl import threadpool_limits


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def clean_mask_3d(mask: np.ndarray) -> torch.Tensor:
    mask = np.asarray(mask)
    st = ndi.generate_binary_structure(3, 1)
    m = ndi.binary_opening(mask.astype(bool), structure=st, iterations=2)
    m = ndi.binary_closing(m, structure=st, iterations=3)
    ms = []
    for s in m:
        s = ndi.binary_fill_holes(s)

        lab, n = ndi.label(s, structure=ndi.generate_binary_structure(2, 1))
        if n:
            s = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        ms.append(torch.tensor(s, dtype=torch.bool))
    return torch.stack(ms, dim=0)


def load_vfa_data(base_path: Path, sub: str, ses: str, min_fa: int = 5) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    pattern = f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_acq-fa*_T1map.nii.gz"
    files = sorted(base_path.glob(pattern))

    fa_values = []
    imgs = []
    ref_img = None

    for path in files:
        fa = int(path.name.split("_acq-fa")[1].split("_")[0])
        if fa < min_fa:
            continue
        fa_values.append(fa)
        img = nib.load(path)
        imgs.append(img.get_fdata(dtype=np.float32))
        if ref_img is None:
            ref_img = img

    return np.stack(imgs, axis=-1), np.array(fa_values, dtype=np.float32), ref_img


def load_b1_map(base_path: Path, sub: str, ses: str, ref_img: nib.Nifti1Image) -> np.ndarray:
    path = base_path / f"sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_TB1map.nii.gz"
    b1_img = nib.load(path)
    # Pin the defaults from the historical Nilearn 0.12 producer.
    b1_resampled = resample_to_img(b1_img, ref_img, interpolation="continuous", force_resample=False, copy_header=False)
    return cast(np.ndarray, b1_resampled.get_fdata(dtype=np.float32) / 1000.0)


def create_mask(vfa_data: np.ndarray, b1_rel: np.ndarray) -> torch.Tensor:
    mean_signal = np.mean(vfa_data, axis=-1)
    threshold = 60
    mask = mean_signal > threshold
    mask &= (b1_rel > 0.4) & (b1_rel < 1.7)
    mask = clean_mask_3d(mask)
    return mask


def fit_t1_linear(
    vfa_data: np.ndarray,
    fa_deg: np.ndarray,
    b1_rel: np.ndarray,
    mask: np.ndarray | torch.Tensor,
    tr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear DESPOT1 fit: signal (X,Y,Z,FA), B1 ratio, TR in seconds; return T1 seconds and M0."""
    fa_rad = np.deg2rad(fa_deg)
    fa_eff = b1_rel[..., None] * fa_rad

    sin_fa = np.sin(fa_eff)
    tan_fa = np.tan(fa_eff)
    np.clip(sin_fa, 1e-6, None, out=sin_fa)
    tan_fa = np.where(np.abs(tan_fa) < 1e-6, 1e-6, tan_fa)

    y = vfa_data / sin_fa
    x = vfa_data / tan_fa

    t1_map = np.zeros(vfa_data.shape[:-1], dtype=np.float32)
    m0_map = np.zeros_like(t1_map)

    idx = np.where(mask)
    y_m = y[idx]
    x_m = x[idx]

    x_mean = x_m.mean(axis=1, keepdims=True)
    y_mean = y_m.mean(axis=1, keepdims=True)
    xc, yc = x_m - x_mean, y_m - y_mean

    slope = (xc * yc).sum(axis=1) / ((xc**2).sum(axis=1) + 1e-10)
    intercept = y_mean[:, 0] - slope * x_mean[:, 0]

    e1 = np.clip(slope, 1e-6, 1 - 1e-6)
    t1 = np.clip(-tr / np.log(e1), 0, 5.0)
    m0 = intercept / (1 - e1 + 1e-10)

    t1_map[idx] = t1
    m0_map[idx] = m0
    return torch.tensor(t1_map), torch.tensor(m0_map)


def t1_map(sub: str, ses: str, base_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    tr = 0.01

    vfa, fa, ref = load_vfa_data(base_path, sub, ses)
    b1 = load_b1_map(base_path, sub, ses, ref)

    vfa = torch.tensor(vfa).swapaxes(0, 2).flip(1).numpy()
    b1 = torch.tensor(b1).swapaxes(0, 2).flip(1).numpy()
    mask = create_mask(vfa, b1)

    t1, m0 = fit_t1_linear(vfa, fa, b1, mask, tr)

    m0 = m0.clamp(min=m0[mask].quantile(0.02), max=m0[mask].quantile(0.95))
    m0 = m0 / m0[mask].quantile(0.8)
    m0 = m0.clamp(min=-0.1, max=1.5)
    m0[~mask] = 0
    t1[~mask] = 0
    m0 = m0[105:-30]
    t1 = t1[105:-30]
    return t1.float(), m0.float()


def prepare(input_path: Path, output_dir: Path, subjects: list[str] | None = None, limit: int | None = None) -> None:
    """Write the paired parameter/S0 files for selected subject/session keys."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    entries = {}
    for path in sorted(input_path.rglob("*fa03*_T1map.nii.gz")):
        parts = path.name.split("_")
        sub, ses = (parts[0].split("-")[1], parts[1].split("-")[1])
        name = f"sub-{sub}_ses-{ses}"
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
    kinds = ("t1map", "m0map")
    outputs = [output_dir / f"ibc_relaxo_{kind}.h5" for kind in kinds]
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
    with ExitStack() as stack:
        handles = [stack.enter_context(exclusive_h5(path)) for path in outputs]
        for name in names:
            sub, ses = entries[name]
            arrays = t1_map(sub, ses, input_path)
            for handle, array in zip(handles, arrays, strict=True):
                write_dataset(handle, name, np.asarray(array))


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
