"""IBC Relaxometry: executed January 7 T1/M0 (B1 /1000), December 16 T2*/S0
(3D mask, 20-slice crop), and datacheck/v13.py T2/S0 fits."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from nilearn.image import resample_to_img
from threadpoolctl import threadpool_limits

from dataset_preparation.common import exclusive_h5, write_dataset


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


def load_t2_data(base_path: Path, sub: str, ses: str, run: str = "02") -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load spin-echo magnitudes (X,Y,Z,echo) and echo times in seconds."""
    mag_path = base_path / f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_run-{run}_T2map.nii.gz"
    json_path = mag_path.with_suffix("").with_suffix(".json")

    img = nib.load(mag_path)
    mag = img.get_fdata(dtype=np.float32)

    with open(json_path) as f:
        meta = json.load(f)

    n_echo = mag.shape[-1]

    if "EchoTimes" in meta:
        te = np.array(meta["EchoTimes"], dtype=np.float32)
    elif "EchoTime1" in meta:
        te = np.array(
            [meta[f"EchoTime{i}"] for i in range(1, n_echo + 1)],
            dtype=np.float32,
        )
    else:
        # Multi-echo SE: TE_n = n * TE_spacing (equidistant)
        te_first = meta["EchoTime"]
        te = te_first * np.arange(1, n_echo + 1, dtype=np.float32)

    return mag, te, img


def create_mask_t2(mag: np.ndarray, threshold: float = 100) -> torch.Tensor:
    mask = mag.mean(-1) > threshold
    mask = clean_mask_3d(mask)
    return mask


def fit_t2(mag: np.ndarray, te: np.ndarray, mask: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-linear spin-echo fit; return T2 in seconds and normalized S0."""
    log_s = np.log(np.clip(mag, 1, None))

    t2 = np.zeros(mag.shape[:-1], dtype=np.float32)
    s0 = np.zeros_like(t2)

    idx = np.where(mask)
    y = log_s[idx]

    te_mean = te.mean()
    te_c = te - te_mean
    y_mean = y.mean(axis=1, keepdims=True)
    y_c = y - y_mean

    slope = (y_c @ te_c) / (te_c @ te_c + 1e-10)
    intercept = y_mean[:, 0] - slope * te_mean

    r2 = np.clip(-slope, 1e-6, 100)
    t2[idx] = 1.0 / r2
    s0[idx] = np.exp(intercept)

    s0_tensor = torch.tensor(s0)
    mask_tensor = torch.stack([torch.tensor(ndi.binary_erosion(m)) for m in mask])
    s0_tensor = s0_tensor.clamp_max(s0_tensor[mask_tensor].quantile(0.95))
    s0_tensor = s0_tensor / s0_tensor[mask_tensor].quantile(0.8)
    s0_tensor = s0_tensor * mask_tensor

    t2_tensor = torch.tensor(t2) * mask_tensor
    t2_tensor = t2_tensor.clamp_max(0.3)  # T2 typically <300ms in tissue (longer than T2*)
    return t2_tensor, s0_tensor


def compute_t2_map(
    base_path: Path,
    sub: str,
    ses: str,
    run: str = "02",
    crop: tuple[int, int] = (10, 10),
) -> tuple[torch.Tensor, torch.Tensor]:
    mag, te, _ = load_t2_data(base_path, sub, ses, run)
    mag_tensor = torch.tensor(mag).swapaxes(0, 2).flip(1)
    mag_tensor = mag_tensor[..., :10]
    te = te[:10]

    mask = create_mask_t2(mag_tensor.numpy())
    t2, s0 = fit_t2(mag_tensor.numpy(), te, mask)

    t2 = t2[crop[0] : -crop[1] if crop[1] else None]
    s0 = s0[crop[0] : -crop[1] if crop[1] else None]

    return t2, s0


def prepare(
    input_path: Path,
    output_dir: Path,
    family: str,
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Write the paired parameter/S0 files for selected subject/session keys."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    patterns = {"t1": "*fa03*_T1map.nii.gz", "t2star": "*mag_T2starmap.nii.gz", "t2": "*_T2map.nii.gz"}
    if family not in patterns:
        raise ValueError(f"Unknown family: {family}")
    entries = {}
    for path in sorted(input_path.rglob(patterns[family])):
        parts = path.name.split("_")
        sub, ses = parts[0].split("-")[1], parts[1].split("-")[1]
        run = next((part.split("-")[1] for part in parts if part.startswith("run-")), "02")
        name = f"sub-{sub}_ses-{ses}" + (f"_run-{run}" if family == "t2" else "")
        if family == "t2star" and sub in ("09", "19"):
            continue
        if name in entries:
            raise ValueError(f"Ambiguous duplicate source for {name}")
        entries[name] = (sub, ses, run)
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
    kinds = {"t1": ("t1map", "m0map"), "t2star": ("t2starmap", "s0map"), "t2": ("t2map", "s0map_t2")}[family]
    outputs = [output_dir / f"ibc_relaxo_{kind}.h5" for kind in kinds]
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
    with ExitStack() as stack:
        handles = [stack.enter_context(exclusive_h5(path)) for path in outputs]
        for name in names:
            sub, ses, run = entries[name]
            arrays: tuple[torch.Tensor | np.ndarray, ...]
            if family == "t1":
                arrays = t1_map(sub, ses, input_path)
            elif family == "t2star":
                arrays = compute_t2star_map(input_path, sub, ses)
            else:
                t2_arrays = compute_t2_map(input_path, sub, ses, run)
                # Original v13 writer uses mrpro.utils.pad_or_crop(224,176).
                # Its odd-size convention differs from normalize_h5.py.
                from dataset_preparation.nifti import pad_or_crop

                arrays = tuple(pad_or_crop(array.numpy(), (224, 176)) for array in t2_arrays)
            for handle, array in zip(handles, arrays, strict=True):
                write_dataset(handle, name, np.asarray(array))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="IBC sourcedata directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for new output files")
    parser.add_argument("--family", choices=("t1", "t2star", "t2"), required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int, default=1, help="Subject count (default: 1); use --all for a full build")
    parser.add_argument("--all", action="store_true", help="Explicitly prepare every selected subject")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = prepare(args.input, args.output, args.family, args.subjects, None if args.all else args.limit)
    print(f"Prepared {len(result)} subjects")


if __name__ == "__main__":
    main()
