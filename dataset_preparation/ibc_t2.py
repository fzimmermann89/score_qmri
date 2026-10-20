"""Build ibc_relaxo_{t2map,s0map_t2}.h5 from IBC relaxometry sourcedata.
Download the relaxometry NIfTIs and JSON sidecars using the IBC data access
instructions at https://individual-brain-charting.github.io/docs/ibc_api.html;
pass the sourcedata root as --input-root. The recovered fit differs by floating-point rounding in the checked subject.
Run: python ibc_t2.py --input-root RAW --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
import json
import math
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


def pad_or_crop(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Historical mrpro center convention, including odd negative padding."""
    pads = []
    for old, new in reversed(list(zip(array.shape[-2:], shape, strict=True))):
        delta = new - old
        left = math.trunc(delta / 2)
        pads.extend([left, delta - left])
    return torch.nn.functional.pad(torch.from_numpy(array), pads).numpy()


def prepare(input_path: Path, output_dir: Path, subjects: list[str] | None = None, limit: int | None = None) -> None:
    """Write the paired parameter/S0 files for selected subject/session keys."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    entries = {}
    for path in sorted(input_path.rglob("*_T2map.nii.gz")):
        parts = path.name.split("_")
        sub, ses = (parts[0].split("-")[1], parts[1].split("-")[1])
        run = next((part.split("-")[1] for part in parts if part.startswith("run-")), "02")
        name = f"sub-{sub}_ses-{ses}" + f"_run-{run}"
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
    kinds = ("t2map", "s0map_t2")
    outputs = [output_dir / f"ibc_relaxo_{kind}.h5" for kind in kinds]
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
    with ExitStack() as stack:
        handles = [stack.enter_context(exclusive_h5(path)) for path in outputs]
        for name in names:
            sub, ses, run = entries[name]
            arrays: tuple[torch.Tensor | np.ndarray, ...]
            t2_arrays = compute_t2_map(input_path, sub, ses, run)

            arrays = tuple(pad_or_crop(array.numpy(), (224, 176)) for array in t2_arrays)
            for handle, array in zip(handles, arrays, strict=True):
                write_dataset(handle, name, np.asarray(array))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        limit = None if args.all else args.limit
        if limit is None and not args.all and not args.subjects:
            limit = 1
        prepare(args.input_root, args.output_dir, args.subjects, limit)


if __name__ == "__main__":
    main()
