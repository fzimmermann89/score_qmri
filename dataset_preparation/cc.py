"""Build cc.h5 by slice-filtering the archived prepared CC H5 (--input-root).
For fresh raw data, download 12-channel Train/Val H5 files from
https://www.ccdataset.com/mr-reconstruction-challenge and pass their root
with --raw. Raw reconstruction writes cc_reconstructed.h5 and is not yet
verified against the training collection; the prepared-input path is exact.
Run: python cc.py --input-root INPUT --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import einsum, rearrange
from torch import Tensor

ACTIVE_THRESHOLD = 1e-6
MIN_ACTIVE_PIXELS = 1000


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "x")


def axial_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Use the historical cleanup layout for image volumes."""
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"Expected a nonempty array, got shape {shape}")
    if len(shape) == 4:
        return (1, 1, *shape[2:])
    return (1, *shape[1:])


def retained_indices(
    dataset: h5py.Dataset | np.ndarray, threshold: float = ACTIVE_THRESHOLD, min_pixels: int = MIN_ACTIVE_PIXELS
) -> list[int]:
    """Return the exact original >=1000 active-pixel selection."""
    if dataset.ndim not in (3, 4):
        raise ValueError(f"Expected (Z,H,W) or (Z,C,H,W): {dataset.shape}")
    valid = []
    for index in range(dataset.shape[0]):
        active = np.abs(dataset[index]) > threshold
        count = int(np.max(np.sum(active, axis=(-2, -1))) if active.ndim == 3 else np.sum(active))
        if count >= min_pixels:
            valid.append(index)
    return valid


def _copy_selected(source: h5py.Dataset, target: h5py.Group, name: str, indices: list[int]) -> None:
    shape = (len(indices), *source.shape[1:])
    output = target.create_dataset(name, shape=shape, dtype=source.dtype, chunks=axial_chunks(shape), compression="lzf", shuffle=True)
    for destination, original in enumerate(indices):
        output[destination] = source[original]


def clean_file(
    input_path: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, dict[str, int]]:
    """Create a new filtered file, optionally selecting a bounded subject subset."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    with h5py.File(input_path, "r") as source:
        groups = [name for name in ("coilwise", "combined") if name in source and isinstance(source[name], h5py.Group)]
        reference = source["combined"] if "combined" in groups else source[groups[0]] if groups else source
        names = sorted(name for name in reference if isinstance(reference[name], h5py.Dataset))
        if not groups and any(isinstance(source[name], h5py.Group) for name in source):
            raise ValueError("Unsupported nested schema; use the Brainweb packing recipe for class volumes")
        if subjects is not None:
            missing = set(subjects) - set(names)
            if missing:
                raise ValueError(f"Unknown subjects: {sorted(missing)}")
            names = [name for name in names if name in subjects]
        if limit is not None:
            names = names[:limit]
        if not names:
            raise ValueError("No subjects selected")
        report = {}
        with exclusive_h5(output_path) as target:
            for name in groups:
                target.create_group(name)
            for name in names:
                indices = retained_indices(reference[name])
                report[name] = {"input_slices": reference[name].shape[0], "output_slices": len(indices)}
                if not indices:
                    continue
                if groups:
                    for group in groups:
                        if name in source[group]:
                            _copy_selected(source[group][name], target[group], name, indices)
                else:
                    _copy_selected(source[name], target, name, indices)
        return report


def pad_or_crop(data: Tensor, new_shape: Sequence[int]) -> Tensor:
    """Preserve the original mrpro constant pad/crop alignment."""
    if len(new_shape) > data.ndim:
        raise ValueError("New shape exceeds tensor dimensions.")
    shape = (*data.shape[: -len(new_shape)], *new_shape)
    padding: list[int] = []
    for old, new in zip(data.shape, shape, strict=True):
        after = math.trunc((new - old) / 2)
        before = new - old - after
        if before or after or padding:
            padding.extend((before, after))
    return torch.nn.functional.pad(data, padding[::-1]) if any(padding) else data


def clean_mask_3d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    mask = np.asarray(mask)
    st = ndi.generate_binary_structure(3, 1)
    m = ndi.binary_opening(mask.astype(bool), structure=st)
    m = ndi.binary_closing(m, structure=st, iterations=3)
    m = np.array([ndi.binary_fill_holes(s) for s in m])
    lab, n = ndi.label(m, structure=ndi.generate_binary_structure(3, 1))
    if n:
        m = lab == np.bincount(lab.ravel())[1:].argmax() + 1
    return torch.tensor(m, dtype=torch.bool)


def pocs_partial_fourier(kspace: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """POCS for partial Fourier reconstruction."""
    mask = (kspace.abs().sum(-1) > 100).unsqueeze(-1)
    sym_mask = mask & torch.flip(mask, dims=(1, 2))
    low_res_k = kspace * sym_mask
    low_res_img = torch.fft.ifftn(torch.fft.ifftshift(low_res_k, dim=(1, 2)), dim=(1, 2))
    phase_est = low_res_img / (low_res_img.abs() + 1e-08)
    img = torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(1, 2)), dim=(1, 2))
    for _ in range(n_iter):
        img = img.abs() * phase_est
        k_est = torch.fft.fftshift(torch.fft.fftn(img, dim=(1, 2)), dim=(1, 2))
        k_est = torch.where(mask, kspace, k_est)
        img = torch.fft.ifftn(torch.fft.ifftshift(k_est, dim=(1, 2)), dim=(1, 2))
    return cast(Tensor, k_est)


def espirit_3d(
    kspace: Tensor, calib_width: int = 32, thresh: float = 0.05, kernel_width: int = 8, crop: float = 0.5, max_iter: int = 100
) -> Tensor:
    """Estimate coil sensitivities from centered (*batch,coils,kz,ky,kx) k-space."""
    *batch_shape, num_coils, nz, ny, nx = kspace.shape
    spatial_shape = (nz, ny, nx)
    spatial_dims = (-3, -2, -1)
    device, dtype = (kspace.device, kspace.dtype)
    pad_calib = []
    for size in spatial_shape[::-1]:
        d = (calib_width - size) // 2
        pad_calib.extend([d, d + (calib_width - size) % 2])
    calib_region = torch.nn.functional.pad(kspace, pad_calib)
    calib_flat = calib_region.reshape(-1, num_coils, calib_width, calib_width, calib_width)
    batch_size = calib_flat.shape[0]
    pad_interp = []
    for size in spatial_shape[::-1]:
        d = (size - kernel_width) // 2
        pad_interp.extend([d, d + (size - kernel_width) % 2])
    interp_scale = (nz * ny * nx / kernel_width**3) ** 0.5
    sens_list = []
    for b in range(batch_size):
        calib_data = calib_flat[b]
        calib_mat = calib_data
        for d in range(3):
            calib_mat = calib_mat.unfold(1 + d, kernel_width, 1)
        calib_mat = rearrange(calib_mat, "c bz by bx kz ky kx -> (bz by bx) (c kz ky kx)")
        _, S, Vh = torch.linalg.svd(calib_mat, full_matrices=False)
        Vh_thresh = Vh[thresh * S.max() < S]
        kernels = rearrange(Vh_thresh, "n (c kz ky kx) -> n c kz ky kx", c=num_coils, kz=kernel_width, ky=kernel_width, kx=kernel_width)
        img_kernels = torch.fft.ifftn(kernels, dim=spatial_dims, norm="ortho")
        AHA = einsum(img_kernels, img_kernels.conj(), "n c1 z y x, n c2 z y x -> z y x c1 c2")
        v = torch.ones((kernel_width, kernel_width, kernel_width, num_coils, 1), dtype=dtype, device=device)
        for _ in range(max_iter):
            v = AHA @ v
            v = v / (v.abs().pow(2).sum(dim=-2, keepdim=True).sqrt() + 1e-12)
        eigenvalues_small = (v.mH @ AHA @ v).squeeze((-2, -1)).abs()
        sens_small = rearrange(v.squeeze(-1), "z y x c -> c z y x")
        phase_ref = sens_small[0:1] / (sens_small[0:1].abs() + 1e-12)
        sens_small = sens_small * phase_ref.conj()
        sens_kspace = torch.fft.fftn(sens_small, dim=spatial_dims, norm="ortho")
        sens_kspace = torch.fft.fftshift(sens_kspace, dim=spatial_dims)
        sens_kspace_padded = torch.nn.functional.pad(sens_kspace, pad_interp)
        sens_kspace_padded = torch.fft.ifftshift(sens_kspace_padded, dim=spatial_dims)
        sens = torch.fft.fftshift(torch.fft.ifftn(sens_kspace_padded, dim=spatial_dims, norm="ortho"), dim=spatial_dims) * interp_scale
        eig_kspace = torch.fft.fftn(eigenvalues_small, dim=spatial_dims, norm="ortho")
        eig_kspace = torch.fft.fftshift(eig_kspace, dim=spatial_dims)
        eig_kspace_padded = torch.nn.functional.pad(eig_kspace, pad_interp)
        eig_kspace_padded = torch.fft.ifftshift(eig_kspace_padded, dim=spatial_dims)
        eigenvalues = (
            torch.fft.fftshift(torch.fft.ifftn(eig_kspace_padded, dim=spatial_dims, norm="ortho"), dim=spatial_dims).real * interp_scale
        )
        eigenvalues = eigenvalues / (eigenvalues.max() + 1e-12)
        sens = sens * (eigenvalues > crop)
        sens_list.append(sens)
    sens_maps = torch.stack(sens_list).reshape(*batch_shape, num_coils, nz, ny, nx)
    return sens_maps


def ifft_3d(k: torch.Tensor) -> torch.Tensor:
    d = (-1, -2, -3)
    return cast(Tensor, torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=d), dim=d), dim=d))


def _write(group: h5py.Group, name: str, tensor: Tensor, coil_chunks: bool = False) -> None:
    data = tensor.detach().cpu().to(torch.complex64).numpy()
    chunks = (1, 1, *data.shape[2:]) if coil_chunks else (1, *data.shape[1:])
    group.create_dataset(name, data=data, compression="lzf", shuffle=True, chunks=chunks)


def cc_images(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Reproduce the separate v16 cc_6 reconstruction, with no legacy correction."""
    kspace = torch.view_as_complex(raw.unflatten(-1, (-1, 2)))
    filled = pocs_partial_fourier(kspace)
    ks = torch.fft.fft2(torch.fft.fftshift(torch.fft.ifft2(filled.moveaxis(-1, 0)), dim=(-1, -2)))
    ks = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(ks, dim=1), dim=1), dim=1)
    csm = espirit_3d(ks[None])[0]
    coilwise = ifft_3d(ks)
    img = (coilwise * csm.conj()).sum(0)
    img /= img[img.abs() > 0].abs().quantile(0.99)
    img = img.cpu()
    mask = clean_mask_3d(img.abs() > 0.05)
    img = pad_or_crop(img * mask, (224, 224)).to(torch.complex64)
    coilwise = pad_or_crop(coilwise.cpu() * mask, (224, 224)).moveaxis(0, 1).to(torch.complex64)
    coilwise /= np.quantile(coilwise[coilwise.abs() > 0].abs().numpy(), 0.99)
    return img, coilwise


def fix_legacy_cc(input_file: Path, output_file: Path, subjects: Sequence[str] | None = None, limit: int | None = None) -> None:
    """Stream the original axis swaps and50:-20 crop without reading whole volumes."""
    if output_file.exists() or output_file.is_symlink():
        raise FileExistsError(output_file)
    with h5py.File(input_file, "r") as source, h5py.File(output_file, "x") as target:
        combined = target.create_group("combined")
        coilwise = target.create_group("coilwise")
        names = sorted(source["combined"])
        if subjects:
            names = [name for name in names if name in subjects]
        for name in names[:limit]:
            src = source["combined"][name]
            coils = source["coilwise"][name]
            length = src.shape[0] - 70
            if length < 1 or coils.ndim != 4 or coils.shape[2] != src.shape[0]:
                raise ValueError(f"{name}: not the legacy CC schema expected by fix_cc.py.")
            dst = combined.create_dataset(
                name, shape=(length, *src.shape[1:]), dtype=src.dtype, compression="lzf", shuffle=True, chunks=(1, *src.shape[1:])
            )
            # swapaxes(0,2), then swapaxes(-1,-2): (Z,C,W,H) -> (W,C,H,Z).
            spatial_shape = (coils.shape[3], coils.shape[0])
            out_coils = coilwise.create_dataset(
                name,
                shape=(length, coils.shape[1], *spatial_shape),
                dtype=coils.dtype,
                compression="lzf",
                shuffle=True,
                chunks=(1, 1, *spatial_shape),
            )
            for z in range(length):
                dst[z] = src[z + 50]
                out_coils[z] = np.moveaxis(coils[:, :, z + 50, :], 0, -1)


def reconstruct(
    input_root: Path,
    output_dir: Path,
    modalities: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
) -> None:
    """Prepare selected existing raw files; all output files use exclusive creation."""
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive.")
    files = list(input_root.rglob("*.h5"))
    files.sort()
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for file in files:
        name, modality = (file.stem, "raw")
        if not modality or (modalities and modality not in modalities) or (subjects and name not in subjects):
            continue
        groups[modality, name].append(file)
    selected = sorted(groups)[:limit]
    if not selected:
        raise ValueError("No matching raw inputs found; no output files created.")
    if any(len(groups[key]) != 1 for key in selected):
        raise ValueError("Multiple raw files have the same output subject key; select an unambiguous input root.")
    filenames = {modality: "cc_reconstructed.h5" for modality, _ in selected}
    for filename in filenames.values():
        output = output_dir / filename
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        targets = {modality: stack.enter_context(h5py.File(output_dir / filename, "x")) for modality, filename in filenames.items()}
        for target in targets.values():
            target.create_group("combined")
            target.create_group("coilwise")
        for modality, name in selected:
            paths = groups[modality, name]
            values = []
            for path in paths:
                with h5py.File(path, "r") as source:
                    values.append(torch.tensor(np.array(source["kspace"]), device=device))
            target = targets[modality]
            combined, coilwise = cc_images(values[0])
            _write(target["combined"], name, combined)
            _write(target["coilwise"], name, coilwise, coil_chunks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Archived prepared cc.h5; a raw H5 directory with --raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--raw", action="store_true")
    modes.add_argument("--legacy", action="store_true", help="Apply fix_cc.py to its old axis layout")
    args = parser.parse_args()
    limit = None if args.all else args.limit
    torch.set_num_threads(1)
    if args.raw:
        with torch.inference_mode():
            reconstruct(args.input_root, args.output_dir, subjects=args.subjects, limit=limit)
    elif args.legacy:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fix_legacy_cc(args.input_root, args.output_dir / "cc.h5", args.subjects, limit)
    else:
        clean_file(args.input_root, args.output_dir / "cc.h5", args.subjects, limit)


if __name__ == "__main__":
    main()
