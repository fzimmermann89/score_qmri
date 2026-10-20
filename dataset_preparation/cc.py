"""Build cc.h5 from CC 12-channel k-space H5 files.
Download the Train/Val archives from
https://www.ccdataset.com/mr-reconstruction-challenge and extract beneath --input-root.
Reconstruct combined and coilwise images, then remove empty slices.
Run: python cc.py --input-root RAW --output-dir NEW.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import einsum, rearrange
from torch import Tensor


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


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
    """Reproduce the separate v16 cc_6 reconstruction."""
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


def reconstruct(
    input_root: Path,
    output_dir: Path,
    subjects: Sequence[str] | None = None,
    device: str = "cpu",
) -> None:
    files = sorted(path for path in input_root.rglob("*.h5") if not subjects or path.stem in subjects)
    if not files:
        raise ValueError("No matching raw inputs found; no output files created.")
    if len({path.stem for path in files}) != len(files):
        raise ValueError("Multiple raw files have the same output subject key; select an unambiguous input root.")
    with exclusive_h5(output_dir / "cc.h5") as target:
        combined_group = target.create_group("combined")
        coilwise_group = target.create_group("coilwise")
        for path in files:
            with h5py.File(str(path), "r") as source:
                raw = torch.tensor(np.array(source["kspace"]), device=device)
            combined, coilwise = cc_images(raw)
            keep = (combined.abs() > 1e-6).sum((-2, -1)) >= 1000
            if not keep.any():
                continue
            combined, coilwise = combined[keep], coilwise[keep]
            _write(combined_group, path.stem, combined)
            _write(coilwise_group, path.stem, coilwise, coil_chunks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        reconstruct(args.input_root, args.output_dir, subjects=args.subjects)


if __name__ == "__main__":
    main()
