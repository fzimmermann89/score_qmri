"""Reconstruct fastMRI, M4Raw and CC using recovered preparation mathematics.

Original SHA256: fastmri.py 98fb3baa1b7de1af4034fcb538ca69c4a66a0c3e0213d03988e42aacc7df4390;
m4raw.py f8989146047f15b53a22aac7848c6e231507165f1367515f0e18450369e37b65;
v16.py 68ec28ef5213977850c0f25e7f39872375f8f253b4e0689254858a92a56d73ba.

Pure FFT, ESPIRiT, phase fitting and masks were extracted from those originals.
CPU is the default, one subject at a time; no downloads or GPU autodetection.
Whole-subject FFT/calibration can still require substantial RAM and CPU time.
CPU/CUDA SVD numerical differences mean byte equality is not guaranteed. Full
Full collection reproduction has not been verified. One real fastMRI subject
was numerically compared; CC prepared-to-curated copying was bit exact.
M4Raw non-GRE uses the complete editor-history all-modality recipe hR8v.py
SHA256 2c3be5e7f515e586eb9084d4d195b5a1a51f9aebc1999a72c1a6699f2d50bd0e
(calibration0.05/0.5, mask0.2). GRE uses the later m4raw.py correction
(calibration0.001/0.1, mask0.1). Both original recipes are preserved in the audit.
CC raw reproduces v16's cc_6 recipe and writes cc_reconstructed.h5. The separate
cc-fix recipe converts the older prepared CC axes and crops50:-20 into cc.h5;
never apply that legacy correction to cc_reconstructed.h5. cc-prepared applies
the verified historical slice filter to an existing correctly oriented prepared
CC H5, producing cc.h5 exactly without reconstructing its 3D raw k-space.
m4raw-prepared imports archived pre-curation m4raw_{t1w,t2w,flair,gre}.h5
through that same filter. fastmri-prepared does the same for fastmri_{t1,t2,flair}.h5
using combined images to select slices for both groups. These modes preserve
every retained value rather than rerunning
the unresolved raw reconstruction. It accepts --modalities and an input directory.
fastMRI/M4Raw outputs apply the historical slice filter by default; --uncurated
keeps every prepared slice. M4Raw repeat order is significant: default native
filesystem order preserves the original recipe; --repeat-order sorted or a JSON
--repeat-manifest makes a chosen coil order explicit and portable.

Pad/crop implements the constant last-dimension operation from installed mrpro
0.260420+dirty4e90b588 (including its odd-padding alignment), avoiding the whole mrpro import.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import einsum, rearrange
from torch import Tensor


def pad_or_crop(data: Tensor, new_shape: Sequence[int]) -> Tensor:
    """Preserve the original mrpro constant pad/crop alignment."""
    if len(new_shape) > data.ndim:
        raise ValueError("New shape exceeds tensor dimensions.")
    shape = (*data.shape[: -len(new_shape)], *new_shape)
    padding = []
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
    return torch.tensor(m, dtype=bool)


def clean_mask_2d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    mask = np.asarray(mask)
    st = ndi.generate_binary_structure(2, 1)
    m = ndi.binary_opening(mask.astype(bool), structure=st)
    m = ndi.binary_closing(m, structure=st, iterations=1)
    m = ndi.binary_dilation(m, structure=st, iterations=2)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m, structure=ndi.generate_binary_structure(2, 1))
    if n:
        m = lab == np.bincount(lab.ravel())[1:].argmax() + 1
    return torch.tensor(m, dtype=bool)


def remove_polynomial_phase(img: torch.Tensor, order: int = 1, threshold: float = 0.2, threshold_max: float = 0.6) -> torch.Tensor:
    """
    Removes polynomial phase ramps using magnitude-weighted gradient fitting.

    Parameters
    ----------
    img : torch.Tensor
        Complex input tensor of shape (B, H, W).
    order : int
        Polynomial order (1 for linear ramp).
    threshold : float
        Magnitude threshold to mask background artifacts.

    Returns
    -------
    torch.Tensor
        Phase-corrected complex tensor.
    """
    B, H, W = img.shape
    device, dtype = (img.device, img.real.dtype)
    mag = img.abs()
    scale = mag.view(B, -1).max(dim=1)[0].view(B, 1, 1)
    mask = (mag > threshold * scale) & (mag < threshold_max * scale)
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    poly_terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i == 0 and j == 0:
                continue
            poly_terms.append(grid_y**i * grid_x**j)
    basis = torch.stack(poly_terms, dim=-1)
    K = basis.shape[-1]
    db_dy = basis[1:, :, :] - basis[:-1, :, :]
    db_dx = basis[:, 1:, :] - basis[:, :-1, :]
    A = torch.cat([db_dy.reshape(-1, K), db_dx.reshape(-1, K)], dim=0)
    gy = torch.angle(img[:, 1:, :] * img[:, :-1, :].conj())
    gx = torch.angle(img[:, :, 1:] * img[:, :, :-1].conj())
    target = torch.cat([gy.reshape(B, -1), gx.reshape(B, -1)], dim=-1)
    wy = mag[:, 1:, :] * mag[:, :-1, :] * mask[:, 1:, :] * mask[:, :-1, :]
    wx = mag[:, :, 1:] * mag[:, :, :-1] * mask[:, :, 1:] * mask[:, :, :-1]
    weight = torch.cat([wy.reshape(B, -1), wx.reshape(B, -1)], dim=-1)
    atw = A.T.unsqueeze(0) * weight.unsqueeze(1)
    lhs = atw @ A.unsqueeze(0)
    rhs = atw @ target.unsqueeze(-1)
    coeffs = torch.linalg.solve(lhs, rhs)
    phase_fit = (basis.view(-1, K) @ coeffs).view(B, H, W)
    corrected = img * torch.exp(-1j * phase_fit.to(img.dtype))
    dc_weights = mask.to(img.dtype)
    avg_phase = torch.angle((corrected * dc_weights).sum(dim=(1, 2), keepdim=True))
    return corrected * torch.exp(-1j * avg_phase)


def ift(x: torch.Tensor) -> torch.Tensor:
    d = (-1, -2)
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(x, dim=d), dim=d), dim=d)


def espirit_2d(
    kspace: Tensor,
    calib_width: int | tuple[int, int, int] = 48,
    thresh: float = 0.05,
    kernel_width: int | tuple[int, int, int] = 12,
    crop: float = 0.5,
    max_iter: int = 50,
) -> Tensor:
    *batch_shape, num_coils, nz, ny, nx = kspace.shape
    spatial_shape = torch.Size([nz, ny, nx])
    spatial_dims = (-3, -2, -1)
    device, dtype = (kspace.device, kspace.dtype)
    calib_width_ = torch.Size([min(calib_width, s) for s in spatial_shape]) if isinstance(calib_width, int) else torch.Size(calib_width)
    kernel_width_ = torch.Size([min(kernel_width, s) for s in calib_width_]) if isinstance(kernel_width, int) else torch.Size(kernel_width)
    calib_region = pad_or_crop(kspace, calib_width_)
    calib_flat = calib_region.flatten(end_dim=-5)
    scale = (spatial_shape.numel() / calib_width_.numel()) ** 0.5
    sens_list = []
    for calib_mat in calib_flat:
        for d, kernel in enumerate(kernel_width_):
            calib_mat = calib_mat.unfold(1 + d, kernel, 1)
        calib_mat = rearrange(calib_mat, "c bz by bx kz ky kx -> (bz by bx) (c kz ky kx)")
        _, S, Vh = torch.linalg.svd(calib_mat, full_matrices=False)
        Vh_thresh = Vh[thresh * S.max() < S]
        kernels = Vh_thresh.unflatten(-1, (num_coils, *kernel_width_))
        kernels_padded = pad_or_crop(kernels, calib_width_)
        img_kernels = torch.fft.ifftn(kernels_padded, dim=spatial_dims, norm="ortho")
        AHA = einsum(img_kernels, img_kernels.conj(), "n coil1 z y x, n coil2 z y x -> z y x coil1 coil2")
        v = torch.ones((*calib_width_, num_coils, 1), dtype=dtype, device=device)
        for _ in range(max_iter):
            v = AHA @ v
            v = v / (v.abs().pow(2).sum(dim=-2, keepdim=True).sqrt() + 1e-12)
        eigenvalues_small = (v.mH @ AHA @ v).squeeze((-2, -1)).abs()
        sens_small = rearrange(v, "z y x coils 1 -> coils z y x")
        phase_ref = sens_small[:1] / (sens_small[:1].abs() + 1e-12)
        sens_small = sens_small * phase_ref.conj()
        sens_kspace = torch.fft.fftn(sens_small, dim=spatial_dims, norm="ortho")
        sens_kspace = torch.fft.fftshift(sens_kspace, dim=spatial_dims)
        sens_kspace_padded = pad_or_crop(sens_kspace, spatial_shape)
        sens_kspace_padded = torch.fft.ifftshift(sens_kspace_padded, dim=spatial_dims)
        sens = torch.fft.fftshift(torch.fft.ifftn(sens_kspace_padded, dim=spatial_dims, norm="ortho"), dim=spatial_dims) * scale
        eig_kspace = torch.fft.fftn(eigenvalues_small, dim=spatial_dims, norm="ortho")
        eig_kspace = torch.fft.fftshift(eig_kspace, dim=spatial_dims)
        eig_kspace_padded = pad_or_crop(eig_kspace, spatial_shape)
        eig_kspace_padded = torch.fft.ifftshift(eig_kspace_padded, dim=spatial_dims)
        eigenvalues = torch.fft.fftshift(torch.fft.ifftn(eig_kspace_padded, dim=spatial_dims, norm="ortho"), dim=spatial_dims).real
        eigenvalues = eigenvalues / (eigenvalues.max() + 1e-12)
        sens = sens * (eigenvalues > crop)
        sens_list.append(sens)
    sens_maps = torch.stack(sens_list).reshape(*batch_shape, num_coils, *spatial_shape)
    return sens_maps


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
    return k_est


def espirit_3d(
    kspace: Tensor, calib_width: int = 32, thresh: float = 0.05, kernel_width: int = 8, crop: float = 0.5, max_iter: int = 100
) -> Tensor:
    """
    ESPIRiT calibration for coil sensitivity estimation.

    Parameters
    ----------
    kspace : Tensor
        K-space data of shape (*batch, coils, kz, ky, kx), centered (DC at center).
    calib_width : int
        Width of calibration region.
    thresh : float
        Singular value threshold.
    kernel_width : int
        Calibration kernel width.
    crop : float
        Eigenvalue threshold for cropping.
    max_iter : int
        Power iteration count.

    Returns
    -------
    Tensor
        Coil sensitivity maps of shape (*batch, coils, kz, ky, kx), centered.
    """
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
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(k, dim=d), dim=d), dim=d)


def _write(group: h5py.Group, name: str, tensor: Tensor, coil_chunks: bool = False) -> None:
    data = tensor.detach().cpu().to(torch.complex64).numpy()
    chunks = (1, 1, *data.shape[2:]) if coil_chunks else (1, *data.shape[1:])
    group.create_dataset(name, data=data, compression="lzf", shuffle=True, chunks=chunks)


def fastmri_images(kspace: Tensor, matrix: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Preserve the original fastMRI crop, phase and quantile conventions."""
    data = kspace.unsqueeze(-3)[:-2]
    if len(data) < 2:
        raise ValueError("fastMRI requires at least four raw slices (last two are omitted).")
    csm = pad_or_crop(espirit_2d(data), matrix)
    coilwise = pad_or_crop(ift(data), matrix)
    img = (coilwise * csm.conj()).sum(-4).squeeze().cpu()
    img /= img.abs()[img.abs() > 1e-8].quantile(0.995)
    img = img.flip(1)
    mask = torch.stack([clean_mask_2d(m) for m in img.abs() > 0.1])
    img_phase = torch.stack([i[m].sum().angle() for i, m in zip(img, mask, strict=True)])
    img *= torch.exp(-1j * img_phase[:, None, None])
    img[~mask] = 0
    coilwise /= coilwise.abs()[coilwise.abs() > 1e-8][::3].quantile(0.995)
    coilwise = coilwise.squeeze(-3).flip(-2).cpu()
    coil_phase = torch.stack([i.sum().angle() for s in coilwise for i in s]).reshape(*coilwise.shape[:2], 1, 1)
    coilwise *= torch.exp(-1j * coil_phase)
    coilwise[~mask.unsqueeze(-3).broadcast_to(coilwise.shape)] = 0
    return img, coilwise


def m4raw_image(repeats: Sequence[Tensor], gre: bool = False) -> Tensor:
    """Preserve repeat-as-coils calibration and fourth-order phase correction."""
    data = torch.cat(list(repeats), dim=1).unsqueeze(-3)
    csm = espirit_2d(data, thresh=0.001 if gre else 0.05, crop=0.1 if gre else 0.5)
    img = (ift(data) * csm.conj()).sum(-4).squeeze().cpu()
    img = remove_polynomial_phase(img, order=4, threshold=0.2, threshold_max=0.7)
    img /= img.abs()[img.abs() > 0.1].quantile(0.98)
    mask = torch.stack([clean_mask_2d(m) for m in img.abs() > (0.1 if gre else 0.2)])
    img[~mask] = 0
    return img


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


def prepare(
    family: str,
    input_root: Path,
    output_dir: Path,
    modalities: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
    curated: bool = True,
    repeat_order: str = "native",
    repeat_manifest: Path | None = None,
) -> None:
    """Prepare selected existing raw files; all output files use exclusive creation."""
    allowed = {
        "fastmri": {"t1", "t2", "flair"},
        "m4raw": {"t1w", "t2w", "flair", "gre"},
        "cc-raw": {"raw"},
        "cc-fix": set(),
        "cc-prepared": set(),
        "m4raw-prepared": {"t1w", "t2w", "flair", "gre"},
        "fastmri-prepared": {"t1", "t2", "flair"},
    }
    if family not in allowed or modalities and not set(modalities) <= allowed[family]:
        raise ValueError("Unknown family or modality selection.")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive.")
    if family in ("m4raw-prepared", "fastmri-prepared"):
        from dataset_preparation.clean_h5 import clean_file

        prefix = family.removesuffix("-prepared")
        chosen = list(modalities) if modalities else sorted(allowed[family])
        for modality in chosen:
            source = input_root / f"{prefix}_{modality}.h5"
            destination = output_dir / source.name
            if not source.is_file():
                raise FileNotFoundError(source)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(destination)
        for modality in chosen:
            filename = f"{prefix}_{modality}.h5"
            clean_file(input_root / filename, output_dir / filename, list(subjects) if subjects else None, limit)
        return
    if family == "cc-prepared":
        from dataset_preparation.clean_h5 import clean_file

        clean_file(input_root, output_dir / "cc.h5", list(subjects) if subjects else None, limit)
        return
    if family == "cc-fix":
        if not input_root.is_file():
            raise ValueError("cc-fix --input-root must be the original prepared CC H5 file.")
        destination = output_dir / "cc.h5"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        output_dir.mkdir(parents=True, exist_ok=True)
        fix_legacy_cc(input_root, destination, subjects, limit)
        return
    files = list(input_root.rglob("*.h5"))
    if family != "m4raw" or repeat_order == "sorted":
        files.sort()
    manifest = json.loads(repeat_manifest.read_text()) if repeat_manifest else None
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for file in files:
        if family == "fastmri":
            parts = file.stem[13:].split("_")
            if len(parts) != 3:
                continue
            sequence, first, second = parts
            modality = "t1" if "T1" in sequence else "t2" if "T2" in sequence else "flair" if "FLAIR" in sequence else ""
            name = f"{first}_{second}"
        elif family == "m4raw":
            parts = file.stem.split("_")
            if len(parts) < 2:
                continue
            name, sequence = parts[0], parts[1][:-2]
            modality = {"T1": "t1w", "T2": "t2w", "FLAIR": "flair", "GRE": "gre"}.get(sequence, "")
        else:
            name, modality = file.stem, "raw"
        if not modality or modalities and modality not in modalities or subjects and name not in subjects:
            continue
        groups[(modality, name)].append(file)
    selected = sorted(groups)[:limit]
    if not selected:
        raise ValueError("No matching raw inputs found; no output files created.")
    if family != "m4raw" and any(len(groups[key]) != 1 for key in selected):
        raise ValueError("Multiple raw files have the same output subject key; select an unambiguous input root.")
    filenames = {modality: (f"{family}_{modality}.h5" if family != "cc-raw" else "cc_reconstructed.h5") for modality, _ in selected}
    for filename in filenames.values():
        output = output_dir / filename
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        targets = {modality: stack.enter_context(h5py.File(output_dir / filename, "x")) for modality, filename in filenames.items()}
        if family != "m4raw":
            for target in targets.values():
                target.create_group("combined")
                target.create_group("coilwise")
        for modality, name in selected:
            paths = groups[(modality, name)]
            if manifest is not None and family == "m4raw":
                key = f"{name}/{modality}"
                if key not in manifest:
                    raise ValueError(f"Missing repeat manifest entry: {key}")
                ordered = [input_root / relative for relative in manifest[key]]
                if set(ordered) != set(paths) or len(ordered) != len(paths):
                    raise ValueError(f"Repeat manifest does not match selected files: {key}")
                paths = ordered
            values = []
            matrix = None
            for path in paths:
                with h5py.File(path, "r") as source:
                    values.append(torch.tensor(np.array(source["kspace"]), device=device))
                    if family == "fastmri":
                        import ismrmrd

                        header = ismrmrd.xsd.CreateFromDocument(source["ismrmrd_header"][()])
                        size = header.encoding[0].reconSpace.matrixSize
                        matrix = (size.x, size.y)
            target = targets[modality]
            if family == "m4raw":
                image = m4raw_image(values, gre=modality == "gre")
                if curated:
                    from dataset_preparation.clean_h5 import retained_indices

                    image = image[retained_indices(image.numpy())]
                _write(target, name, image)
            else:
                if family == "fastmri":
                    assert matrix is not None
                    combined, coilwise = fastmri_images(values[0], matrix)
                else:
                    combined, coilwise = cc_images(values[0])
                if curated and family == "fastmri":
                    from dataset_preparation.clean_h5 import retained_indices

                    indices = retained_indices(combined.numpy())
                    combined, coilwise = combined[indices], coilwise[indices]
                _write(target["combined"], name, combined)
                _write(target["coilwise"], name, coilwise, coil_chunks=family == "cc-raw")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=("fastmri", "m4raw", "cc-raw", "cc-fix", "cc-prepared", "m4raw-prepared", "fastmri-prepared"))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modalities", nargs="+", help="fastmri: t1 t2 flair; m4raw: t1w t2w flair gre")
    parser.add_argument("--subjects", nargs="+", help="Output subject keys, exact match.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=1, help="Maximum subject/modality groups; default1.")
    selection.add_argument("--all", action="store_true", help="Explicitly process the entire selected collection.")
    parser.add_argument("--device", default="cpu", help="Explicit torch device; CPU default, CUDA never selected automatically.")
    parser.add_argument("--uncurated", action="store_true", help="Skip the historical fastMRI/M4Raw slice filter.")
    parser.add_argument("--repeat-order", choices=("native", "sorted"), default="native")
    parser.add_argument("--repeat-manifest", type=Path, help="JSON mapping subject/modality to ordered paths relative to input root.")
    parser.add_argument("--threads", type=int, default=1, help="Torch CPU threads, default1.")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.limit is not None and args.limit < 1:
        parser.error("--threads and --limit must be positive.")
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        prepare(
            args.family,
            args.input_root,
            args.output_dir,
            args.modalities,
            args.subjects,
            None if args.all else args.limit,
            args.device,
            not args.uncurated,
            args.repeat_order,
            args.repeat_manifest,
        )


if __name__ == "__main__":
    main()
