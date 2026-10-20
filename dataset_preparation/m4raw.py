"""Build m4raw_{t1w,t2w,flair,gre}.h5 from M4Raw V1.5 multicoil repeats.
Download the original V1.5 archives via https://zenodo.org/records/8056074
(version history); extract beneath --input-root, retaining repeat filenames.
Raw reconstruction still differs from the training H5. --prepared instead
filters archived m4raw_*.h5 intermediates and matched the checked subjects exactly.
Run: python m4raw.py --input-root INPUT --output-dir NEW --limit 1.
"""

from __future__ import annotations

import argparse
import json
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
MODALITIES = ("t1w", "t2w", "flair", "gre")


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
    return torch.tensor(m, dtype=torch.bool)


def remove_polynomial_phase(img: torch.Tensor, order: int = 1, threshold: float = 0.2, threshold_max: float = 0.6) -> torch.Tensor:
    """Remove phase ramps from complex (B,H,W) images using magnitude-weighted gradients."""
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
    return cast(Tensor, torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(x, dim=d), dim=d), dim=d))


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


def _write(group: h5py.Group, name: str, tensor: Tensor, coil_chunks: bool = False) -> None:
    data = tensor.detach().cpu().to(torch.complex64).numpy()
    chunks = (1, 1, *data.shape[2:]) if coil_chunks else (1, *data.shape[1:])
    group.create_dataset(name, data=data, compression="lzf", shuffle=True, chunks=chunks)


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


def prepare(
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
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive.")
    files = list(input_root.rglob("*.h5"))
    if repeat_order == "sorted":
        files.sort()
    manifest = json.loads(repeat_manifest.read_text()) if repeat_manifest else None
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for file in files:
        parts = file.stem.split("_")
        if len(parts) < 2:
            continue
        name, sequence = (parts[0], parts[1][:-2])
        modality = {"T1": "t1w", "T2": "t2w", "FLAIR": "flair", "GRE": "gre"}.get(sequence, "")
        if not modality or (modalities and modality not in modalities) or (subjects and name not in subjects):
            continue
        groups[modality, name].append(file)
    selected = sorted(groups)[:limit]
    if not selected:
        raise ValueError("No matching raw inputs found; no output files created.")
    filenames = {modality: f"m4raw_{modality}.h5" for modality, _ in selected}
    for filename in filenames.values():
        output = output_dir / filename
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        targets = {modality: stack.enter_context(h5py.File(output_dir / filename, "x")) for modality, filename in filenames.items()}
        for modality, name in selected:
            paths = groups[modality, name]
            if manifest is not None:
                key = f"{name}/{modality}"
                if key not in manifest:
                    raise ValueError(f"Missing repeat manifest entry: {key}")
                ordered = [input_root / relative for relative in manifest[key]]
                if set(ordered) != set(paths) or len(ordered) != len(paths):
                    raise ValueError(f"Repeat manifest does not match selected files: {key}")
                paths = ordered
            values = []
            for path in paths:
                with h5py.File(path, "r") as source:
                    values.append(torch.tensor(np.array(source["kspace"]), device=device))
            target = targets[modality]
            image = m4raw_image(values, gre=modality == "gre")
            if curated:
                image = image[retained_indices(image.numpy())]
            _write(target, name, image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--prepared", action="store_true", help="Filter archived prepared H5 files instead of reconstructing k-space")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat-order", choices=["native", "sorted"], default="native")
    parser.add_argument("--repeat-manifest", type=Path)
    args = parser.parse_args()
    limit = None if args.all else args.limit
    torch.set_num_threads(1)
    if args.prepared:
        for modality in args.modalities:
            filename = f"m4raw_{modality}.h5"
            clean_file(args.input_root / filename, args.output_dir / filename, args.subjects, limit)
    else:
        with torch.inference_mode():
            prepare(
                args.input_root,
                args.output_dir,
                args.modalities,
                args.subjects,
                limit,
                args.device,
                repeat_order=args.repeat_order,
                repeat_manifest=args.repeat_manifest,
            )


if __name__ == "__main__":
    main()
