"""Build fastmri_{t1,t2,flair}.h5 from brain multicoil k-space H5 files.
Request/download the brain multicoil archives at https://fastmri.med.nyu.edu/
and extract beneath --input-root. Reconstruct combined and coilwise images,
then remove empty slices.
Run: python fastmri.py --input-root INPUT --output-dir NEW.
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

MODALITIES = ("t1", "t2", "flair")


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


def _write(group: h5py.Group, name: str, tensor: Tensor) -> None:
    data = tensor.detach().cpu().to(torch.complex64).numpy()
    chunks = (1, *data.shape[1:])
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


def prepare(
    input_root: Path,
    output_dir: Path,
    modalities: Sequence[str] | None = None,
    subjects: Sequence[str] | None = None,
    device: str = "cpu",
) -> None:
    import ismrmrd

    files: dict[str, list[tuple[str, Path]]] = {}
    for path in sorted(input_root.rglob("*.h5")):
        parts = path.stem[13:].split("_")
        if len(parts) != 3:
            continue
        sequence, first, second = parts
        modality = "t1" if "T1" in sequence else "t2" if "T2" in sequence else "flair" if "FLAIR" in sequence else ""
        name = f"{first}_{second}"
        if not modality or (modalities and modality not in modalities) or (subjects and name not in subjects):
            continue
        files.setdefault(modality, []).append((name, path))
    if not files:
        raise ValueError("No matching raw inputs found; no output files created.")
    for modality in modalities or MODALITIES:
        if modality not in files:
            print(f"No {modality} inputs")
    for modality, candidates in files.items():
        selected = sorted(candidates)
        if len({name for name, _ in selected}) != len(selected):
            raise ValueError("Multiple raw files have the same output subject key; select an unambiguous input root.")
        path = output_dir / f"fastmri_{modality}.h5"
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    for modality, candidates in files.items():
        selected = sorted(candidates)
        with exclusive_h5(output_dir / f"fastmri_{modality}.h5") as target:
            combined_group = target.create_group("combined")
            coilwise_group = target.create_group("coilwise")
            for name, path in selected:
                with h5py.File(str(path), "r") as source:
                    kspace = torch.tensor(np.array(source["kspace"]), device=device)
                    header = ismrmrd.xsd.CreateFromDocument(source["ismrmrd_header"][()])
                    size = header.encoding[0].reconSpace.matrixSize
                combined, coilwise = fastmri_images(kspace, (size.x, size.y))
                keep = (combined.abs() > 1e-6).sum((-2, -1)) >= 1000
                if not keep.any():
                    continue
                combined, coilwise = combined[keep], coilwise[keep]
                _write(combined_group, name, combined)
                _write(coilwise_group, name, coilwise)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        prepare(args.input_root, args.output_dir, args.modalities, args.subjects, args.device)


if __name__ == "__main__":
    main()
