"""Build wand_{t1map,t1w,m0_complex,m0_mag,t2w,t1w_neu}.h5.
Follow https://git.cardiff.ac.uk/cubric/wand to download the required NIfTIs
from https://gin.g-node.org/CUBRIC/WAND; pass the downloaded root as --input-root.
MP2RAGE also needs a complete four-table LUT (--lut); use the original
trusted LUT_UK7T.pkl or generate an NPZ with wand_lut.py.
Run: python wand.py --input-root WAND --output-dir NEW --modalities t2w --limit 1."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def clean_mask_2d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Original opening, closing, dilation, hole fill and largest component."""
    structure = ndi.generate_binary_structure(2, 1)
    result = ndi.binary_opening(np.asarray(mask).astype(bool), structure=structure)
    result = ndi.binary_closing(result, structure=structure, iterations=1)
    result = ndi.binary_dilation(result, structure=structure, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n_labels = ndi.label(result, structure=structure)
    if n_labels:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def select(paths: list[Path], subjects: list[str] | None, limit: int | None) -> list[Path]:
    """Filter by BIDS ancestor or filename without changing volume contents."""
    paths = sorted(set(paths))
    if subjects:
        paths = [p for p in paths if any(s in p.parts or p.name.startswith(s + "_") or p.stem == s for s in subjects)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError("No matching input files; check input root, subjects and downloaded annex content")
    return paths


def load_lut(path: str | Path) -> tuple[np.ndarray, ...]:
    """Load all four simulation tables, not the incomplete three-key NPZ."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return tuple(data[k] for k in ("inv1", "inv2", "psir", "mp2rage"))
    with path.open("rb") as handle:
        return tuple(pickle.load(handle)[:4])


def load_complex_mp2rage(
    inv1_mag_path: str, inv1_phase_path: str, inv2_mag_path: str, inv2_phase_path: str
) -> tuple[torch.Tensor, torch.Tensor, nib.Nifti1Image]:
    """Load MP2RAGE data as complex tensors."""

    def load_complex(mag_path: str, phase_path: str) -> torch.Tensor:
        mag = np.squeeze(nib.load(mag_path).get_fdata()).astype(np.float32)
        phase = np.squeeze(nib.load(phase_path).get_fdata()).astype(np.float32)
        if phase.max() > 4000:
            phase = phase * 2 * np.pi / 4095 - np.pi
        return torch.from_numpy(mag * np.exp(1j * phase))

    S1 = load_complex(inv1_mag_path, inv1_phase_path)
    S2 = load_complex(inv2_mag_path, inv2_phase_path)
    return (S1, S2, nib.load(inv1_mag_path))


def compute_psir_mp2rage(S1: torch.Tensor, S2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PSIR and MP2RAGE from complex signals."""
    T1w = (S1.conj() * S2).real
    S1_signed = S1.abs() * T1w.sign()
    denom_psir = S1_signed.abs() + S2.abs()
    denom_mp2 = S1.abs().square() + S2.abs().square()
    PSIR = torch.where(denom_psir > 0, S1_signed / denom_psir, 0.0).real.clamp(-2, 2)
    MP2RAGE = torch.where(denom_mp2 > 0, T1w / denom_mp2, 0.0)
    return (PSIR.float(), MP2RAGE.float())


def fit_t1_m0(
    PSIR: torch.Tensor,
    MP2RAGE: torch.Tensor,
    S1: torch.Tensor,
    S2: torch.Tensor,
    B1map: torch.Tensor,
    lut_path: str = "LUT_UK7T.pkl",
    device: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match contrasts and B1 percent to the complete LUT; return T1 milliseconds, complex M0 and PSIR."""
    device = device or "cpu"
    LUT_INV1, LUT_INV2, LUT_PSIR, LUT_MP2RAGE = load_lut(lut_path)
    MWM = 0.71
    luts = {
        k: torch.from_numpy(v).float().to(device)
        for k, v in [("psir", LUT_PSIR), ("mp2", LUT_MP2RAGE), ("inv1", LUT_INV1 / MWM), ("inv2", LUT_INV2 / MWM)]
    }
    for k in ("psir", "mp2"):
        luts[k][0, :] = 1000
        luts[k][:, 0] = 1000
    shape = PSIR.shape

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.flatten().to(device)

    psir, mp2, s1, s2 = map(flat, [PSIR, MP2RAGE, S1, S2])
    b1 = B1map.flatten().to(device).clamp(1, luts["psir"].shape[0] - 1).round().long()
    T1 = torch.zeros(psir.shape[0], device=device)
    M0 = torch.zeros(psir.shape[0], dtype=torch.complex64, device=device)
    for i in range(0, psir.shape[0], 256):
        sl = slice(i, min(i + 256, psir.shape[0]))
        b1_sl = b1[sl]
        n = b1_sl.shape[0]
        diff = (psir[sl, None] - luts["psir"][b1_sl]).abs() + (mp2[sl, None] - luts["mp2"][b1_sl]).abs()
        t1_idx = diff.argmin(dim=1)
        T1[sl] = t1_idx.float()
        idx = (torch.arange(n, device=device), t1_idx)
        sim1 = luts["inv1"][b1_sl][idx].abs()
        sim2 = luts["inv2"][b1_sl][idx].abs()
        m0_mag1 = s1[sl].abs() / sim1.clamp(min=1e-06)
        m0_mag2 = s2[sl].abs() / sim2.clamp(min=1e-06)
        w1, w2 = (sim1.square(), sim2.square())
        m0_mag = (w1 * m0_mag1 + w2 * m0_mag2) / (w1 + w2).clamp(min=1e-12)
        M0[sl] = m0_mag * torch.exp(1j * s2[sl].angle())
    return (T1.reshape(shape).cpu(), M0.reshape(shape).cpu(), PSIR)


def mp2rage_volumes(folder: Path, lut: Path) -> dict[str, np.ndarray]:
    """Original WAND mask and intensity pipeline after LUT matching."""
    patterns = ["*inv-1_part-mag*.nii.gz", "*inv-1_part-phase*.nii.gz", "*inv-2_part-mag*.nii.gz", "*inv-2_part-phase*.nii.gz"]
    s1, s2, _ = load_complex_mp2rage(*(str(next(folder.glob(p))) for p in patterns))
    psir, mp2 = compute_psir_mp2rage(s1, s2)
    results = fit_t1_m0(psir, mp2, s1, s2, torch.full(psir.shape, 100.0), str(lut), "cpu")
    t1, m0, t1w = (m.moveaxis((0, 1, 2), (2, 1, 0)).flip(1)[100:] for m in results)
    m0 /= m0.abs().max()
    m0 /= float(np.percentile(torch.view_as_real(m0[m0.abs() > 0.1]), 99))
    m0 = torch.complex(m0.real.clamp(-1.3, 1.3), m0.imag.clamp(-1.3, 1.3))
    mask = torch.stack([clean_mask_2d(m > 0.05) for m in m0.abs()])
    m0[~mask] = 0
    t1 /= 1000
    # Original t1.clamp_max(4) did not assign its return value.
    t1[~mask] = 0
    t1w[~mask] = 0
    good = mask.sum((1, 2)) > 24000
    return {
        "t1map": t1[good].float().numpy(),
        "m0_complex": m0[good].numpy().astype(np.complex64),
        "m0_mag": m0[good].abs().float().numpy(),
        "t1w": t1w[good].float().numpy(),
    }


def magnitude_volume(path: Path, modality: str) -> np.ndarray:
    """Original T2w and native T1w paths, including distinct orientation."""
    image = nib.load(path)
    data = torch.tensor(image.get_fdata())
    spacing = image.header["pixdim"][1:4]
    if modality == "t2w":
        data = torch.nn.functional.interpolate(data[None, None], scale_factor=spacing.tolist(), mode="trilinear")[0, 0]
        data = data.moveaxis((0, 1, 2), (1, 0, 2))
        data /= data.max()
        data.clamp_max_(1.4)
        data /= float(np.percentile(data[data > 0.01], 99))
        data = data[20:-10]
    else:
        data /= data.max()
        if (torch.tensor(spacing) - 1).abs().max() > 0.1:
            data = torch.nn.functional.interpolate(data[None, None], scale_factor=spacing.tolist(), mode="trilinear")[0, 0]
        data = data.moveaxis((0, 1, 2), (2, 1, 0)).flip(1)
        data /= float(np.percentile(data[data > 0.01], 99))
        data.clamp_max_(1.4)
        data = data[40:-10]
    mask = torch.stack([clean_mask_2d(m > 0.05) for m in data.abs()])
    good = mask.float().mean((1, 2)) > 0.2
    data[~mask] = 0
    return data[good].float().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=["t1map", "m0_complex", "m0_mag", "t1w", "t2w", "t1w_neu"], required=True)
    parser.add_argument("--lut", type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    quantitative = set(args.modalities) & {"t1map", "m0_complex", "m0_mag", "t1w"}
    if quantitative and (args.lut is None or not args.lut.is_file()):
        parser.error("MP2RAGE requires the original complete LUT via --lut")
    torch.set_num_threads(1)
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(args.output_dir / f"wand_{m}.h5")) for m in args.modalities}
        if quantitative:
            groups = defaultdict(list)
            for path in sorted(args.input_root.rglob("*MP2RAGE*.nii.gz")):
                groups[path.stem.split("_acq")[0]].append(path)
            folders = select([v[0].parent for v in groups.values() if len(v) == 4], args.subjects, args.limit)
            for folder in folders:
                name = "_".join(folder.parts[-3:-1])
                for modality, array in mp2rage_volumes(folder, args.lut).items():
                    if modality in outputs:
                        write_dataset(outputs[modality], name, array)
        for modality, pattern in [("t2w", "*nlgradcorr_T2w*.nii.gz"), ("t1w_neu", "**/anat/**/*T1w*.nii.gz")]:
            if modality not in outputs:
                continue
            for path in select(list(args.input_root.rglob(pattern)), args.subjects, args.limit):
                name = "_".join(path.stem.split("_")[:2]) if modality == "t2w" else path.stem.split("_T1w")[0]
                write_dataset(outputs[modality], name, magnitude_volume(path, modality))


if __name__ == "__main__":
    main()
