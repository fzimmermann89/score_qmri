"""Reproduce saved priors and a bounded, explicitly labelled retrospective MRI smoke case.

This is an engineering check, not a held-out model-selection benchmark. No image
amplitude rescaling is applied; synthetic measurement noise is specified per real
coordinate. Model/EMA choice, case membership, seeds, arrays and metrics are saved.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from jsonargparse import ArgumentParser, Namespace
from pytorch_msssim import ssim

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.daps_sampler import DAPSConfig, DAPSSampler
from score_qmri.sampling.gaussian import complex_normal_like, sample_linear_gaussian


class MaskedFourier:
    """Unitary FFT and a Cartesian mask; input/output retain mrpro's tuple layout."""

    def __init__(self, mask: torch.Tensor, adjoint: bool = False) -> None:
        self.mask, self.adjoint = mask, adjoint

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.adjoint:
            return (torch.fft.ifft2(x * self.mask, norm="ortho"),)
        return (torch.fft.fft2(x, norm="ortho") * self.mask,)

    @property
    def H(self) -> MaskedFourier:
        return MaskedFourier(self.mask, not self.adjoint)


def metrics(x: torch.Tensor, truth: torch.Tensor, op: MaskedFourier, y: torch.Tensor) -> dict[str, float]:
    magnitude, reference = x.abs(), truth.abs()
    peak = reference.max().clamp_min(1e-12)
    mse = (magnitude - reference).square().mean()
    residual = op(x)[0] - y
    return {
        "complex_nmse": float((x - truth).abs().square().sum() / truth.abs().square().sum().clamp_min(1e-12)),
        "magnitude_psnr": float(10 * torch.log10(peak.square() / mse.clamp_min(1e-20))),
        "magnitude_ssim": float(ssim(magnitude.flatten(0, 2).unsqueeze(1), reference.flatten(0, 2).unsqueeze(1), data_range=float(peak))),
        "observed_residual_rms": float((residual.abs().square().sum() / op.mask.count_nonzero()).sqrt()),
    }


def plot_case(truth: torch.Tensor, zf: torch.Tensor, cg: torch.Tensor, draws: torch.Tensor, output: Path) -> None:
    mean = draws.mean(0, keepdim=True)
    spread = (draws - mean).abs().square().mean(0).sqrt()
    vmax = float(truth.abs().max())
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), layout="constrained")
    images = (truth.abs(), zf.abs(), cg.abs(), mean.abs(), spread)
    labels = ("Reference", "Zero-filled", "Ridge/CG", "DAPS mean", "Complex RMS spread")
    for i, (ax, image, label) in enumerate(zip(axes, images, labels, strict=True)):
        artist = ax.imshow(image.squeeze().cpu().numpy(), cmap="gray" if i < 4 else "magma", vmin=0, vmax=vmax if i < 4 else None)
        ax.set_title(label)
        ax.axis("off")
        if i == 4:
            fig.colorbar(artist, ax=ax, shrink=0.8)
    fig.suptitle("Retrospective smoke case; shared magnitude scale; spread is not calibrated uncertainty")
    fig.savefig(output, dpi=130)
    plt.close(fig)


def parse_arguments(argv: list[str] | None = None) -> Namespace:
    """Parse command-line paths, including multiple checkpoints and training stores."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--group", type=str, default="combined")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--slice", type=int, default=-1)
    parser.add_argument("--training-h5", type=str, nargs="+", default=[])
    parser.add_argument("--require-held-out", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--reverse-steps", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=0.0009)
    parser.add_argument("--sigma-start", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--inner-kernel", choices=["langevin", "gaussian"], default="langevin")
    args = parser.parse_args(argv)
    args.checkpoint = [Path(path) for path in args.checkpoint]
    args.training_h5 = [Path(path) for path in args.training_h5]
    return args


def main() -> None:
    args = parse_arguments()
    if args.size < 32 or args.size % 32 or args.draws < 1 or args.noise_std <= 0 or args.ridge <= 0:
        raise ValueError("size must be a positive multiple of 32; draws and noise-std must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    with h5py.File(args.h5, "r") as file:
        group = file[args.group] if args.group else file
        subject = args.subject or sorted(group.keys())[0]
        dataset = group[subject]
        index = args.slice if args.slice >= 0 else dataset.shape[0] // 2
        image = np.asarray(dataset[index])
    if not np.iscomplexobj(image) or image.ndim != 2 or min(image.shape) < args.size:
        raise ValueError("case must be a complex image slice at least as large as size")
    offsets = [(s - args.size) // 2 for s in image.shape]
    image = image[offsets[0] : offsets[0] + args.size, offsets[1] : offsets[1] + args.size].copy()
    overlap = []
    for path in args.training_h5:
        with h5py.File(path, "r") as file:
            if subject in file or ("combined" in file and subject in file["combined"]):
                overlap.append(str(path))
    if args.require_held_out and (not args.training_h5 or overlap):
        raise ValueError("held-out membership requires supplied training stores and no subject-key overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    truth = torch.from_numpy(image).to(device=device, dtype=torch.complex64)[None, None, None]
    mask = torch.zeros((args.size, args.size), device=device)
    mask[::4] = 1
    # FFT output uses unshifted coordinates, so ACS is at both array ends.
    mask[:8] = mask[-8:] = 1
    op = MaskedFourier(mask)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    y = op(truth)[0] + args.noise_std * complex_normal_like(truth, generator) * mask
    zf = op.H(y)[0]
    cg, cg_info = sample_linear_gaussian(op, y, torch.zeros_like(truth), 1.0, args.ridge**0.5, mean_only=True)
    # Independently verify the operator adjoint in the actual case layout.
    probe = complex_normal_like(truth, generator)
    data_probe = complex_normal_like(y, generator)
    lhs = torch.vdot(op(probe)[0].flatten(), data_probe.flatten())
    rhs = torch.vdot(probe.flatten(), op.H(data_probe)[0].flatten())
    adjoint_error = float((lhs - rhs).abs() / lhs.abs().clamp_min(1e-12))
    if adjoint_error > 1e-5:
        raise RuntimeError(f"adjoint check failed: {adjoint_error}")
    cfg = DAPSConfig(
        inner_kernel=args.inner_kernel,
        anneal_steps=args.steps,
        reverse_steps=args.reverse_steps,
        tau=args.noise_std,
        sigma_start=args.sigma_start,
        sigma_min=0.05,
        sigma_x0=0.01,
        debug_every=0,
    )
    source_hash = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in [*sorted((source_root / "src").rglob("*.py")), Path(__file__).resolve()]:
        source_hash.update(path.relative_to(source_root).as_posix().encode() + b"\0")
        source_hash.update(path.read_bytes())
    report: dict[str, Any] = {
        "purpose": "retrospective engineering smoke test, not held-out model selection",
        "arguments": vars(args),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_tree_sha256": source_hash.hexdigest(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "case": {
            "subject": subject,
            "slice": index,
            "amplitude_rescaling": "none",
            "training_subject_key_overlap": overlap,
            "membership_checks_supplied": [str(p) for p in args.training_h5],
            "patient_independence_verified": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "lightning", "jsonargparse", "pytorch-msssim")},
        },
        "daps_config": asdict(cfg),
        "adjoint_relative_error": adjoint_error,
        "zero_filled": metrics(zf, truth, op, y),
        "ridge_cg": metrics(cg, truth, op, y),
        "cg_relative_residual": float(cg_info.relative_residual.max()),
        "checkpoints": [],
    }

    def save_report() -> None:
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2, default=str))

    np.savez_compressed(args.output / "case.npz", truth=truth.cpu().numpy(), measurement=y.cpu().numpy(), mask=mask.cpu().numpy())
    save_report()
    for number, checkpoint in enumerate(args.checkpoint):
        print(f"Loading {checkpoint} ({args.weights})", flush=True)
        start = time.perf_counter()
        model = DiffusionModel.from_checkpoint(checkpoint, map_location="cpu", compile_model=False, weights=args.weights).to(device)
        record = dict(model.checkpoint_info)
        record["file_size"] = checkpoint.stat().st_size
        with checkpoint.open("rb") as file:
            digest = hashlib.sha256()
            while chunk := file.read(8 * 1024 * 1024):
                digest.update(chunk)
            record["sha256"] = digest.hexdigest()
        kind = torch.tensor([int(DatasetKind.COMPLEX_T1W_COIL_COMBINED)], device=device)
        denoising = []
        with torch.no_grad():
            for sigma in (0.1, 0.5, 1.0):
                eps = complex_normal_like(truth, torch.Generator(device=device).manual_seed(args.seed + 10))
                clean = model.denoise(
                    truth.squeeze(1).squeeze(1) + sigma * eps.squeeze(1).squeeze(1), torch.tensor([sigma], device=device), kind
                )[:, None, None]
                denoising.append({"sigma": sigma, **metrics(clean, truth, op, y)})
            sampler = DAPSSampler(model, cfg)
            draws = sampler.sample(
                op,
                y.expand(args.draws, *y.shape[1:]),
                DatasetKind.COMPLEX_T1W_COIL_COMBINED,
                generator=torch.Generator(device=device).manual_seed(args.seed + 20),
            )
        if not draws.isfinite().all():
            raise RuntimeError("nonfinite reconstruction")
        record.update(
            {
                "denoising": denoising,
                "posterior_mean": metrics(draws.mean(0, keepdim=True), truth, op, y),
                "per_draw": [metrics(draw[None], truth, op, y) for draw in draws],
                "elapsed_seconds": time.perf_counter() - start,
                "inner_cg_max_relative_residual": max(
                    (float(info.relative_residual.max()) for info in sampler.cg_diagnostics), default=None
                ),
                "inner_cg_iterations": [info.iterations for info in sampler.cg_diagnostics],
            }
        )
        name = f"{number}-{checkpoint.parent.name}"
        np.savez_compressed(args.output / f"{name}.npz", draws=draws.cpu().numpy())
        plot_case(truth, zf, cg, draws, args.output / f"{name}.png")
        report["checkpoints"].append(record)
        save_report()
        print(json.dumps(record["posterior_mean"]), flush=True)
        del model, draws
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
