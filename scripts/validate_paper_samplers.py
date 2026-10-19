"""Run seeded paper-form samplers on one explicit frozen Fourier test case."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from jsonargparse import ArgumentParser
from pytorch_msssim import ssim

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import AncestralDPSConfig, AncestralDPSSampler, PaperDAPSConfig, PaperDAPSSampler


class MaskedFourier:
    def __init__(self, mask: torch.Tensor, adjoint: bool = False) -> None:
        self.mask, self.adjoint = mask, adjoint

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.fft.ifft2(x * self.mask, norm="ortho") if self.adjoint else torch.fft.fft2(x, norm="ortho") * self.mask,)

    @property
    def H(self) -> MaskedFourier:
        return MaskedFourier(self.mask, not self.adjoint)


def metrics(x: torch.Tensor, truth: torch.Tensor, op: MaskedFourier, y: torch.Tensor) -> dict[str, float]:
    reference, magnitude = truth.abs(), x.abs()
    peak = float(reference.max())
    return {
        "magnitude_psnr": float(10 * torch.log10(reference.max().square() / (magnitude - reference).square().mean().clamp_min(1e-20))),
        "magnitude_ssim": float(ssim(magnitude.flatten(0, 2).unsqueeze(1), reference.flatten(0, 2).unsqueeze(1), data_range=peak)),
        "complex_nmse": float((x - truth).abs().square().sum() / truth.abs().square().sum()),
        "observed_residual_rms": float(((op(x)[0] - y).abs().square().sum() / op.mask.count_nonzero()).sqrt()),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", type=str, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--sampler", choices=["ancestral-dps", "paper-daps"], required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[62, 63])
    parser.add_argument("--guidance-scale", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--daps-prior", choices=["gaussian", "score"], default="gaussian")
    parser.add_argument("--prior-sigma", type=float, default=None)
    parser.add_argument("--langevin-lr", type=float, default=1.5e-5)
    parser.add_argument("--langevin-steps", type=int, default=100)
    parser.add_argument("--lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    arrays = np.load(args.cases)
    truth = torch.from_numpy(arrays["truth"][args.case_index : args.case_index + 1]).to(args.device)
    measurement = torch.from_numpy(arrays["measurement"][args.case_index : args.case_index + 1]).to(args.device)
    op = MaskedFourier(torch.from_numpy(arrays["mask"]).to(args.device))
    model = DiffusionModel.from_checkpoint(args.checkpoint, map_location="cpu", compile_model=False, weights="ema").to(args.device)
    cfg = (
        AncestralDPSConfig(num_steps=args.steps or 1000, guidance_scale=args.guidance_scale)
        if args.sampler == "ancestral-dps"
        else PaperDAPSConfig(
            tau=args.tau,
            anneal_steps=args.steps or 200,
            prior=args.daps_prior,
            prior_sigma=args.prior_sigma,
            langevin_lr=args.langevin_lr,
            langevin_steps=args.langevin_steps,
            lr_min_ratio=args.lr_min_ratio,
            reverse_steps=args.reverse_steps,
        )
    )
    sampler = AncestralDPSSampler(model, cfg) if args.sampler == "ancestral-dps" else PaperDAPSSampler(model, cfg)
    source = Path(__file__).resolve().parents[1] / "src/score_qmri/sampling"
    report = {
        "scope": "retrospective engineering screen; no posterior calibration or held-out generalization claim",
        "arguments": vars(args),
        "model": model.checkpoint_info,
        "configuration": asdict(cfg),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))},
        "case_file_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "zero_filled": metrics(op.H(measurement)[0], truth, op, measurement),
        "draws": [],
    }

    def save() -> None:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))

    save()
    samples = []
    for seed in args.seeds:
        start = time.perf_counter()
        draw = sampler.sample(op, measurement, DatasetKind[args.kind], generator=torch.Generator(device=args.device).manual_seed(seed))
        samples.append(draw.detach().cpu())
        np.savez_compressed(args.output / f"draw-{seed}.npz", sample=samples[-1].numpy())
        report["draws"].append({"seed": seed, "elapsed_seconds": time.perf_counter() - start, **metrics(draw, truth, op, measurement)})
        save()
        print(json.dumps(report["draws"][-1]), flush=True)
    fig, axes = plt.subplots(1, 2 + len(samples), figsize=(3 * (2 + len(samples)), 3), layout="constrained")
    images = [truth.detach().cpu(), op.H(measurement)[0].detach().cpu(), *samples]
    for ax, image, label in zip(axes, images, ["Reference", "Zero-filled", *[f"Seed {s}" for s in args.seeds]], strict=True):
        ax.imshow(image.abs().squeeze().numpy(), cmap="gray", vmin=0, vmax=float(truth.abs().max()))
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(f"{args.sampler} · {args.kind} · shared magnitude scale")
    fig.savefig(args.output / "draws.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
