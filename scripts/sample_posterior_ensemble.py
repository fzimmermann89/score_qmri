"""Save independent batched draws for explicit frozen MRI posterior experiments."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jsonargparse import ArgumentParser
from pytorch_msssim import ssim

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler


class MaskedFourier:
    def __init__(self, mask: torch.Tensor, adjoint: bool = False) -> None:
        self.mask, self.adjoint = mask, adjoint

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.fft.ifft2(x * self.mask.conj(), norm="ortho") if self.adjoint else torch.fft.fft2(x, norm="ortho") * self.mask,)

    @property
    def H(self) -> MaskedFourier:
        return MaskedFourier(self.mask, not self.adjoint)


def metrics(x: torch.Tensor, truth: torch.Tensor, op: MaskedFourier, y: torch.Tensor) -> dict[str, float]:
    reference, magnitude = truth.abs(), x.abs()
    return {
        "magnitude_psnr": float(10 * torch.log10(reference.max().square() / (magnitude - reference).square().mean().clamp_min(1e-20))),
        "magnitude_ssim": float(
            ssim(magnitude.flatten(0, 2).unsqueeze(1), reference.flatten(0, 2).unsqueeze(1), data_range=float(reference.max()))
        ),
        "complex_nmse": float((x - truth).abs().square().sum() / truth.abs().square().sum()),
        "observed_residual_rms": float(((op(x)[0] - y).abs().square().sum() / op.mask.count_nonzero()).sqrt()),
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--kind", type=str, required=True)
    parser.add_argument("--sampler", choices=["dps", "daps"], required=True)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--prior-sigma", type=float, default=0.01)
    parser.add_argument("--langevin-step", type=float, default=0.1)
    parser.add_argument("--langevin-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.draws < 2 or args.batch_size < 1:
        raise ValueError("Require at least two draws and positive batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    arrays = np.load(args.cases)
    if not 0 <= args.case_index < len(arrays["truth"]):
        raise ValueError("case-index outside frozen cases")
    truth = torch.from_numpy(arrays["truth"][args.case_index : args.case_index + 1]).to(args.device)
    measurement = torch.from_numpy(arrays["measurement"][args.case_index : args.case_index + 1]).to(args.device)
    mask = torch.from_numpy(arrays["mask"]).to(args.device)
    op = MaskedFourier(mask)
    model = DiffusionModel.from_checkpoint(args.checkpoint, map_location="cpu", compile_model=False, weights="ema").to(args.device)
    cfg = (
        DPSConfig(tau=args.tau, num_steps=args.steps or 1000)
        if args.sampler == "dps"
        else DAPSConfig(
            tau=args.tau,
            num_steps=args.steps or 32,
            prior_sigma=args.prior_sigma,
            langevin_step=args.langevin_step,
            langevin_steps=args.langevin_steps,
        )
    )
    sampler = DPSSampler(model, cfg) if isinstance(cfg, DPSConfig) else DAPSSampler(model, cfg)
    source = Path(__file__).resolve().parents[1] / "src/score_qmri/sampling"
    report: dict[str, Any] = {
        "scope": "retrospective same-measurement ensemble; training-store overlap; no calibration claim",
        "arguments": vars(args),
        "configuration": asdict(cfg),
        "model": model.checkpoint_info,
        "case_file_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))},
        "randomness": "independent pseudorandom draws within each batch; batch seed=seed+batch index; changing batch size changes draws",
        "zero_filled": metrics(op.H(measurement)[0], truth, op, measurement),
        "batches": [],
    }
    np.savez_compressed(args.output / "case.npz", truth=truth.cpu().numpy(), measurement=measurement.cpu().numpy(), mask=mask.cpu().numpy())
    for batch_index, first in enumerate(range(0, args.draws, args.batch_size)):
        count = min(args.batch_size, args.draws - first)
        y = measurement.expand(count, *measurement.shape[1:])
        start = time.perf_counter()
        draw = sampler.sample(
            op, y, DatasetKind[args.kind], generator=torch.Generator(device=args.device).manual_seed(args.seed + batch_index)
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        values = []
        for j in range(count):
            sample = draw[j : j + 1]
            np.savez_compressed(args.output / f"draw-{first + j:03d}.npz", sample=sample.cpu().numpy())
            values.append(metrics(sample, truth, op, measurement))
        batch_report = {"first_draw": first, "count": count, "seed": args.seed + batch_index, "seconds": seconds, "metrics": values}
        report["batches"].append(batch_report)
        (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(batch_report), flush=True)


if __name__ == "__main__":
    main()
