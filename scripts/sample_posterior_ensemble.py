"""Save independent batched draws for explicit frozen MRI posterior experiments."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from jsonargparse import ArgumentParser
from validate_paper_samplers import MaskedFourier, metrics

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import AncestralDPSConfig, AncestralDPSSampler, PaperDAPSConfig, PaperDAPSSampler


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
    parser.add_argument("--guidance-scale", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--prior-sigma", type=float, default=0.01)
    parser.add_argument("--langevin-lr", type=float, default=1.5e-5)
    parser.add_argument("--langevin-steps", type=int, default=100)
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--lr-min-ratio", type=float, default=1.0)
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
        AncestralDPSConfig(num_steps=args.steps or 1000, guidance_scale=args.guidance_scale)
        if args.sampler == "dps"
        else PaperDAPSConfig(
            tau=args.tau,
            anneal_steps=args.steps or 32,
            prior="score",
            prior_sigma=args.prior_sigma,
            langevin_lr=args.langevin_lr,
            langevin_steps=args.langevin_steps,
            reverse_steps=args.reverse_steps,
            lr_min_ratio=args.lr_min_ratio,
        )
    )
    sampler = AncestralDPSSampler(model, cfg) if args.sampler == "dps" else PaperDAPSSampler(model, cfg)
    source = Path(__file__).resolve().parents[1] / "src/score_qmri/sampling"
    report = {
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
