"""Evaluate unclipped main-head denoising and local score consistency on frozen cases."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from jsonargparse import ArgumentParser
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import MRIDataModule
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling._utils import to_2ch
from score_qmri.training.losses import create_channel_mask
from score_qmri.training.validation import backbone_gradient_diagnostics, validation_reference


@torch.enable_grad()
def score_diagnostics(
    model: DiffusionModel, clean: Tensor, kind: Tensor, sigma: float, noise: Tensor, probe: Tensor, *, roi: Tensor | None = None
) -> dict[str, float]:
    """Check a normalized (B,2,H,W) batch using explicit noise and direction.

    The model must be in evaluation mode. An optional boolean (B,1,H,W) ROI
    restricts both sides of the Jacobian to that coordinate block. Directional
    probes can disprove symmetry/PSD, but cannot certify them. A single noisy
    target does not provide a ground-truth score or establish calibration.
    """
    if not isfinite(sigma) or not model.sigma_min <= sigma <= model.sigma_max:
        raise ValueError("sigma must lie in the model's supported range")
    if clean.ndim != 4 or clean.shape[1] != 2 or noise.shape != clean.shape or probe.shape != clean.shape:
        raise ValueError("clean, noise and probe must have matching (B,2,H,W) shapes")
    if roi is not None and (roi.dtype != torch.bool or roi.shape != (len(clean), 1, *clean.shape[-2:])):
        raise ValueError("roi must be boolean with shape (B,1,H,W)")
    channel = create_channel_mask(kind).to(clean)
    selected = channel.expand_as(clean) * (roi if roi is not None else 1)
    direction = probe * selected
    if not torch.isfinite(clean).all() or not torch.isfinite(noise).all() or not torch.isfinite(direction).all():
        raise ValueError("Diagnostic inputs must be finite")
    if (direction.square().flatten(1).sum(1) == 0).any():
        raise ValueError("Each case requires a nonzero probe in the selected coordinates")
    point = ((clean + sigma * noise) * channel).detach().requires_grad_(True)
    sigmas = point.new_full((len(point),), sigma)

    def denoise(x: Tensor) -> Tensor:
        return model.denoise(x, sigmas, kind)

    # Double backward for the exact JVP is supported by math SDPA.
    with torch.autocast(device_type=point.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        prediction, jvp = torch.autograd.functional.jvp(denoise, point, direction)
        (vjp,) = torch.autograd.grad(denoise(point), point, direction)
    jvp, vjp = jvp * selected, vjp * selected
    score_jvp = (jvp - direction) / sigma**2
    score_vjp = (vjp - direction) / sigma**2
    skew = score_jvp - score_vjp
    count = selected.sum()
    mse = ((prediction - clean).square() * selected).sum() / count
    c_skip = model.sigma_data**2 / (sigma**2 + model.sigma_data**2)
    baseline_mse = ((c_skip * point - clean).square() * selected).sum() / count
    weight = (sigma**2 + model.sigma_data**2) / (sigma * model.sigma_data) ** 2
    rayleigh = (direction * jvp).flatten(1).sum(1) / direction.square().flatten(1).sum(1)
    result = {
        "main_edm_mse": float(weight * mse),
        "zero_network_edm_mse": float(weight * baseline_mse.detach()),
        "mse_over_noise_variance": float(mse / sigma**2),
        "score_jacobian_relative_skew": float(skew.norm() / (0.5 * (score_jvp.norm() + score_vjp.norm())).clamp_min(1e-30)),
        "score_jacobian_skew_rms": float((skew.square().sum() / count).sqrt()),
        "denoiser_rayleigh_min": float(rayleigh.min()),
    }
    if not all(isfinite(value) for value in result.values()):
        raise RuntimeError("Nonfinite denoiser diagnostic")
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", type=Path, help="NPZ truth in normalized model units, complex (N,1,1,H,W)")
    source.add_argument("--data-config", type=Path, help="data.yaml with explicit validation_samples")
    parser.add_argument("--kind", choices=[kind.name for kind in DatasetKind], default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--sigmas", type=list[float], default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2, 2.0])
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--roi-key", type=str, default=None, help="Optional NPZ boolean (N,H,W) ROI for foreground/block checks")
    parser.add_argument("--backbone-gradients", action="store_true", help="Measure shared-feature gradients once per sigma/case")
    parser.add_argument("--aux-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.probes < 1 or args.output.exists():
        raise ValueError("Require positive probes and a new output path")
    model = DiffusionModel.from_checkpoint(args.checkpoint, map_location="cpu", weights=args.weights, compile_model=False).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    batches = []
    rois = []
    if args.data_config is not None:
        data_parser = ArgumentParser()
        data_parser.add_subclass_arguments(MRIDataModule, "data", required=True)
        data_parser.add_argument("--trainer", type=dict, default={})
        datamodule = data_parser.instantiate_classes(data_parser.parse_path(args.data_config)).data
        datamodule.dataset_configs = [config for config in datamodule.dataset_configs if config.validation_samples]
        datamodule.setup("fit")
        if datamodule.validation_batch is None:
            raise ValueError("data.yaml must configure fixed validation_samples")
        for index in range(len(datamodule.validation_batch["image"])):
            batches.append({key: value[index : index + 1].to(args.device) for key, value in datamodule.validation_batch.items()})
            rois.append(~batches[-1]["background_mask"].unsqueeze(1))
    else:
        if args.kind is None:
            raise ValueError("--cases requires --kind")
        with np.load(args.cases) as cases:
            for index, image in enumerate(cases["truth"]):
                clean = to_2ch(torch.from_numpy(image[None]).to(args.device))
                batches.append(
                    {
                        "image": clean,
                        "kind": torch.tensor([int(DatasetKind[args.kind])], device=args.device),
                        "background_mask": torch.zeros_like(clean[:, 0], dtype=torch.bool),
                        "penumbra_mask": torch.zeros_like(clean[:, 0], dtype=torch.bool),
                    }
                )
                rois.append(
                    None if args.roi_key is None else torch.from_numpy(cases[args.roi_key][index : index + 1, None]).to(args.device)
                )
    rows = []
    for index, (batch, roi) in enumerate(zip(batches, rois, strict=True)):
        clean = validation_reference(model, batch) if args.data_config is not None else batch["image"]
        kind = batch["kind"]
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        for probe_index in range(args.probes):
            probe = torch.randint(0, 2, clean.shape, device=clean.device, generator=generator).to(clean) * 2 - 1
            for sigma in args.sigmas:
                rows.append(
                    {
                        "case": index,
                        "probe": probe_index,
                        "sigma": sigma,
                        **score_diagnostics(model, clean, kind, sigma, noise, probe, roi=roi),
                        **(
                            backbone_gradient_diagnostics(model, batch, sigma, noise, aux_weight=args.aux_weight)
                            if args.backbone_gradients and probe_index == 0
                            else {}
                        ),
                    }
                )
    report = {
        "scope": "Local diagnostics; verify subject-held-out provenance separately. Passing probes is not a posterior calibration certificate.",
        "arguments": vars(args),
        "checkpoint": model.checkpoint_info,
        "case_sha256": hashlib.sha256((args.data_config or args.cases).read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
