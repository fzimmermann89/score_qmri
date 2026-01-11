"""Debug why logged samples are black.

This script is meant to be run on the training server.

It inspects:
- Dataset(s) for a given DatasetKind: value range, mask coverage, augmentation effects.
- Checkpoint normalization buffers (RunningNormalizer): mean/std/count per kind.
- (Optional) DDIM sampling output statistics and how denormalization differs between
  the training model and EMA shadow.

Examples
--------
python debug_black_samples.py --config config.yaml --kind COMPLEX_T1W_COIL_COMBINED --n 8
python debug_black_samples.py --config config.yaml --checkpoint /path/to/epoch=...ckpt --kind COMPLEX_T1W_COIL_COMBINED --sample
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from score_qmri.datasets.config import AugmentationConfig
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import (
    ComplexDatasetConfig,
    ConstantSchedule,
    DatasetConfig,
    MagnitudeDatasetConfig,
    MappingDatasetConfig,
    PhaseInSchedule,
    PhaseOutSchedule,
)
from score_qmri.sampling.ddim_sampler import DDIMSampler, create_fixed_noise
from score_qmri.training.lightning_module import DiffusionLightningModule


@dataclass(frozen=True)
class ImageStats:
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float


def _as_tuple2(x: Any) -> tuple[Any, Any]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return (x[0], x[1])
    raise ValueError(f"Expected a 2-element list/tuple, got: {x!r}")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping, got: {type(data)}")
    return data


def _parse_schedule(raw: dict[str, Any] | None) -> ConstantSchedule | PhaseInSchedule | PhaseOutSchedule:
    if raw is None:
        return ConstantSchedule()

    schedule_type = raw.get("type")
    if schedule_type == "constant":
        return ConstantSchedule(weight=float(raw.get("weight", 1.0)))
    if schedule_type == "phase_in":
        return PhaseInSchedule(steps=int(raw["steps"]), weight=float(raw.get("weight", 1.0)))
    if schedule_type == "phase_out":
        return PhaseOutSchedule(steps=int(raw["steps"]), weight=float(raw.get("weight", 1.0)))

    raise ValueError(f"Unknown schedule type: {schedule_type!r}")


def _parse_aug(raw: dict[str, Any] | None) -> AugmentationConfig | None:
    if raw is None:
        return None

    # Convert common list fields to tuples to match type expectations.
    raw = dict(raw)
    for key in [
        "shift_range",
        "scale_range",
        "rotation_range",
        "shear_range",
        "poly_coeff_range",
    ]:
        if key in raw:
            raw[key] = _as_tuple2(raw[key])

    return AugmentationConfig(**raw)


def _resolve_h5_path(h5_path: str, config_dir: Path) -> str:
    p = Path(h5_path)
    if p.is_absolute():
        return str(p)
    return str((config_dir / p).resolve())


def _parse_dataset_configs(raw: list[dict[str, Any]], config_dir: Path, disable_aug: bool) -> list[DatasetConfig]:
    configs: list[DatasetConfig] = []

    for entry in raw:
        ds_type = entry["type"]
        kind_raw = entry["kind"]
        kind = DatasetKind[kind_raw] if isinstance(kind_raw, str) else DatasetKind(int(kind_raw))

        augmentation_config = None if disable_aug else _parse_aug(entry.get("augmentation_config"))
        schedule = _parse_schedule(entry.get("schedule"))
        h5_path = _resolve_h5_path(str(entry["h5_path"]), config_dir)
        scaling_factor = float(entry.get("scaling_factor", 1.0))

        if ds_type == "complex":
            mode = entry.get("mode", "flat")
            configs.append(
                ComplexDatasetConfig(
                    h5_path=h5_path,
                    mode=mode,
                    kind=kind,
                    scaling_factor=scaling_factor,
                    augmentation_config=augmentation_config,
                    schedule=schedule,
                )
            )
        elif ds_type == "magnitude":
            configs.append(
                MagnitudeDatasetConfig(
                    h5_path=h5_path,
                    kind=kind,
                    scaling_factor=scaling_factor,
                    augmentation_config=augmentation_config,
                    schedule=schedule,
                )
            )
        elif ds_type == "mapping":
            configs.append(
                MappingDatasetConfig(
                    h5_path=h5_path,
                    kind=kind,
                    scaling_factor=scaling_factor,
                    augmentation_config=augmentation_config,
                    schedule=schedule,
                )
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type!r}")

    return configs


def _compute_stats(x: np.ndarray) -> ImageStats:
    x_f = x.astype(np.float64, copy=False)
    return ImageStats(
        min=float(np.min(x_f)),
        max=float(np.max(x_f)),
        mean=float(np.mean(x_f)),
        std=float(np.std(x_f)),
        p01=float(np.percentile(x_f, 1)),
        p50=float(np.percentile(x_f, 50)),
        p99=float(np.percentile(x_f, 99)),
    )


def _print_stats(label: str, stats: ImageStats) -> None:
    print(
        f"{label}: min={stats.min:.3g} max={stats.max:.3g} mean={stats.mean:.3g} std={stats.std:.3g} "
        f"p01={stats.p01:.3g} p50={stats.p50:.3g} p99={stats.p99:.3g}"
    )


def _extract_normalizer_prefix(state_dict: dict[str, Any]) -> str | None:
    # We look for keys like:
    # - "model.normalizer.mean_0" (LightningModule state_dict)
    # - "normalizer.mean_0" (DiffusionModel state_dict)
    for key in state_dict.keys():
        if key.endswith("normalizer.mean_0"):
            return key[: -len(".mean_0")]
    for key in state_dict.keys():
        if key.endswith(".normalizer.mean_0"):
            return key[: -len(".mean_0")]
    return None


def _print_checkpoint_normalizer_stats(ckpt_path: Path, kind: DatasetKind) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint has no state_dict")

    k = int(kind)

    print("\n== Checkpoint normalizer stats ==")

    prefix = _extract_normalizer_prefix(state_dict)
    if prefix is None:
        print("Could not find normalizer buffers in checkpoint state_dict")
        return

    mean = state_dict.get(f"{prefix}.mean_{k}")
    var = state_dict.get(f"{prefix}.var_{k}")
    n = state_dict.get(f"{prefix}.n_{k}")

    if mean is None or var is None or n is None:
        print(f"Missing buffers for kind={kind.name} under prefix {prefix!r}")
    else:
        std = var.sqrt().clamp(min=1e-12)
        print(
            f"model.normalizer[{kind.name}] n={int(n.item())} "
            f"mean={mean.tolist()} std={std.tolist()}"
        )

    ema = ckpt.get("ema")
    if not isinstance(ema, dict):
        print("No EMA state in checkpoint (ckpt['ema'] missing)")
        return

    shadow = ema.get("shadow")
    if not isinstance(shadow, dict):
        print("ckpt['ema'] has no 'shadow' state_dict")
        return

    ema_prefix = _extract_normalizer_prefix(shadow)
    if ema_prefix is None:
        print("Could not find normalizer buffers in EMA shadow")
        return

    ema_mean = shadow.get(f"{ema_prefix}.mean_{k}")
    ema_var = shadow.get(f"{ema_prefix}.var_{k}")
    ema_n = shadow.get(f"{ema_prefix}.n_{k}")

    if ema_mean is None or ema_var is None or ema_n is None:
        print(f"Missing EMA buffers for kind={kind.name} under prefix {ema_prefix!r}")
        return

    ema_std = ema_var.sqrt().clamp(min=1e-12)
    print(
        f"ema.shadow.normalizer[{kind.name}] n={int(ema_n.item())} "
        f"mean={ema_mean.tolist()} std={ema_std.tolist()}"
    )


def _inspect_datasets(config_path: Path, kind: DatasetKind, n: int, disable_aug: bool, seed: int) -> None:
    cfg = _load_yaml(config_path)
    data_block = cfg.get("data", {})
    init_args = data_block.get("init_args", {})
    raw_ds_cfgs = init_args.get("dataset_configs")

    if not isinstance(raw_ds_cfgs, list):
        raise KeyError("Expected config['data']['init_args']['dataset_configs'] to be a list")

    config_dir = config_path.parent
    ds_configs = _parse_dataset_configs(raw_ds_cfgs, config_dir=config_dir, disable_aug=disable_aug)

    matching = [d for d in ds_configs if d.kind == kind]

    print("== Dataset configs for kind ==")
    print(f"kind={kind.name} ({int(kind)})")
    print(f"count={len(matching)}")
    for i, d in enumerate(matching):
        if isinstance(d, ComplexDatasetConfig):
            mode = d.mode
        else:
            mode = "-"
        print(
            f"[{i}] type={d.type} mode={mode} scaling_factor={getattr(d, 'scaling_factor', 1.0)} "
            f"aug={'off' if disable_aug else ('on' if d.augmentation_config is not None else 'none')} "
            f"h5_path={d.h5_path}"
        )

    if not matching:
        return

    rng = np.random.default_rng(seed)

    print("\n== Dataset sample stats ==")
    for i, d in enumerate(matching):
        ds = d.create()
        length = len(ds)
        if length == 0:
            print(f"[{i}] {d.h5_path}: empty dataset")
            continue

        idxs = rng.integers(low=0, high=length, size=min(n, length), endpoint=False)
        print(f"\n[{i}] {d.h5_path} (len={length})")

        abs_all: list[np.ndarray] = []
        abs_valid: list[np.ndarray] = []
        mask_fracs: list[float] = []

        for idx in idxs:
            sample = ds[int(idx)]
            img_t: torch.Tensor = sample["image"]
            mask_t: torch.Tensor = sample["mask"]

            img_abs = img_t.abs() if img_t.is_complex() else img_t
            mask = mask_t.bool()

            mask_frac = float(mask.float().mean().item())
            mask_fracs.append(mask_frac)

            img_abs_np = img_abs.detach().cpu().numpy()
            abs_all.append(img_abs_np.reshape(-1))
            if mask.any():
                abs_valid.append(img_abs[mask].detach().cpu().numpy().reshape(-1))

        abs_all_cat = np.concatenate(abs_all)
        _print_stats("abs(all)", _compute_stats(abs_all_cat))
        if abs_valid:
            abs_valid_cat = np.concatenate(abs_valid)
            _print_stats("abs(masked)", _compute_stats(abs_valid_cat))
        else:
            print("abs(masked): no valid pixels")

        print(
            f"mask_frac: mean={float(np.mean(mask_fracs)):.3f} min={float(np.min(mask_fracs)):.3f} "
            f"max={float(np.max(mask_fracs)):.3f}"
        )


def _maybe_sample(checkpoint: Path, kind: DatasetKind, device: str, size: int, steps: int) -> None:
    print("\n== DDIM sampling check ==")
    module = DiffusionLightningModule.load_from_checkpoint(str(checkpoint), map_location=device)
    module.eval()

    # Sampling like training: it uses EMA if present. For debugging, compare:
    # - denormalization using module.model stats
    # - denormalization using EMA shadow stats loaded from checkpoint (if present)
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    ema_shadow = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("ema"), dict):
        shadow_sd = ckpt["ema"].get("shadow")
        if isinstance(shadow_sd, dict):
            ema_shadow = DiffusionLightningModule.load_from_checkpoint(str(checkpoint), map_location=device).model
            # Replace weights with EMA shadow weights.
            # EMA shadow state_dict key format matches DiffusionModel state_dict.
            ema_shadow.load_state_dict(shadow_sd, strict=True)
            ema_shadow.eval()

    k = int(kind)
    kind_tensor = torch.tensor([k], device=device)
    fixed = create_fixed_noise((size, size))[k].to(device)

    def run_one(model: torch.nn.Module, denorm_model: DiffusionLightningModule) -> None:
        sampler = DDIMSampler(model, num_steps=steps)
        with torch.no_grad():
            sample = sampler.sample((1, size, size), kind_tensor, fixed)
            den = denorm_model.model.denormalize(sample, kind_tensor)
            img = den[0].abs().detach().cpu().numpy() if den.is_complex() else den[0].detach().cpu().numpy()
        _print_stats(f"denorm(abs) using stats from module.model", _compute_stats(img.reshape(-1)))

    # 1) Sample with module.model weights.
    model = module.model.to(device)
    model.eval()
    print(f"Sampling kind={kind.name} with module.model weights")
    run_one(model, module)

    # 2) Sample with EMA shadow weights (if available) but still denormalize using module.model stats.
    if ema_shadow is not None:
        print(f"Sampling kind={kind.name} with EMA shadow weights")
        run_one(ema_shadow.to(device), module)
    else:
        print("No EMA shadow found in checkpoint; skipping EMA sampling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--kind", type=str, default="COMPLEX_T1W_COIL_COMBINED")
    parser.add_argument("--n", type=int, default=8, help="Samples per matching dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_aug", action="store_true")

    parser.add_argument("--sample", action="store_true", help="Run DDIM sampling from checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=250)

    args = parser.parse_args()

    kind = DatasetKind[args.kind]
    config_path = Path(args.config).resolve()

    _inspect_datasets(config_path, kind=kind, n=args.n, disable_aug=args.disable_aug, seed=args.seed)

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint).resolve()
        _print_checkpoint_normalizer_stats(ckpt_path, kind=kind)

        if args.sample:
            _maybe_sample(ckpt_path, kind=kind, device=args.device, size=args.size, steps=args.steps)


if __name__ == "__main__":
    main()
