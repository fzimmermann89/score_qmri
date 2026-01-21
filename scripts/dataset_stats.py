"""Compute dataset intensity statistics after config scaling.

This script loads the same dataset configs used for training (including augmentation
and dataset-level scaling_factor) and then applies the model's per-kind
normalization (kind_scales for mapping kinds).

It prints per-dataset statistics over sampled slices:
- mean, std, min, max, 99th percentile

By default, statistics are computed over the signal region (~background_mask) to
avoid background zeros dominating.

For complex kinds, statistics are computed over the *real and imaginary channels
separately* (treating them as separate samples), not on magnitude.

Example
-------
python scripts/dataset_stats.py --config config.yaml --num_samples 200 --batch_size 8
"""

from __future__ import annotations

import contextlib
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from jsonargparse import ArgumentParser
from torch.utils.data import DataLoader, Subset

from score_qmri.datasets.kind import DatasetKind, is_complex_kind, is_mapping_kind
from score_qmri.datasets.lightning import Collate, ComplexDatasetConfig, DatasetConfig, MRIDataModule


def _seed(key: str, offset: int, base_seed: int) -> int:
    return (zlib.crc32(f"{key}:{offset}".encode()) + int(base_seed)) & 0xFFFFFFFF


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict-like config, got {type(cfg)}")
    return cfg


def _instantiate_datamodule(cfg: dict[str, Any]) -> MRIDataModule:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Missing/invalid 'data' section in config")

    init_args = data_cfg.get("init_args")
    if init_args is None:
        raise ValueError("Expected data.init_args in config")

    parser = ArgumentParser()
    parser.add_class_arguments(MRIDataModule, "data")
    parsed = parser.parse_object({"data": init_args})
    instantiated = parser.instantiate_classes(parsed)
    dm = instantiated.data
    if not isinstance(dm, MRIDataModule):
        raise TypeError(f"Expected MRIDataModule, got {type(dm)}")
    return dm


def _parse_kind_scales(cfg: dict[str, Any]) -> dict[int, float]:
    # Defaults mirrored from repo config convention.
    default: dict[str, float] = {
        "T1MAP": 0.0003,
        "T2MAP": 0.005,
        "T2STARMAP": 0.01,
    }
    model_obj = cfg.get("model")
    model_cfg: dict[str, Any] = model_obj if isinstance(model_obj, dict) else {}

    norm_obj = model_cfg.get("normalization")
    norm_cfg: dict[str, Any] = norm_obj if isinstance(norm_obj, dict) else {}

    scales_obj = norm_cfg.get("kind_scales")
    kind_scales: dict[str, Any] = scales_obj if isinstance(scales_obj, dict) else {}

    merged: dict[str, float] = {**default, **{str(k): float(v) for k, v in kind_scales.items()}}
    out: dict[int, float] = {}
    for name, scale in merged.items():
        if name not in DatasetKind.__members__:
            continue
        out[int(DatasetKind[name])] = float(scale)
    return out


def _parse_train_clip(cfg: dict[str, Any], default: float = 1.2) -> float:
    model_obj = cfg.get("model")
    model_cfg: dict[str, Any] = model_obj if isinstance(model_obj, dict) else {}

    norm_obj = model_cfg.get("normalization")
    norm_cfg: dict[str, Any] = norm_obj if isinstance(norm_obj, dict) else {}

    train_clip = norm_cfg.get("train_clip", default)
    return float(train_clip)


def _dataset_label(i: int, ds_cfg: DatasetConfig) -> str:
    fname = Path(ds_cfg.h5_path).name
    kind = ds_cfg.kind.name
    if isinstance(ds_cfg, ComplexDatasetConfig):
        return f"{i:03d} {kind}/{ds_cfg.mode}:{fname}"
    return f"{i:03d} {kind}:{fname}"


def _close_dataset_file(ds: object) -> None:
    h5_file = getattr(ds, "_h5_file", None)
    if h5_file is None:
        return
    with contextlib.suppress(Exception):
        h5_file.close()


def _kind_normalize(x: torch.Tensor, kind: torch.Tensor, kind_scales: dict[int, float]) -> torch.Tensor:
    # x: (B, 2, H, W)
    out = x
    for k in kind.unique():
        k_idx = int(k.item())
        if not is_mapping_kind(k_idx):
            continue
        scale = float(kind_scales.get(k_idx, 1.0))
        out = out.clone() if out is x else out
        out[kind == k] = out[kind == k] * scale
    return out


def _clamp_x0(x: torch.Tensor, kind: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamp per-kind, preserving phase for complex kinds.

    Mirrors `DiffusionModel.clamp_x0` behavior (without depending on model instantiation).
    """
    from score_qmri.datasets.kind import is_complex_kind

    result = x.clone()
    eps = 1e-8
    for k in kind.unique():
        k_idx = int(k.item())
        idx = kind == k
        if is_complex_kind(k_idx):
            x_k = result[idx]
            mag = (x_k[:, 0] ** 2 + x_k[:, 1] ** 2).sqrt()
            mag_clipped = mag.clamp(0.0, limit)
            scale_factor = mag_clipped / (mag + eps)
            result[idx, 0] = x_k[:, 0] * scale_factor
            result[idx, 1] = x_k[:, 1] * scale_factor
        else:
            result[idx, 0] = result[idx, 0].clamp(0.0, limit)
            result[idx, 1] = 0.0
    return result


def _values_for_stats(
    x_2ch: torch.Tensor,
    kind: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    signal_only: bool,
) -> torch.Tensor:
    """Extract values for statistics.

    - Complex kinds: treat real and imag channels as separate samples.
    - Real kinds: use only channel 0.

    Parameters
    ----------
    x_2ch
        Tensor of shape (B, 2, H, W).
    kind
        Kind indices of shape (B,).
    background_mask
        Boolean mask of shape (B, H, W).
    signal_only
        If True, restrict stats to ~background_mask.

    Returns
    -------
    Tensor
        Flattened values.
    """
    sig = ~background_mask if signal_only else torch.ones_like(background_mask, dtype=torch.bool)

    chunks: list[torch.Tensor] = []
    for k in kind.unique():
        k_idx = int(k.item())
        sel = kind == k
        if is_complex_kind(k_idx):
            x_k = x_2ch[sel]  # (Bk, 2, H, W)
            sig_k = sig[sel].unsqueeze(1).expand(-1, 2, -1, -1)
            chunks.append(x_k[sig_k])
        else:
            # Real kinds: use channel 0 only.
            x_k = x_2ch[sel, 0]  # (Bk, H, W)
            chunks.append(x_k[sig[sel]])

    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks, dim=0)


def _compute_stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    v = values.float()
    return {
        "min": float(v.min().item()),
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()),
        "p99": float(torch.quantile(v, 0.99).item()),
        "max": float(v.max().item()),
    }


def _format_stats(stats: dict[str, float]) -> str:
    if not stats:
        return "(no values)"
    return " ".join(f"{k}={stats[k]:.6g}" for k in ["min", "mean", "std", "p99", "max"])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal_only", type=bool, default=True)
    parser.add_argument("--apply_train_clip", type=bool, default=False)
    args = parser.parse_args()

    config_path: Path = args.config
    num_samples: int = args.num_samples
    batch_size: int = args.batch_size
    num_workers: int = args.num_workers
    seed: int = args.seed
    signal_only: bool = args.signal_only
    apply_train_clip: bool = args.apply_train_clip

    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_cfg = _load_yaml(config_path)
    dm = _instantiate_datamodule(raw_cfg)
    kind_scales = _parse_kind_scales(raw_cfg)
    train_clip = _parse_train_clip(raw_cfg)

    dm.setup("fit")
    target_h, target_w = dm.sample_size

    print(f"config: {config_path}")
    print(f"sample_size: {target_h}x{target_w}")
    print(f"num_samples_per_dataset: {num_samples}")
    print(f"batch_size: {batch_size} num_workers: {num_workers}")
    print(f"signal_only: {signal_only} apply_train_clip: {apply_train_clip} (train_clip={train_clip})")
    print("")

    for i, ds_cfg in enumerate(dm.dataset_configs):
        label = _dataset_label(i, ds_cfg)
        h5_path = Path(ds_cfg.h5_path)
        if not h5_path.exists():
            print(f"{label}  -> missing file, skipping")
            continue

        ds = ds_cfg.create()
        try:
            n = len(ds)
            if n == 0:
                print(f"{label}  -> empty dataset")
                continue

            k = min(num_samples, n)
            rng = np.random.default_rng(_seed(label, 0, seed))
            idxs = rng.choice(n, size=k, replace=False).astype(np.int64)

            subset = Subset(ds, idxs.tolist())
            collate = Collate(target_h, target_w)
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate,
                pin_memory=False,
            )

            vals_list: list[torch.Tensor] = []

            for batch in loader:
                x = batch["image"]  # (B, 2, H, W)
                kind = batch["kind"]  # (B,)
                background_mask = batch["background_mask"]  # (B, H, W)

                x = _kind_normalize(x, kind, kind_scales)
                if apply_train_clip:
                    x = _clamp_x0(x, kind, limit=train_clip)

                vals = _values_for_stats(x, kind, background_mask, signal_only=signal_only)
                if vals.numel() > 0:
                    vals_list.append(vals.detach().cpu())

            if not vals_list:
                print(f"{label}  -> no signal values")
                continue

            values = torch.cat(vals_list, dim=0)
            stats = _compute_stats(values)

            scale_str = f"scaling_factor={getattr(ds_cfg, 'scaling_factor', 1.0)}"
            aug_str = "aug=on" if ds_cfg.augmentation_config is not None else "aug=off"
            print(f"{label}  {scale_str} {aug_str}  n_slices={n} sampled={k}  {_format_stats(stats)}")
        except Exception as ex:
            print(f"{label}  -> failed: {type(ex).__name__}: {ex}")
        finally:
            _close_dataset_file(ds)


if __name__ == "__main__":
    main()
