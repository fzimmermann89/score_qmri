from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
import yaml  # type: ignore[import-untyped, unused-ignore]
from jsonargparse import ArgumentParser

from score_qmri.datasets.lightning import (
    ComplexDatasetConfig,
    DatasetConfig,
    MRIDataModule,
)  # type: ignore[import-untyped, unused-ignore]


class _SizedDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...


CONFIG_PATH = Path("config.yaml")
NUM_SAMPLES_PER_DATASET = 100
SEED = 42


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def load_datamodule(config: dict[str, Any]) -> MRIDataModule:
    parser = ArgumentParser()
    parser.add_class_arguments(MRIDataModule, "data")
    parsed = parser.parse_object({"data": config["data"]["init_args"]})
    instantiated = parser.instantiate_classes(parsed)
    return cast(MRIDataModule, instantiated.data)


def strategy_percentile(data: np.ndarray, q: float = 0.99) -> float:
    val = float(np.quantile(data, q))
    return 1.0 / val if val > 0 else 1.0


def strategy_mean_std(data: np.ndarray, num_std: float = 3.0) -> float:
    mean = float(np.mean(data))
    std = float(np.std(data))
    val = mean + num_std * std
    return 1.0 / val if val > 0 else 1.0


def strategy_robust_mad(data: np.ndarray, k: float = 1.4826, num_sigma: float = 3.0) -> float:
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    sigma_est = k * mad
    val = median + num_sigma * sigma_est
    return 1.0 / val if val > 0 else 1.0


def strategy_max(data: np.ndarray) -> float:
    val = float(np.max(data))
    return 1.0 / val if val > 0 else 1.0


STRATEGIES: dict[str, Callable[[np.ndarray], float]] = {
    "percentile": lambda d: strategy_percentile(d, q=0.99),
    "mean_std": lambda d: strategy_mean_std(d, num_std=3.0),
    "robust_mad": lambda d: strategy_robust_mad(d, num_sigma=3.0),
    "max": strategy_max,
}

STRATEGY_NAME = "percentile"


def _clone_dataset_config_for_calibration(cfg: DatasetConfig) -> DatasetConfig:
    return replace(cfg, augmentation_config=None, scaling_factor=1.0)


def _iter_signal_values(
    dataset: _SizedDataset,
    num_samples: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(len(dataset))
    if n == 0:
        return []

    k = min(num_samples, n)
    indices = rng.choice(n, size=k, replace=False)

    values: list[np.ndarray] = []
    for idx in indices:
        sample = dataset[int(idx)]
        image = sample["image"]
        background_mask = sample["background_mask"]
        signal_mask = ~background_mask

        mag = image.abs() if image.is_complex() else image
        vals = mag[signal_mask]
        vals = vals[vals > 1e-6]

        if vals.numel() > 0:
            values.append(vals.detach().cpu().numpy().astype(np.float64, copy=False))

    return values


def compute_dataset_stats(
    dataset: _SizedDataset,
    num_samples: int,
    seed: int,
    strategy_fn: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    value_chunks = _iter_signal_values(dataset, num_samples=num_samples, seed=seed)
    if not value_chunks:
        return {"error": "No valid data found"}

    arr = np.concatenate(value_chunks)
    scaling_factor = strategy_fn(arr)

    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q99": float(np.quantile(arr, 0.99)),
        "recommended_scaling_factor": scaling_factor,
        "num_samples": int(sum(x.size for x in value_chunks)),
    }


def analyze_datasets(data_module: MRIDataModule, strategy_fn: Callable[[np.ndarray], float]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    for ds_config in data_module.dataset_configs:
        kind = ds_config.kind.name
        key = f"{kind}/{ds_config.mode}" if isinstance(ds_config, ComplexDatasetConfig) else kind

        if not Path(ds_config.h5_path).exists():
            results[key] = {"error": f"File not found: {ds_config.h5_path}"}
            continue

        try:
            ds_for_cal = _clone_dataset_config_for_calibration(ds_config)
            dataset = cast(_SizedDataset, ds_for_cal.create())
        except Exception as e:
            results[key] = {"error": f"Failed to create dataset: {e}"}
            continue

        stats = compute_dataset_stats(dataset, num_samples=NUM_SAMPLES_PER_DATASET, seed=SEED, strategy_fn=strategy_fn)
        stats["h5_path"] = ds_config.h5_path
        stats["kind"] = kind
        stats["current_scaling_factor"] = ds_config.scaling_factor
        if isinstance(ds_config, ComplexDatasetConfig):
            stats["mode"] = ds_config.mode
        results[key] = stats

    return results


def print_yaml_fragment(results: dict[str, dict[str, Any]]) -> None:
    print("\n# Recommended dataset scaling_factors:")
    for key, stats in results.items():
        if "error" in stats:
            print(f"# {key}: {stats['error']}")
            continue
        mode = stats.get("mode")
        mode_str = f" mode={mode}" if mode is not None else ""
        print(f"# {key}:{mode_str} q99={stats['q99']:.4f} max={stats['max']:.4f}")
        print(f"#   h5_path: {stats['h5_path']}")
        print(f"#   current_scaling_factor: {stats['current_scaling_factor']}")
        print(f"#   recommended_scaling_factor: {stats['recommended_scaling_factor']:.6f}\n")


if __name__ == "__main__":
    if STRATEGY_NAME not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {STRATEGY_NAME!r}. Available: {sorted(STRATEGIES)}")

    torch.set_grad_enabled(False)

    cfg = load_config(CONFIG_PATH)
    dm = load_datamodule(cfg)
    res = analyze_datasets(dm, strategy_fn=STRATEGIES[STRATEGY_NAME])

    print_yaml_fragment(res)
