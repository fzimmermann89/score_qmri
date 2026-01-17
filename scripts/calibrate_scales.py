# %%
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import yaml  # type: ignore[import-untyped, unused-ignore]
from jsonargparse import ArgumentParser

from score_qmri.datasets.lightning import MRIDataModule

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


def compute_dataset_stats(
    h5_path: str,
    num_samples: int = 100,
    seed: int = 42,
    strategy_fn: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        container: h5py.File | h5py.Group = f
        dataset_keys = [k for k in container if isinstance(container[k], h5py.Dataset)]
        if not dataset_keys:
            for group_name in f:
                group = f[group_name]
                if isinstance(group, h5py.Group):
                    container = group
                    dataset_keys = [k for k in container if isinstance(container[k], h5py.Dataset)]
                    break

        all_values = []
        samples_collected = 0

        keys = list(dataset_keys)
        rng.shuffle(keys)
        for key in keys:
            if samples_collected >= num_samples:
                break

            dataset = container[key]
            if not isinstance(dataset, h5py.Dataset):
                continue
            data = dataset[:]

            if data.ndim == 3:
                n_slices = min(data.shape[0], num_samples - samples_collected)
                indices = rng.choice(data.shape[0], size=n_slices, replace=False)
                for idx in indices:
                    slice_data = data[idx]
                    if np.iscomplexobj(slice_data):
                        slice_data = np.abs(slice_data)
                    mask = np.abs(slice_data) > 1e-6
                    if mask.any():
                        all_values.extend(slice_data[mask].flatten())
                    samples_collected += 1
            elif data.ndim == 4:
                n_slices = min(data.shape[0], num_samples - samples_collected)
                indices = rng.choice(data.shape[0], size=n_slices, replace=False)
                for idx in indices:
                    slice_data = data[idx]
                    if np.iscomplexobj(slice_data):
                        slice_data = np.abs(slice_data)
                    coil_idx = rng.integers(slice_data.shape[0])
                    coil_data = slice_data[coil_idx]
                    mask = np.abs(coil_data) > 1e-6
                    if mask.any():
                        all_values.extend(coil_data[mask].flatten())
                    samples_collected += 1

    if not all_values:
        return {"error": "No valid data found"}

    if strategy_fn is None:
        raise ValueError("strategy_fn must be provided")

    arr = np.array(all_values)
    recommended_scaling_factor = strategy_fn(arr)

    q99 = float(np.quantile(arr, 0.99))

    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q99": q99,
        "recommended_scaling_factor": recommended_scaling_factor,
        "num_samples": samples_collected,
    }


def analyze_datasets(
    data_module: MRIDataModule,
    strategy_fn: Callable[[np.ndarray], float],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    if not data_module.dataset_configs:
        print("No datasets found in config")
        return results

    for ds_config in data_module.dataset_configs:
        h5_path = ds_config.h5_path
        kind = ds_config.kind.name
        current_scaling_factor = ds_config.scaling_factor

        if not Path(h5_path).exists():
            print(f"Skipping {kind}: {h5_path} not found")
            continue

        print(f"Analyzing {kind} from {h5_path}...")
        stats = compute_dataset_stats(h5_path, NUM_SAMPLES_PER_DATASET, SEED, strategy_fn=strategy_fn)
        stats["current_scaling_factor"] = current_scaling_factor
        stats["h5_path"] = h5_path
        results[kind] = stats

    return results


def print_yaml_fragment(results: dict[str, dict[str, Any]]) -> None:
    print("\n# Recommended dataset scaling_factors (paste into config.yaml):")
    print("data:")
    print("  datasets:")

    for kind, stats in results.items():
        if "error" in stats:
            continue
        print(f"    # {kind}: q99={stats.get('q99', 'N/A'):.4f} max={stats.get('max', 'N/A'):.4f}")
        print(f"    - kind: {kind}")
        print(f"      h5_path: {stats['h5_path']}")
        print(f"      scaling_factor: {stats['recommended_scaling_factor']:.6f}")
        print()


def visualize_samples(h5_path: str, scaling_factor: float = 1.0, num_samples: int = 4) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return

    rng = np.random.default_rng(SEED)

    with h5py.File(h5_path, "r") as f:
        dataset_keys = [k for k in f if isinstance(f[k], h5py.Dataset)]
        if not dataset_keys:
            print("No datasets found")
            return

        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        keys = list(dataset_keys)

        for i in range(num_samples):
            key = rng.choice(np.array(keys, dtype=object))
            dataset = f[key]
            if not isinstance(dataset, h5py.Dataset):
                continue
            data = dataset[:]

            if data.ndim >= 3:
                z_idx = rng.integers(data.shape[0])
                slice_data = data[z_idx]
                if slice_data.ndim == 3:
                    slice_data = slice_data[0]
            else:
                slice_data = data

            if np.iscomplexobj(slice_data):
                slice_data = np.abs(slice_data)

            axes[0, i].imshow(slice_data, cmap="gray")
            axes[0, i].set_title(f"Original (max={slice_data.max():.2f})")
            axes[0, i].axis("off")

            scaled = slice_data * scaling_factor
            axes[1, i].imshow(scaled, cmap="gray", vmin=0, vmax=1.2)
            axes[1, i].set_title(f"Scaled (max={scaled.max():.2f})")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if STRATEGY_NAME not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {STRATEGY_NAME!r}. Available: {sorted(STRATEGIES)}")

    config = load_config(CONFIG_PATH)
    data_module = load_datamodule(config)
    results = analyze_datasets(data_module, strategy_fn=STRATEGIES[STRATEGY_NAME])

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    for kind, stats in results.items():
        print(f"\n{kind}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    print_yaml_fragment(results)
