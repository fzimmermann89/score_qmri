# %%
from __future__ import annotations

import contextlib
import zlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml  # type: ignore[import-untyped, unused-ignore]
from jsonargparse import ArgumentParser

from score_qmri.datasets.lightning import (  # type: ignore[import-untyped, unused-ignore]
    ComplexDatasetConfig,
    DatasetConfig,
    MRIDataModule,
)


class _SizedDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DatasetEntry:
    idx: int
    key: str
    cfg: DatasetConfig


@dataclass(frozen=True)
class CachedSamples:
    plot_images: list[np.ndarray]
    values: np.ndarray


# %%
CONFIG_PATH = Path("config.yaml")
SEED = 42

NUM_STATS_SAMPLES = 100
NUM_PLOT_SAMPLES = 8
DISPLAY_VMAX = 1.2

STRATEGY_ORDER = ["p99", "mean+3std", "median+3mad", "max"]


# %%
def _seed(key: str, offset: int) -> int:
    return (zlib.crc32(f"{key}:{offset}".encode()) + SEED) & 0xFFFFFFFF


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict-like config, got {type(cfg)}")
    return cfg


def load_datamodule(cfg: dict[str, Any]) -> MRIDataModule:
    parser = ArgumentParser()
    parser.add_class_arguments(MRIDataModule, "data")
    parsed = parser.parse_object({"data": cfg["data"]["init_args"]})
    instantiated = parser.instantiate_classes(parsed)
    dm = instantiated.data
    if not isinstance(dm, MRIDataModule):
        raise TypeError(f"Expected MRIDataModule, got {type(dm)}")
    return dm


def entry_label(cfg: DatasetConfig) -> str:
    fname = Path(cfg.h5_path).name
    kind = cfg.kind.name
    if isinstance(cfg, ComplexDatasetConfig):
        return f"{kind}/{cfg.mode}:{fname}"
    return f"{kind}:{fname}"


def clone_cfg_for_calibration(cfg: DatasetConfig) -> DatasetConfig:
    return replace(cfg, augmentation_config=None, scaling_factor=1.0)


def close_dataset_file(ds: object) -> None:
    h5_file = getattr(ds, "_h5_file", None)
    if h5_file is None:
        return
    with contextlib.suppress(Exception):
        h5_file.close()


# %%
def magnitude(sample: dict[str, Any]) -> torch.Tensor:
    img = cast(torch.Tensor, sample["image"])
    mag = img.abs() if img.is_complex() else img
    return mag.float()


def sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=min(k, n), replace=False).astype(np.int64)


def collect_signal_values(dataset: _SizedDataset, idxs: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for idx in idxs:
        s = dataset[int(idx)]
        sig = ~cast(torch.Tensor, s["background_mask"]).bool()
        vals = magnitude(s)[sig]
        vals = vals[vals > 1e-6]
        if vals.numel() > 0:
            chunks.append(vals.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)


def collect_images(dataset: _SizedDataset, idxs: np.ndarray) -> list[np.ndarray]:
    return [magnitude(dataset[int(idx)]).detach().cpu().numpy() for idx in idxs]


# %%
def scale_from_percentile(values: np.ndarray, q: float = 0.99) -> float:
    v = float(np.quantile(values, q))
    return 1.0 / v if v > 0 else 1.0


def scale_from_mean_std(values: np.ndarray, num_std: float = 3.0) -> float:
    v = float(values.mean() + num_std * values.std())
    return 1.0 / v if v > 0 else 1.0


def scale_from_robust_mad(values: np.ndarray, num_sigma: float = 3.0) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma_est = 1.4826 * mad
    v = median + num_sigma * sigma_est
    return 1.0 / v if v > 0 else 1.0


def scale_from_max(values: np.ndarray) -> float:
    v = float(values.max())
    return 1.0 / v if v > 0 else 1.0


STRATEGIES: dict[str, Callable[[np.ndarray], float]] = {
    "p99": lambda x: scale_from_percentile(x, q=0.99),
    "mean+3std": lambda x: scale_from_mean_std(x, num_std=3.0),
    "median+3mad": lambda x: scale_from_robust_mad(x, num_sigma=3.0),
    "max": scale_from_max,
}


# %%
cfg = load_config(CONFIG_PATH)
dm = load_datamodule(cfg)

ENTRIES: list[DatasetEntry] = [DatasetEntry(idx=i, key=f"{i:03d} {entry_label(c)}", cfg=c) for i, c in enumerate(dm.dataset_configs)]

CURRENT_INDEX = 0

SCALING_FACTORS: dict[int, float] = {}

_SAMPLE_CACHE: dict[int, CachedSamples] = {}
_SAMPLE_VERSION: dict[int, int] = {}


# %%
def current_entry() -> DatasetEntry:
    return ENTRIES[CURRENT_INDEX]


def goto(i: int) -> DatasetEntry:
    global CURRENT_INDEX
    CURRENT_INDEX = int(np.clip(i, 0, len(ENTRIES) - 1))
    return current_entry()


def next_dataset() -> DatasetEntry:
    return goto(CURRENT_INDEX + 1)


def prev_dataset() -> DatasetEntry:
    return goto(CURRENT_INDEX - 1)


def undecided_indices() -> list[int]:
    return [e.idx for e in ENTRIES if e.idx not in SCALING_FACTORS]


def next_undecided() -> DatasetEntry:
    global CURRENT_INDEX
    for _ in range(len(ENTRIES)):
        CURRENT_INDEX = (CURRENT_INDEX + 1) % len(ENTRIES)
        if ENTRIES[CURRENT_INDEX].idx not in SCALING_FACTORS:
            break
    return current_entry()


def set_scaling_factor(value: float) -> None:
    SCALING_FACTORS[current_entry().idx] = float(value)


def accept_strategy(name: str) -> float:
    scale = suggestions(current_entry())[name]
    set_scaling_factor(scale)
    return scale


def clear_cache(idx: int | None = None) -> None:
    if idx is None:
        _SAMPLE_CACHE.clear()
    else:
        _SAMPLE_CACHE.pop(idx, None)


def resample_current() -> None:
    idx = current_entry().idx
    _SAMPLE_VERSION[idx] = _SAMPLE_VERSION.get(idx, 0) + 1
    clear_cache(idx)


# %%
def get_samples(e: DatasetEntry) -> CachedSamples:
    if e.idx in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[e.idx]

    if not Path(e.cfg.h5_path).exists():
        raise FileNotFoundError(e.cfg.h5_path)

    ds_cfg = clone_cfg_for_calibration(e.cfg)
    ds = cast(_SizedDataset, ds_cfg.create())
    try:
        n = len(ds)
        version = _SAMPLE_VERSION.get(e.idx, 0)
        idxs_stats = sample_indices(n, NUM_STATS_SAMPLES, seed=_seed(e.key, 0 + 2 * version))
        idxs_plot = sample_indices(n, NUM_PLOT_SAMPLES, seed=_seed(e.key, 1 + 2 * version))

        values = collect_signal_values(ds, idxs_stats)
        plot_images = collect_images(ds, idxs_plot)
    finally:
        close_dataset_file(ds)

    cached = CachedSamples(plot_images=plot_images, values=values)
    _SAMPLE_CACHE[e.idx] = cached
    return cached


def suggestions(e: DatasetEntry) -> dict[str, float]:
    samples = get_samples(e)
    if samples.values.size == 0:
        return {}

    ordered = [k for k in STRATEGY_ORDER if k in STRATEGIES]
    return {name: STRATEGIES[name](samples.values) for name in ordered}


def value_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


# %%
def scaled(images: list[np.ndarray], factor: float) -> list[np.ndarray]:
    return [img * float(factor) for img in images]


def imshow_row(axs: list[Any], images: list[np.ndarray], label: str) -> None:
    for ax, img in zip(axs, images, strict=False):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=DISPLAY_VMAX)
        ax.axis("off")
    for ax in axs[len(images) :]:
        ax.axis("off")
    axs[0].set_ylabel(label, rotation=0, ha="right", va="center")


def plot_current(*, chosen_scale: float | None = None, show_decided: bool = True, max_decided: int = 6) -> None:
    e = current_entry()
    print(f"[{CURRENT_INDEX + 1}/{len(ENTRIES)}] {e.key}")
    print("h5_path:", e.cfg.h5_path)
    print("config scaling_factor:", e.cfg.scaling_factor)
    print("decided:", len(SCALING_FACTORS), "undecided:", len(undecided_indices()))

    try:
        current_samples = get_samples(e)
    except FileNotFoundError:
        print("Missing file")
        return
    except Exception as ex:
        print("Failed to sample:", ex)
        return

    s = suggestions(e)
    if s:
        print("suggestions:", {k: round(v, 6) for k, v in s.items()})
    stats = value_stats(current_samples.values)
    if stats:
        print("stats:", {k: round(v, 6) for k, v in stats.items()})

    if chosen_scale is None:
        chosen_scale = SCALING_FACTORS.get(e.idx)
    if chosen_scale is None:
        chosen_scale = s.get("p99", 1.0)

    decided_items = list(SCALING_FACTORS.items())
    decided_items = decided_items[-max_decided:]

    n_decided_rows = len(decided_items) if show_decided else 0
    n_strategy_rows = 2 * len(s)
    n_rows = n_decided_rows + n_strategy_rows + 2

    fig, axes = plt.subplots(n_rows, NUM_PLOT_SAMPLES, figsize=(NUM_PLOT_SAMPLES * 2.1, n_rows * 2.0))
    axes_arr = np.array(axes)

    row = 0

    if show_decided:
        for idx, factor in decided_items:
            entry = ENTRIES[idx]
            try:
                decided_samples = get_samples(entry)
            except Exception:
                continue
            imshow_row(
                list(axes_arr[row]),
                scaled(decided_samples.plot_images, factor),
                f"DECIDED\n{factor:.3g}\n{entry.key}",
            )
            row += 1

    for name, factor in s.items():
        imshow_row(list(axes_arr[row]), current_samples.plot_images, f"{name}\nraw")
        row += 1
        imshow_row(list(axes_arr[row]), scaled(current_samples.plot_images, factor), f"{name}\n{factor:.3g}")
        row += 1

    imshow_row(list(axes_arr[row]), current_samples.plot_images, "CHOSEN\nraw")
    row += 1
    imshow_row(list(axes_arr[row]), scaled(current_samples.plot_images, chosen_scale), f"CHOSEN\n{chosen_scale:.3g}")

    plt.tight_layout()
    plt.show()


def plot_histogram(*, bins: int = 200, chosen_scale: float | None = None) -> None:
    e = current_entry()
    try:
        samples = get_samples(e)
    except Exception as ex:
        print("Failed to load values:", ex)
        return

    values = samples.values
    if values.size == 0:
        print("No signal values")
        return

    s = suggestions(e)
    if chosen_scale is None:
        chosen_scale = SCALING_FACTORS.get(e.idx) or s.get("p99")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(values, bins=bins, log=True)

    for name, factor in s.items():
        ax.axvline(1.0 / factor, linewidth=1, label=name)

    if chosen_scale is not None:
        ax.axvline(1.0 / chosen_scale, linewidth=2, color="black", label="chosen")

    ax.set_title(e.key)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def export_scaling_factors() -> None:
    for e in ENTRIES:
        if e.idx in SCALING_FACTORS:
            print(f"# dataset_configs[{e.idx}] {e.key}\n# h5_path: {e.cfg.h5_path}\nscaling_factor: {SCALING_FACTORS[e.idx]:.6f}\n")


def print_help() -> None:
    print("Commands:")
    print("- plot_current(show_decided=True, max_decided=6)")
    print("- plot_histogram()")
    print("- accept_strategy('p99' | 'mean+3std' | 'median+3mad' | 'max')")
    print("- set_scaling_factor(x)")
    print("- next_dataset() / prev_dataset() / next_undecided() / goto(i)")
    print("- resample_current()")
    print("- export_scaling_factors()")


# %%
print_help()
plot_current()
