"""Config loader for dataset debug viewer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from score_qmri.datasets.complex import ComplexImageDataset
from score_qmri.datasets.config import AugmentationConfig
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import ConstantSchedule, PhaseInSchedule, PhaseOutSchedule, ScheduleType
from score_qmri.datasets.magnitude import MagnitudeImageDataset
from score_qmri.datasets.mapping import MappingDataset

DatasetMode = Literal["flat", "combined", "coilwise"]
DatasetType = Literal["complex", "magnitude", "mapping"]


@dataclass(frozen=True)
class DebugDatasetConfig:
    """Parsed dataset configuration for debug viewer."""

    type: DatasetType
    h5_path: str
    kind: DatasetKind
    augmentation_config: AugmentationConfig | None
    schedule: ScheduleType | None
    penumbra_width: int
    scaling_factor: float
    mode: DatasetMode | None = None

    def create(self) -> ComplexImageDataset | MagnitudeImageDataset | MappingDataset:
        if self.type == "complex":
            return ComplexImageDataset(
                h5_path=self.h5_path,
                mode=self.mode or "flat",
                kind=self.kind,
                augmentation_config=self.augmentation_config,
                scaling_factor=self.scaling_factor,
                penumbra_width=self.penumbra_width,
            )
        if self.type == "magnitude":
            return MagnitudeImageDataset(
                h5_path=self.h5_path,
                kind=self.kind,
                augmentation_config=self.augmentation_config,
                scaling_factor=self.scaling_factor,
                penumbra_width=self.penumbra_width,
            )
        return MappingDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
            scaling_factor=self.scaling_factor,
            penumbra_width=self.penumbra_width,
        )


@dataclass(frozen=True)
class LoadedConfig:
    """Parsed configuration for dataset debug viewer."""

    config_path: Path
    dataset_configs: list[DebugDatasetConfig]
    raw_dataset_configs: list[dict[str, Any]]
    data_settings: dict[str, Any]


def load_config(config_path: str | Path) -> LoadedConfig:
    """Load dataset configuration from YAML.

    Parameters
    ----------
    config_path
        Path to YAML config.
    """
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_section = config.get("data", {})
    init_args = data_section.get("init_args", data_section)
    dataset_configs_raw = list(init_args.get("dataset_configs", []))

    dataset_configs = []
    for entry in dataset_configs_raw:
        h5_path = str(entry.get("h5_path", ""))
        resolved_path = Path(h5_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {h5_path} (resolved to {resolved_path})")
        dataset_configs.append(parse_dataset_config(entry))

    if len(dataset_configs) == 0:
        raise ValueError("No valid datasets found (all files missing)")

    data_settings = {
        "batch_size": init_args.get("batch_size"),
        "sample_size": init_args.get("sample_size"),
        "num_workers": init_args.get("num_workers"),
        "base_seed": init_args.get("base_seed"),
        "kind_normalization": init_args.get("kind_normalization"),
        "num_datasets": len(dataset_configs),
    }

    return LoadedConfig(
        config_path=path,
        dataset_configs=dataset_configs,
        raw_dataset_configs=dataset_configs_raw,
        data_settings=data_settings,
    )


def clone_without_augmentation(config: DebugDatasetConfig) -> DebugDatasetConfig:
    """Return a copy of dataset config with augmentation disabled."""
    return replace(config, augmentation_config=None)


def parse_dataset_config(raw: dict[str, Any]) -> DebugDatasetConfig:
    """Parse a dataset config entry from YAML.

    Parameters
    ----------
    raw
        Raw config entry from YAML.
    """
    config_type = raw.get("type")
    if config_type not in {"complex", "magnitude", "mapping"}:
        raise ValueError(f"Unsupported dataset type: {config_type}")

    h5_path = str(raw["h5_path"])
    kind = parse_kind(raw.get("kind"))
    augmentation = parse_augmentation(raw.get("augmentation_config"))
    schedule = parse_schedule(raw.get("schedule"))

    penumbra_width = int(raw.get("penumbra_width", 30))
    scaling_factor = float(raw.get("scaling_factor", 1.0))

    mode: DatasetMode | None = None
    if config_type == "complex":
        mode_raw = str(raw.get("mode", "flat"))
        if mode_raw not in {"flat", "combined", "coilwise"}:
            raise ValueError(f"Unsupported complex dataset mode: {mode_raw}")
        mode = cast(DatasetMode, mode_raw)

    if config_type == "mapping":
        penumbra_width = int(raw.get("penumbra_width", -1))

    return DebugDatasetConfig(
        type=config_type,
        h5_path=h5_path,
        kind=kind,
        augmentation_config=augmentation,
        schedule=schedule,
        penumbra_width=penumbra_width,
        scaling_factor=scaling_factor,
        mode=mode,
    )


def parse_kind(value: Any) -> DatasetKind:
    """Parse dataset kind from string or int."""
    if isinstance(value, DatasetKind):
        return value
    if isinstance(value, int):
        return DatasetKind(value)
    if isinstance(value, str):
        return DatasetKind[value]
    raise ValueError(f"Invalid dataset kind: {value}")


def parse_schedule(raw: dict[str, Any] | None) -> ScheduleType | None:
    """Parse schedule from dict configuration."""
    if raw is None:
        return None
    schedule_type = raw.get("type", "constant")
    match schedule_type:
        case "constant":
            return ConstantSchedule(weight=float(raw.get("weight", 1.0)))
        case "phase_in":
            return PhaseInSchedule(steps=int(raw.get("steps", 0)), weight=float(raw.get("weight", 1.0)))
        case "phase_out":
            return PhaseOutSchedule(steps=int(raw.get("steps", 0)), weight=float(raw.get("weight", 1.0)))
    raise ValueError(f"Unsupported schedule type: {schedule_type}")


def parse_augmentation(raw: dict[str, Any] | None) -> AugmentationConfig | None:
    """Parse augmentation configuration from dict, if provided."""
    if raw is None:
        return None

    defaults = AugmentationConfig()
    return AugmentationConfig(
        geometric_prob=_to_float(raw.get("geometric_prob", defaults.geometric_prob), defaults.geometric_prob),
        contrast_prob=_to_float(raw.get("contrast_prob", defaults.contrast_prob), defaults.contrast_prob),
        phase_prob=_to_float(raw.get("phase_prob", defaults.phase_prob), defaults.phase_prob),
        flip_prob=_to_float(raw.get("flip_prob", defaults.flip_prob), defaults.flip_prob),
        roll_prob=_to_float(raw.get("roll_prob", defaults.roll_prob), defaults.roll_prob),
        shift_range=_to_int_tuple(raw.get("shift_range", defaults.shift_range)),
        scale_range=_to_float_tuple(raw.get("scale_range", defaults.scale_range)),
        rotation_range=_to_float_tuple(raw.get("rotation_range", defaults.rotation_range)),
        shear_range=_to_float_tuple(raw.get("shear_range", defaults.shear_range)),
        contrast_strength=_to_float(raw.get("contrast_strength", defaults.contrast_strength), defaults.contrast_strength),
        poly_coeff_range=_to_float_tuple(raw.get("poly_coeff_range", defaults.poly_coeff_range)),
    )


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int_tuple(value: Any) -> tuple[int, int]:
    items = _to_tuple(value)
    return (int(items[0]), int(items[1]))


def _to_float_tuple(value: Any) -> tuple[float, float]:
    items = _to_tuple(value)
    return (float(items[0]), float(items[1]))


def _to_tuple(value: Any) -> tuple[Any, Any]:
    if isinstance(value, tuple):
        if len(value) == 2:
            return value
        return (value[0], value[0])
    if isinstance(value, list):
        if len(value) == 2:
            return (value[0], value[1])
        return (value[0], value[0])
    return (value, value)
