"""Config loading for viewer using jsonargparse (same as training)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from score_qmri.datasets.config import COMPLEX_CONFIG, MAGNITUDE_CONFIG, MAPPING_CONFIG, AugmentationConfig
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import (
    BrainwebComplexQualitativeDatasetConfig,
    BrainwebDatasetConfig,
    BrainwebMagnitudeQualitativeDatasetConfig,
    ComplexDatasetConfig,
    ConstantSchedule,
    DatasetConfig,
    MagnitudeDatasetConfig,
    MappingDatasetConfig,
    PhaseInSchedule,
    PhaseOutSchedule,
)

if TYPE_CHECKING:
    from score_qmri.datasets.lightning import ScheduleType

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load config.yaml and parse dataset configs.

    Uses the same dataclass types as training to ensure parity.
    Filters out missing H5 files.

    Parameters
    ----------
    config_path
        Path to config.yaml file.

    Returns
    -------
    dict
        Contains:
        - dataset_configs: list of DatasetConfig objects
        - raw_configs: list of raw dict entries
        - data_settings: batch_size, sample_size, kind_normalization, etc.
        - skipped_paths: list of paths that were skipped

    Raises
    ------
    FileNotFoundError
        If no valid datasets found.
    """
    path = Path(config_path).resolve()
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    data_section = raw.get("data", {})
    init_args = data_section.get("init_args", data_section)

    dataset_configs: list[DatasetConfig] = []
    raw_configs: list[dict] = []
    skipped_paths: list[str] = []

    for entry in init_args.get("dataset_configs", []):
        h5_path = Path(entry.get("h5_path", "")).resolve()
        if not h5_path.exists():
            skipped_paths.append(str(h5_path))
            continue

        try:
            cfg = _parse_dataset_config(entry)
            dataset_configs.append(cfg)
            raw_configs.append(entry)
        except Exception as e:
            logger.warning(f"Failed to parse dataset config: {e}")
            continue

    if not dataset_configs:
        raise FileNotFoundError(
            f"No valid datasets found in {config_path}. Missing files: {skipped_paths[:5]}{'...' if len(skipped_paths) > 5 else ''}"
        )

    logger.info(f"Loaded {len(dataset_configs)} datasets from {config_path}")
    if skipped_paths:
        logger.warning(f"Skipped {len(skipped_paths)} missing files")

    return {
        "dataset_configs": dataset_configs,
        "raw_configs": raw_configs,
        "data_settings": {
            "batch_size": init_args.get("batch_size", 64),
            "sample_size": tuple(init_args.get("sample_size", [256, 256])),
            "kind_normalization": init_args.get("kind_normalization", 0.5),
            "base_seed": init_args.get("base_seed", 42),
            "num_workers": init_args.get("num_workers", 4),
        },
        "skipped_paths": skipped_paths,
        "config_path": str(path),
    }


def _parse_dataset_config(entry: dict) -> DatasetConfig:
    """Parse a single dataset config entry into typed dataclass.

    Uses the SAME types as MRIDataModule to ensure parity.
    """
    config_type = entry.get("type")
    if config_type not in {
        "complex",
        "magnitude",
        "mapping",
        "brainweb",
        "brainweb_qualitative_magnitude",
        "brainweb_qualitative_complex",
    }:
        raise ValueError(f"Unsupported dataset type: {config_type}")

    # Parse common fields
    aug_raw = entry.get("augmentation_config")
    aug_explicit_none = "augmentation_config" in entry and aug_raw is None
    schedule = _parse_schedule(entry.get("schedule"))

    kind_str = entry.get("kind", "T1W_MAGNITUDE")
    kind = DatasetKind[kind_str] if isinstance(kind_str, str) else DatasetKind(kind_str)

    if config_type == "complex":
        return ComplexDatasetConfig(
            h5_path=str(entry["h5_path"]),
            kind=kind,
            mode=entry.get("mode", "flat"),
            scaling_factor=entry.get("scaling_factor", 1.0),
            schedule=schedule,
            penumbra_width=entry.get("penumbra_width", 30),
            augmentation_config=None if aug_explicit_none else _parse_augmentation(aug_raw, COMPLEX_CONFIG) if aug_raw else COMPLEX_CONFIG,
        )
    elif config_type == "magnitude":
        return MagnitudeDatasetConfig(
            h5_path=str(entry["h5_path"]),
            kind=kind,
            scaling_factor=entry.get("scaling_factor", 1.0),
            schedule=schedule,
            penumbra_width=entry.get("penumbra_width", 30),
            augmentation_config=None
            if aug_explicit_none
            else _parse_augmentation(aug_raw, MAGNITUDE_CONFIG)
            if aug_raw
            else MAGNITUDE_CONFIG,
        )
    elif config_type == "mapping":
        return MappingDatasetConfig(
            h5_path=str(entry["h5_path"]),
            kind=kind,
            scaling_factor=entry.get("scaling_factor", 1.0),
            schedule=schedule,
            penumbra_width=entry.get("penumbra_width", -1),
            augmentation_config=None if aug_explicit_none else _parse_augmentation(aug_raw, MAPPING_CONFIG) if aug_raw else MAPPING_CONFIG,
        )
    elif config_type == "brainweb":
        skip_slices_raw = entry.get("skip_slices", [80, 80])
        skip_slices = (int(skip_slices_raw[0]), int(skip_slices_raw[1]))
        return BrainwebDatasetConfig(
            h5_path=str(entry["h5_path"]),
            kind=kind,
            scaling_factor=entry.get("scaling_factor", 1.0),
            schedule=schedule,
            penumbra_width=entry.get("penumbra_width", -1),
            step=entry.get("step", 1),
            skip_slices=skip_slices,
            subject_id=entry.get("subject_id"),
            augmentation_config=None if aug_explicit_none else _parse_augmentation(aug_raw, MAPPING_CONFIG) if aug_raw else MAPPING_CONFIG,
        )
    elif config_type == "brainweb_qualitative_magnitude":
        skip_slices_raw = entry.get("skip_slices", [80, 80])
        skip_slices = (int(skip_slices_raw[0]), int(skip_slices_raw[1]))
        return BrainwebMagnitudeQualitativeDatasetConfig(
            h5_path=str(entry["h5_path"]),
            kind=kind,
            scaling_factor=entry.get("scaling_factor", 1.0),
            schedule=schedule,
            penumbra_width=entry.get("penumbra_width", -1),
            step=entry.get("step", 1),
            skip_slices=skip_slices,
            subject_id=entry.get("subject_id"),
            background_phase_std=entry.get("background_phase_std", 0.3),
            augmentation_config=None
            if aug_explicit_none
            else _parse_augmentation(aug_raw, MAGNITUDE_CONFIG)
            if aug_raw
            else MAGNITUDE_CONFIG,
        )
    else:  # brainweb_qualitative_complex
        skip_slices_raw = entry.get("skip_slices", [80, 80])
        skip_slices = (int(skip_slices_raw[0]), int(skip_slices_raw[1]))
        return BrainwebComplexQualitativeDatasetConfig(
            h5_path=str(entry["h5_path"]),
            kind=kind,
            scaling_factor=entry.get("scaling_factor", 1.0),
            schedule=schedule,
            penumbra_width=entry.get("penumbra_width", -1),
            step=entry.get("step", 1),
            skip_slices=skip_slices,
            subject_id=entry.get("subject_id"),
            background_phase_std=entry.get("background_phase_std", 0.3),
            augmentation_config=None if aug_explicit_none else _parse_augmentation(aug_raw, COMPLEX_CONFIG) if aug_raw else COMPLEX_CONFIG,
        )


def _parse_augmentation(raw: dict, defaults: AugmentationConfig) -> AugmentationConfig:
    """Parse augmentation config dict into AugmentationConfig dataclass."""

    def _tuple_float(v: list | tuple | None, default: tuple[float, float]) -> tuple[float, float]:
        if v is None:
            return default
        return (float(v[0]), float(v[1]))

    def _tuple_int(v: list | tuple | None, default: tuple[int, int]) -> tuple[int, int]:
        if v is None:
            return default
        return (int(v[0]), int(v[1]))

    return AugmentationConfig(
        geometric_prob=raw.get("geometric_prob", defaults.geometric_prob),
        contrast_prob=raw.get("contrast_prob", defaults.contrast_prob),
        phase_prob=raw.get("phase_prob", defaults.phase_prob),
        flip_prob=raw.get("flip_prob", defaults.flip_prob),
        roll_prob=raw.get("roll_prob", defaults.roll_prob),
        shift_range=_tuple_int(raw.get("shift_range"), defaults.shift_range),
        scale_range=_tuple_float(raw.get("scale_range"), defaults.scale_range),
        rotation_range=_tuple_float(raw.get("rotation_range"), defaults.rotation_range),
        shear_range=_tuple_float(raw.get("shear_range"), defaults.shear_range),
        contrast_strength=raw.get("contrast_strength", defaults.contrast_strength),
        poly_coeff_range=_tuple_float(raw.get("poly_coeff_range"), defaults.poly_coeff_range),
        intensity_scale_prob=raw.get("intensity_scale_prob", defaults.intensity_scale_prob),
        intensity_scale_range=_tuple_float(raw.get("intensity_scale_range"), defaults.intensity_scale_range),
    )


def _parse_schedule(raw: dict | None) -> ScheduleType:
    """Parse schedule config into typed schedule object."""
    if raw is None:
        return ConstantSchedule()

    schedule_type = raw.get("type", "constant")
    weight = raw.get("weight", 1.0)

    if schedule_type == "constant":
        return ConstantSchedule(weight=weight)
    elif schedule_type == "phase_in":
        return PhaseInSchedule(steps=raw.get("steps", 0), weight=weight)
    elif schedule_type == "phase_out":
        return PhaseOutSchedule(steps=raw.get("steps", 0), weight=weight)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
