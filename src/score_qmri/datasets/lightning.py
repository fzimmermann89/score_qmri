"""PyTorch Lightning DataModule for MRI datasets."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .batch_sampler import ScheduledBatchSampler
from .brainweb import VALUES_3T_RANDOMIZED, BrainwebQualitativeDataset, BrainwebQuantitativeDataset, BrainwebTissue
from .complex import ComplexImageDataset
from .config import (
    COMPLEX_CONFIG,
    MAGNITUDE_CONFIG,
    MAPPING_CONFIG,
    AugmentationConfig,
)
from .kind import DatasetKind
from .magnitude import MagnitudeImageDataset
from .mapping import MappingDataset
from .utils import _pad_or_crop_centered

MRIDataset = ComplexImageDataset | MagnitudeImageDataset | MappingDataset | BrainwebQuantitativeDataset | BrainwebQualitativeDataset


@dataclass(kw_only=True)
class ConstantSchedule:
    """Constant sampling weight throughout training."""

    type: Literal["constant"] = "constant"
    weight: float = 1.0

    def __call__(self, step: int) -> float:
        return self.weight


@dataclass(kw_only=True)
class PhaseInSchedule:
    """Phase in sampling weight linearly over N steps."""

    type: Literal["phase_in"] = "phase_in"
    steps: int
    weight: float = 1.0

    def __call__(self, step: int) -> float:
        return self.weight * min(1.0, step / self.steps) if self.steps > 0 else self.weight


@dataclass(kw_only=True)
class PhaseOutSchedule:
    """Phase out sampling weight linearly over N steps."""

    type: Literal["phase_out"] = "phase_out"
    steps: int
    weight: float = 1.0

    def __call__(self, step: int) -> float:
        return self.weight * max(0.0, 1.0 - step / self.steps) if self.steps > 0 else 0.0


ScheduleType = ConstantSchedule | PhaseInSchedule | PhaseOutSchedule


@dataclass(kw_only=True)
class BaseDatasetConfig(ABC):
    """MRI dataset settings; validation_samples are zero-based indices before augmentation."""

    h5_path: str
    augmentation_config: AugmentationConfig | None = None
    schedule: ScheduleType | None = None
    penumbra_width: int = 30
    validation_samples: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if any(index < 0 for index in self.validation_samples) or len(set(self.validation_samples)) != len(self.validation_samples):
            raise ValueError("validation_samples must contain unique nonnegative sample numbers")
        if self.schedule is None:
            self.schedule = ConstantSchedule()

    @abstractmethod
    def create(self) -> MRIDataset: ...


@dataclass(kw_only=True)
class ComplexDatasetConfig(BaseDatasetConfig):
    """Configuration for complex-valued MRI datasets."""

    type: Literal["complex"] = "complex"
    mode: Literal["flat", "combined", "coilwise"] = "flat"
    kind: DatasetKind = DatasetKind.COMPLEX_T1W_COILWISE
    augmentation_config: AugmentationConfig | None = COMPLEX_CONFIG
    scaling_factor: float = 1.0

    def create(self) -> ComplexImageDataset:
        return ComplexImageDataset(
            h5_path=self.h5_path,
            mode=self.mode,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
            scaling_factor=self.scaling_factor,
            penumbra_width=self.penumbra_width,
        )


@dataclass(kw_only=True)
class MagnitudeDatasetConfig(BaseDatasetConfig):
    """Configuration for magnitude MRI datasets."""

    type: Literal["magnitude"] = "magnitude"
    kind: DatasetKind = DatasetKind.T1W_MAGNITUDE
    augmentation_config: AugmentationConfig | None = MAGNITUDE_CONFIG
    scaling_factor: float = 1.0

    def create(self) -> MagnitudeImageDataset:
        return MagnitudeImageDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
            scaling_factor=self.scaling_factor,
            penumbra_width=self.penumbra_width,
        )


@dataclass(kw_only=True)
class MappingDatasetConfig(BaseDatasetConfig):
    """Configuration for quantitative mapping datasets."""

    type: Literal["mapping"] = "mapping"
    kind: DatasetKind = DatasetKind.T1MAP
    augmentation_config: AugmentationConfig | None = MAPPING_CONFIG
    scaling_factor: float = 1.0
    penumbra_width: int = -1

    def create(self) -> MappingDataset:
        return MappingDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            scaling_factor=self.scaling_factor,
            augmentation_config=self.augmentation_config,
            penumbra_width=self.penumbra_width,
        )


@dataclass(kw_only=True)
class BrainwebDatasetConfig(BaseDatasetConfig):
    """Configuration for Brainweb quantitative datasets."""

    type: Literal["brainweb"] = "brainweb"
    kind: DatasetKind = DatasetKind.T1MAP
    augmentation_config: AugmentationConfig | None = MAPPING_CONFIG
    scaling_factor: float = 1.0
    penumbra_width: int = -1
    step: int = 1
    skip_slices: tuple[int, int] = (80, 80)
    parameters: dict[str, BrainwebTissue] | None = None
    subject_id: int | None = None

    def create(self) -> BrainwebQuantitativeDataset:
        params = self.parameters if self.parameters is not None else VALUES_3T_RANDOMIZED
        return BrainwebQuantitativeDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            parameters=params,
            augmentation_config=self.augmentation_config,
            scaling_factor=self.scaling_factor,
            step=self.step,
            skip_slices=self.skip_slices,
            penumbra_width=self.penumbra_width,
            subject_id=self.subject_id,
        )


@dataclass(kw_only=True)
class BrainwebMagnitudeQualitativeDatasetConfig(BaseDatasetConfig):
    """Configuration for Brainweb magnitude qualitative datasets."""

    type: Literal["brainweb_qualitative_magnitude"] = "brainweb_qualitative_magnitude"
    kind: DatasetKind = DatasetKind.T1W_MAGNITUDE
    augmentation_config: AugmentationConfig | None = MAGNITUDE_CONFIG
    scaling_factor: float = 1.0
    penumbra_width: int = -1
    step: int = 1
    skip_slices: tuple[int, int] = (80, 80)
    parameters: dict[str, BrainwebTissue] | None = None
    subject_id: int | None = None
    tr_range: tuple[float, float] | None = None
    te_range: tuple[float, float] | None = None
    flip_range: tuple[float, float] | None = None
    background_phase_std: float = 0.3

    _VALID_KINDS: ClassVar[frozenset[DatasetKind]] = frozenset(
        {
            DatasetKind.T1W_MAGNITUDE,
            DatasetKind.T2W_MAGNITUDE,
        }
    )

    def create(self: "BrainwebMagnitudeQualitativeDatasetConfig") -> BrainwebQualitativeDataset:
        params = self.parameters if self.parameters is not None else VALUES_3T_RANDOMIZED
        if self.kind not in self._VALID_KINDS:
            raise ValueError(f"kind must be one of {self._VALID_KINDS}, got {self.kind}")
        augmentation_config = self.augmentation_config
        return BrainwebQualitativeDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            parameters=params,
            augmentation_config=augmentation_config,
            scaling_factor=self.scaling_factor,
            step=self.step,
            skip_slices=self.skip_slices,
            penumbra_width=self.penumbra_width,
            subject_id=self.subject_id,
            tr_range=self.tr_range,
            te_range=self.te_range,
            flip_range=self.flip_range,
            background_phase_std=self.background_phase_std,
        )


@dataclass(kw_only=True)
class BrainwebComplexQualitativeDatasetConfig(BaseDatasetConfig):
    """Configuration for Brainweb complex qualitative datasets."""

    type: Literal["brainweb_qualitative_complex"] = "brainweb_qualitative_complex"
    kind: DatasetKind = DatasetKind.COMPLEX_T1W_COIL_COMBINED
    augmentation_config: AugmentationConfig | None = COMPLEX_CONFIG
    scaling_factor: float = 1.0
    penumbra_width: int = -1
    step: int = 1
    skip_slices: tuple[int, int] = (80, 80)
    parameters: dict[str, BrainwebTissue] | None = None
    subject_id: int | None = None
    tr_range: tuple[float, float] | None = None
    te_range: tuple[float, float] | None = None
    flip_range: tuple[float, float] | None = None
    background_phase_std: float = 0.3

    _VALID_KINDS: ClassVar[frozenset[DatasetKind]] = frozenset(
        {
            DatasetKind.COMPLEX_T1W_COIL_COMBINED,
            DatasetKind.COMPLEX_T2W_COIL_COMBINED,
        }
    )

    def create(self: "BrainwebComplexQualitativeDatasetConfig") -> BrainwebQualitativeDataset:
        params = self.parameters if self.parameters is not None else VALUES_3T_RANDOMIZED
        if self.kind not in self._VALID_KINDS:
            raise ValueError(f"kind must be one of {self._VALID_KINDS}, got {self.kind}")
        augmentation_config = self.augmentation_config
        return BrainwebQualitativeDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            parameters=params,
            augmentation_config=augmentation_config,
            scaling_factor=self.scaling_factor,
            step=self.step,
            skip_slices=self.skip_slices,
            penumbra_width=self.penumbra_width,
            subject_id=self.subject_id,
            tr_range=self.tr_range,
            te_range=self.te_range,
            flip_range=self.flip_range,
            background_phase_std=self.background_phase_std,
        )


DatasetConfig = (
    ComplexDatasetConfig
    | MagnitudeDatasetConfig
    | MappingDatasetConfig
    | BrainwebDatasetConfig
    | BrainwebMagnitudeQualitativeDatasetConfig
    | BrainwebComplexQualitativeDatasetConfig
)


@dataclass
class MRIDataModuleHparams:
    """Type-safe hyperparameter definitions for MRIDataModule."""

    dataset_configs: list[DatasetConfig] | None
    batch_size: int
    sample_size: tuple[int, int]
    num_workers: int
    base_seed: int
    kind_normalization: float


def _validation_key(dataset: MRIDataset, index: int) -> tuple[str, str, int]:
    """Identify a source slice across combined and coilwise entries of the same H5."""
    entries = getattr(dataset, "_slice_index", None)
    return (dataset.h5_path, str(entries[index][0]), int(entries[index][1])) if entries is not None else (dataset.h5_path, "", index)


def _to_2ch(image: torch.Tensor) -> torch.Tensor:
    """Convert native (H, W) image to 2-channel (2, H, W) float tensor."""
    if image.is_complex():
        return torch.moveaxis(torch.view_as_real(image), -1, 0)
    return torch.stack([image, torch.zeros_like(image)], dim=0)


class Collate:
    """Collate batch samples into tensors with center crop/pad and 2-channel conversion.

    Converts complex tensors to (real, imag) and real tensors to (value, zeros).
    Output image shape is (B, 2, H, W) float32.
    """

    def __init__(self, target_h: int, target_w: int) -> None:
        self.target_h = target_h
        self.target_w = target_w

    def _pad_mask(self, mask: torch.Tensor, replicate: bool = False) -> torch.Tensor:
        return _pad_or_crop_centered(mask.float(), self.target_h, self.target_w, replicate=replicate).bool()

    def _pad_background_mask(self, mask: torch.Tensor) -> torch.Tensor:
        signal_mask = (~mask).float()
        padded_signal = _pad_or_crop_centered(signal_mask, self.target_h, self.target_w)
        return ~padded_signal.bool()

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images_2ch = []
        for s in batch:
            img = _pad_or_crop_centered(s["image"], self.target_h, self.target_w)
            images_2ch.append(_to_2ch(img))

        return {
            "image": torch.stack(images_2ch),
            "penumbra_mask": torch.stack([self._pad_mask(s["penumbra_mask"], replicate=True) for s in batch]),
            "background_mask": torch.stack([self._pad_background_mask(s["background_mask"]) for s in batch]),
            "kind": torch.stack([s["kind"] for s in batch]),
            "h5_path": [s["h5_path"] for s in batch],
        }


class MRIDataModule(LightningDataModule):
    """DataModule for MRI training datasets with scheduled sampling."""

    def __init__(
        self,
        dataset_configs: list[DatasetConfig] | None = None,
        batch_size: int = 32,
        sample_size: tuple[int, int] = (256, 256),
        num_workers: int = 8,
        base_seed: int = 42,
        kind_normalization: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        dataset_configs
            List of dataset configurations.
        batch_size
            Training batch size.
        sample_size
            Target (H, W) for center crop/pad. Default 256x256.
        num_workers
            DataLoader workers.
        base_seed
            Random seed for reproducibility.
        kind_normalization
            Interpolation between size-proportional (0.0) and kind-uniform (1.0) sampling.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_configs: list[DatasetConfig] = dataset_configs or []
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.kind_normalization = kind_normalization

        self.concat_ds: ConcatDataset | None = None
        self.dataset_lengths: list[int] | None = None
        self.validation_batch: dict[str, torch.Tensor] | None = None

    def get_config_summary(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "data/batch_size": self.batch_size,
            "data/sample_size": list(self.sample_size),
            "data/num_workers": self.num_workers,
            "data/base_seed": self.base_seed,
            "data/kind_normalization": self.kind_normalization,
            "data/num_datasets": len(self.dataset_configs),
        }

        kinds = [cfg.kind.name for cfg in self.dataset_configs]
        summary["data/dataset_kinds"] = kinds

        for i, cfg in enumerate(self.dataset_configs):
            prefix = f"data/dataset_{i}"
            summary[f"{prefix}/kind"] = cfg.kind.name
            summary[f"{prefix}/h5_path"] = cfg.h5_path
            summary[f"{prefix}/type"] = cfg.type
            summary[f"{prefix}/validation_samples"] = cfg.validation_samples
            schedule: ScheduleType = cfg.schedule if cfg.schedule is not None else ConstantSchedule()
            summary[f"{prefix}/schedule_type"] = schedule.type
            summary[f"{prefix}/schedule_weight"] = schedule.weight

            if isinstance(cfg, ComplexDatasetConfig):
                summary[f"{prefix}/mode"] = cfg.mode
            if hasattr(cfg, "scaling_factor") and cfg.scaling_factor != 1.0:
                summary[f"{prefix}/scaling_factor"] = cfg.scaling_factor

            if cfg.augmentation_config is not None:
                aug = cfg.augmentation_config
                aug_prefix = f"{prefix}/aug"
                summary[f"{aug_prefix}/geometric_prob"] = aug.geometric_prob
                summary[f"{aug_prefix}/contrast_prob"] = aug.contrast_prob
                summary[f"{aug_prefix}/phase_prob"] = aug.phase_prob
                summary[f"{aug_prefix}/flip_prob"] = aug.flip_prob
                summary[f"{aug_prefix}/roll_prob"] = aug.roll_prob
                summary[f"{aug_prefix}/shift_range"] = list(aug.shift_range)
                summary[f"{aug_prefix}/scale_range"] = list(aug.scale_range)
                summary[f"{aug_prefix}/rotation_range"] = list(aug.rotation_range)
                summary[f"{aug_prefix}/shear_range"] = list(aug.shear_range)
                summary[f"{aug_prefix}/contrast_strength"] = aug.contrast_strength
                summary[f"{aug_prefix}/poly_coeff_range"] = list(aug.poly_coeff_range)

        return summary

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            if not self.dataset_configs:
                raise ValueError("No datasets configured.")
            datasets = [cfg.create() for cfg in self.dataset_configs]
            self.validation_batch = None
            reserved: dict[str, set[tuple[str, str, int]]] = {}
            validation = []
            for config, dataset in zip(self.dataset_configs, datasets, strict=True):
                if not config.validation_samples:
                    continue
                if max(config.validation_samples) >= len(dataset):
                    raise ValueError(f"validation_samples outside dataset of length {len(dataset)}: {config.h5_path}")
                fixed = replace(config, augmentation_config=None).create()
                for index in config.validation_samples:
                    validation.append(fixed[index])
                    reserved.setdefault(dataset.h5_path, set()).add(_validation_key(dataset, index))
            if validation:
                batch = Collate(*self.sample_size)(validation)
                self.validation_batch = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
            training: list[MRIDataset | Subset] = []
            for dataset in datasets:
                if dataset.h5_path in reserved:
                    # Withhold the selected source slice from every coil/mode
                    # entry, preventing its combined image from leaking back in.
                    indices = [i for i in range(len(dataset)) if _validation_key(dataset, i) not in reserved[dataset.h5_path]]
                    if not indices:
                        raise ValueError(f"validation_samples leave no training samples in {dataset.h5_path}")
                    training.append(Subset(dataset, indices))
                else:
                    training.append(dataset)
            self.concat_ds = ConcatDataset(training)
            self.dataset_lengths = [len(ds) for ds in training]

    def train_dataloader(self) -> DataLoader:
        if self.concat_ds is None or self.dataset_lengths is None:
            raise RuntimeError("Datasets not initialized.")

        rank = self.trainer.global_rank if self.trainer is not None else int(os.environ.get("LOCAL_RANK", 0))

        max_steps = None
        if self.trainer is not None and self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            max_steps = self.trainer.max_steps

        schedules: list[ScheduleType] = [cfg.schedule if cfg.schedule is not None else ConstantSchedule() for cfg in self.dataset_configs]

        batch_sampler = ScheduledBatchSampler(
            self.concat_ds,
            self.dataset_lengths,
            dataset_kinds=[cfg.kind for cfg in self.dataset_configs],
            schedules=schedules,
            batch_size=self.batch_size,
            kind_normalization=self.kind_normalization,
            base_seed=self.base_seed,
            rank=rank,
            max_steps=max_steps,
        )

        return DataLoader(
            self.concat_ds,
            batch_sampler=batch_sampler,
            collate_fn=Collate(*self.sample_size),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=8 if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            timeout=120 if self.num_workers > 0 else 0,
        )
