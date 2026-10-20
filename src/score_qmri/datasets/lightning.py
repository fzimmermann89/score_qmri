"""PyTorch Lightning DataModule for MRI datasets."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
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
    """Base configuration for MRI datasets."""

    h5_path: str
    augmentation_config: AugmentationConfig | None = None
    schedule: ScheduleType | None = None
    penumbra_width: int = 30
    subject_namespace: str | None = None

    def __post_init__(self) -> None:
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


@dataclass(kw_only=True)
class ValidationSampleConfig:
    """One fixed subject/slice/coil, loaded without random augmentation."""

    dataset: ComplexDatasetConfig | MagnitudeDatasetConfig | MappingDatasetConfig
    subject: str
    slice_index: int
    coil_index: int | None = None


@dataclass
class MRIDataModuleHparams:
    """Type-safe hyperparameter definitions for MRIDataModule."""

    dataset_configs: list[DatasetConfig] | None
    batch_size: int
    sample_size: tuple[int, int]
    num_workers: int
    base_seed: int
    kind_normalization: float


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
        validation_samples: list[ValidationSampleConfig] | None = None,
        held_out_subjects: dict[str, list[str]] | None = None,
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
        validation_samples
            Explicit H5/subject/slice/coil references, evaluated without augmentation.
        held_out_subjects
            Subjects excluded across all training configs in the same namespace.
            Fixed validation subjects are added automatically.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_configs: list[DatasetConfig] = dataset_configs or []
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.kind_normalization = kind_normalization
        self.validation_samples = validation_samples or []
        self.held_out_subjects = {name: set(subjects) for name, subjects in (held_out_subjects or {}).items()}
        for sample in self.validation_samples:
            if not sample.dataset.subject_namespace:
                raise ValueError("Fixed validation samples require a subject_namespace shared across contrasts/coils")
            if sample.slice_index < 0 or (sample.coil_index is not None and sample.coil_index < 0):
                raise ValueError("Validation slice and coil indices must be nonnegative")
            self.held_out_subjects.setdefault(sample.dataset.subject_namespace, set()).add(sample.subject)
        validation_paths = {Path(sample.dataset.h5_path).resolve(): sample.dataset.subject_namespace for sample in self.validation_samples}
        if any(
            Path(cfg.h5_path).resolve() in validation_paths and cfg.subject_namespace != validation_paths[Path(cfg.h5_path).resolve()]
            for cfg in self.dataset_configs
        ):
            raise ValueError("Training configs sharing validation files must use the same subject_namespace")
        self.validation_batch: dict[str, torch.Tensor] | None = None

        self.concat_ds: ConcatDataset | None = None
        self.dataset_lengths: list[int] | None = None

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
            summary[f"{prefix}/subject_namespace"] = cfg.subject_namespace
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

        summary["data/held_out_subjects"] = {name: sorted(subjects) for name, subjects in self.held_out_subjects.items()}
        summary["data/validation_samples"] = [
            {
                "h5_path": sample.dataset.h5_path,
                "kind": sample.dataset.kind.name,
                "subject": sample.subject,
                "slice_index": sample.slice_index,
                "coil_index": sample.coil_index,
            }
            for sample in self.validation_samples
        ]
        return summary

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate", None) and self.validation_samples:
            samples = []
            for config in self.validation_samples:
                dataset = replace(config.dataset, augmentation_config=None).create()
                coilwise = isinstance(config.dataset, ComplexDatasetConfig) and config.dataset.mode == "coilwise"
                if coilwise != (config.coil_index is not None):
                    raise ValueError("Only coilwise validation requires an explicit coil_index")
                try:
                    if isinstance(dataset, ComplexImageDataset):
                        index = dataset._slice_index.index((config.subject, config.slice_index, config.coil_index or 0))
                    else:
                        index = dataset._slice_index.index((config.subject, config.slice_index))
                except ValueError as error:
                    raise ValueError(
                        f"Missing validation sample {config.subject}/{config.slice_index} in {config.dataset.h5_path}"
                    ) from error
                samples.append(dataset[index])
            collated = Collate(*self.sample_size)(samples)
            if not torch.isfinite(collated["image"]).all() or ((~collated["background_mask"]).sum(dim=(-2, -1)) < 1000).any():
                raise ValueError("Fixed validation images must be finite with at least 1000 signal pixels after cropping")
            self.validation_batch = {key: value for key, value in collated.items() if isinstance(value, torch.Tensor)}
        if stage == "fit" or stage is None:
            if not self.dataset_configs:
                raise ValueError("No datasets configured.")
            datasets: list[MRIDataset | Subset] = []
            for training_config in self.dataset_configs:
                training_dataset = training_config.create()
                excluded = self.held_out_subjects.get(training_config.subject_namespace or "", set())
                if excluded:
                    if not hasattr(training_dataset, "_slice_index"):
                        raise ValueError("Subject exclusions require a dataset with explicit subject/slice indexing")
                    indices = [i for i, entry in enumerate(training_dataset._slice_index) if entry[0] not in excluded]
                    if not indices:
                        raise ValueError(f"Subject exclusions leave no training data in {training_config.h5_path}")
                    datasets.append(Subset(training_dataset, indices))
                else:
                    datasets.append(training_dataset)
            self.concat_ds = ConcatDataset(datasets)
            self.dataset_lengths = [len(ds) for ds in datasets]

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
