"""PyTorch Lightning DataModule for MRI datasets."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from .batch_sampler import ScheduledBatchSampler
from .complex import ComplexImageDataset
from .concat import get_size_labels
from .config import AugmentationConfig
from .kind import DatasetKind
from .magnitude import MagnitudeImageDataset
from .mapping import MappingDataset

MRIDataset = ComplexImageDataset | MagnitudeImageDataset | MappingDataset


@dataclass(kw_only=True)
class ConstantSchedule:
    """Constant sampling weight throughout training."""

    type: Literal["constant"] = "constant"
    weight: float = 1.0

    def __call__(self, step: int) -> float:
        """Return constant weight."""
        return self.weight


@dataclass(kw_only=True)
class PhaseInSchedule:
    """Phase in sampling weight linearly over N steps."""

    type: Literal["phase_in"] = "phase_in"
    steps: int
    weight: float = 1.0

    def __call__(self, step: int) -> float:
        """Return weight ramped from 0 to target over `steps`."""
        return self.weight * min(1.0, step / self.steps) if self.steps > 0 else self.weight


@dataclass(kw_only=True)
class PhaseOutSchedule:
    """Phase out sampling weight linearly over N steps."""

    type: Literal["phase_out"] = "phase_out"
    steps: int
    weight: float = 1.0

    def __call__(self, step: int) -> float:
        """Return weight ramped from target to 0 over `steps`."""
        return self.weight * max(0.0, 1.0 - step / self.steps) if self.steps > 0 else 0.0


ScheduleType = ConstantSchedule | PhaseInSchedule | PhaseOutSchedule


@dataclass(kw_only=True)
class BaseDatasetConfig(ABC):
    """Base configuration for MRI datasets."""

    h5_path: str
    augmentation_config: AugmentationConfig | None = None
    schedule: ScheduleType = field(default_factory=ConstantSchedule)

    @abstractmethod
    def create(self) -> MRIDataset:
        """Create dataset instance from this config."""
        ...


@dataclass(kw_only=True)
class ComplexDatasetConfig(BaseDatasetConfig):
    """Configuration for complex-valued MRI datasets."""

    type: Literal["complex"] = "complex"
    mode: Literal["combined", "coilwise"] = "combined"
    kind: DatasetKind = DatasetKind.COMPLEX_T1W_COILWISE

    def create(self) -> ComplexImageDataset:
        """Create ComplexImageDataset from this config."""
        return ComplexImageDataset(
            h5_path=self.h5_path,
            mode=self.mode,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
        )


@dataclass(kw_only=True)
class MagnitudeDatasetConfig(BaseDatasetConfig):
    """Configuration for magnitude MRI datasets."""

    type: Literal["magnitude"] = "magnitude"
    kind: DatasetKind = DatasetKind.T1W_MAGNITUDE

    def create(self) -> MagnitudeImageDataset:
        """Create MagnitudeImageDataset from this config."""
        return MagnitudeImageDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
        )


@dataclass(kw_only=True)
class MappingDatasetConfig(BaseDatasetConfig):
    """Configuration for quantitative mapping datasets."""

    type: Literal["mapping"] = "mapping"
    kind: DatasetKind = DatasetKind.T1MAP
    scaling_factor: float = 1.0

    def create(self) -> MappingDataset:
        """Create MappingDataset from this config."""
        return MappingDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            scaling_factor=self.scaling_factor,
            augmentation_config=self.augmentation_config,
        )


DatasetConfig = ComplexDatasetConfig | MagnitudeDatasetConfig | MappingDatasetConfig


@dataclass
class MRIDataModuleHparams:
    """Type-safe hyperparameter definitions for MRIDataModule."""

    dataset_configs: list[DatasetConfig] | None
    batch_size: int
    num_workers: int
    base_seed: int


class MRIDataModule(LightningDataModule):
    """DataModule for MRI training datasets with scheduled sampling."""

    hparams: MRIDataModuleHparams  # type: ignore[assignment]

    def __init__(
        self,
        dataset_configs: list[DatasetConfig] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        base_seed: int = 42,
    ) -> None:
        """
        Initialize MRI data module.

        Parameters
        ----------
        dataset_configs
            List of dataset configurations (dataclasses).
        batch_size
            Training batch size.
        num_workers
            DataLoader workers.
        base_seed
            Random seed for reproducibility.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_configs: list[DatasetConfig] = dataset_configs or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_seed = base_seed

        self.concat_ds: ConcatDataset | None = None
        self.size_labels: list[tuple[int, int]] | None = None
        self.dataset_lengths: list[int] | None = None

    def get_config_summary(self) -> dict[str, object]:
        """
        Get flattened configuration summary for logging to W&B.

        Returns
        -------
        Dictionary with flattened dataset and augmentation parameters.
        """
        summary: dict[str, object] = {
            "data/batch_size": self.batch_size,
            "data/num_workers": self.num_workers,
            "data/base_seed": self.base_seed,
            "data/num_datasets": len(self.dataset_configs),
        }

        # List of dataset kinds being trained on
        kinds = [cfg.kind.name for cfg in self.dataset_configs]
        summary["data/dataset_kinds"] = kinds

        # Per-dataset config
        for i, cfg in enumerate(self.dataset_configs):
            prefix = f"data/dataset_{i}"
            summary[f"{prefix}/kind"] = cfg.kind.name
            summary[f"{prefix}/h5_path"] = cfg.h5_path
            summary[f"{prefix}/type"] = cfg.type
            summary[f"{prefix}/schedule_type"] = cfg.schedule.type
            summary[f"{prefix}/schedule_weight"] = cfg.schedule.weight

            if isinstance(cfg, ComplexDatasetConfig):
                summary[f"{prefix}/mode"] = cfg.mode
            if isinstance(cfg, MappingDatasetConfig):
                summary[f"{prefix}/scaling_factor"] = cfg.scaling_factor

            # Augmentation config
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
        """Set up datasets for training."""
        if (stage == "fit" or stage is None) and self.dataset_configs:
            datasets = [cfg.create() for cfg in self.dataset_configs]
            self.concat_ds = ConcatDataset(datasets)
            self.size_labels = get_size_labels(datasets)
            self.dataset_lengths = [len(ds) for ds in datasets]

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with scheduled batch sampler."""
        if self.concat_ds is None or self.size_labels is None or self.dataset_lengths is None:
            raise RuntimeError("Must call setup() before train_dataloader()")

        rank = self.trainer.global_rank if self.trainer is not None else int(os.environ.get("LOCAL_RANK", 0))

        batch_sampler = ScheduledBatchSampler(
            self.concat_ds,
            self.size_labels,
            self.dataset_lengths,
            [cfg.schedule for cfg in self.dataset_configs],
            batch_size=self.batch_size,
            base_seed=self.base_seed,
            rank=rank,
        )

        return DataLoader(
            self.concat_ds,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
