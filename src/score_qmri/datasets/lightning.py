"""PyTorch Lightning DataModule for MRI datasets."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from .batch_sampler import ScheduledBatchSampler
from .complex import ComplexImageDataset
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
    scaling_factor: float = 1.0

    def create(self) -> ComplexImageDataset:
        return ComplexImageDataset(
            h5_path=self.h5_path,
            mode=self.mode,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
            scaling_factor=self.scaling_factor,
        )


@dataclass(kw_only=True)
class MagnitudeDatasetConfig(BaseDatasetConfig):
    """Configuration for magnitude MRI datasets."""

    type: Literal["magnitude"] = "magnitude"
    kind: DatasetKind = DatasetKind.T1W_MAGNITUDE
    scaling_factor: float = 1.0

    def create(self) -> MagnitudeImageDataset:
        return MagnitudeImageDataset(
            h5_path=self.h5_path,
            kind=self.kind,
            augmentation_config=self.augmentation_config,
            scaling_factor=self.scaling_factor,
        )


@dataclass(kw_only=True)
class MappingDatasetConfig(BaseDatasetConfig):
    """Configuration for quantitative mapping datasets."""

    type: Literal["mapping"] = "mapping"
    kind: DatasetKind = DatasetKind.T1MAP
    scaling_factor: float = 1.0

    def create(self) -> MappingDataset:
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
    sample_size: tuple[int, int]
    num_workers: int
    base_seed: int
    kind_normalization: float


def _pad_or_crop_centered(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pad (positive) or crop (negative) to target size, centered."""
    h, w = tensor.shape[-2:]
    delta_h = target_h - h
    delta_w = target_w - w
    pad_top = delta_h // 2
    pad_bot = delta_h - pad_top
    pad_left = delta_w // 2
    pad_right = delta_w - pad_left
    return torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bot))


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

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images_2ch = []
        for s in batch:
            img = _pad_or_crop_centered(s["image"], self.target_h, self.target_w)
            images_2ch.append(_to_2ch(img))
        images = torch.stack(images_2ch)
        masks = torch.stack([_pad_or_crop_centered(s["mask"].float(), self.target_h, self.target_w).bool() for s in batch])
        kinds = torch.stack([s["kind"] for s in batch])
        h5_paths = [s["h5_path"] for s in batch]
        return {"image": images, "mask": masks, "kind": kinds, "h5_path": h5_paths}


class MRIDataModule(LightningDataModule):
    """DataModule for MRI training datasets with scheduled sampling."""

    hparams: MRIDataModuleHparams  # type: ignore[assignment]

    def __init__(
        self,
        dataset_configs: list[DatasetConfig] | None = None,
        batch_size: int = 8,
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
            summary[f"{prefix}/schedule_type"] = cfg.schedule.type  # type: ignore[union-attr]
            summary[f"{prefix}/schedule_weight"] = cfg.schedule.weight  # type: ignore[union-attr]

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
            self.concat_ds = ConcatDataset(datasets)
            self.dataset_lengths = [len(ds) for ds in datasets]

    def train_dataloader(self) -> DataLoader:
        if self.concat_ds is None or self.dataset_lengths is None:
            raise RuntimeError("Datasets not initialized.")

        rank = self.trainer.global_rank if self.trainer is not None else int(os.environ.get("LOCAL_RANK", 0))

        max_steps = None
        if self.trainer is not None and self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            max_steps = self.trainer.max_steps

        batch_sampler = ScheduledBatchSampler(
            self.concat_ds,
            self.dataset_lengths,
            dataset_kinds=[cfg.kind for cfg in self.dataset_configs],
            schedules=[cfg.schedule for cfg in self.dataset_configs],  # type: ignore[misc]
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
            prefetch_factor=4 if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
            timeout=120 if self.num_workers > 0 else 0,
        )
