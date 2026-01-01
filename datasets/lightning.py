"""PyTorch Lightning DataModule for MRI datasets."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from .batch_sampler import ScheduledBatchSampler
from .complex import ComplexImageDataset
from .concat import create_concat_dataset, get_size_labels
from .kind import DatasetKind
from .magnitude import MagnitudeImageDataset
from .mapping import MappingDataset


@dataclass
class MRIDataModuleHparams:
    """Type-safe hyperparameter definitions for MRIDataModule."""

    dataset_configs: list[dict[str, Any]] | None
    schedule_ramp_steps: list[int] | None
    batch_size: int
    num_workers: int
    base_seed: int


def create_dataset_from_config(
    config: dict[str, Any],
) -> ComplexImageDataset | MagnitudeImageDataset | MappingDataset:
    """
    Create a dataset from configuration dictionary.

    Parameters
    ----------
    config
        Configuration dict with keys: type, h5_path, and dataset-specific params.

    Returns
    -------
    Dataset
        Created dataset instance.

    Raises
    ------
    ValueError
        If h5_path is missing or type is unknown.
    """
    dataset_type = config.get("type")
    h5_path = config.get("h5_path")

    if not isinstance(h5_path, str):
        raise ValueError("h5_path must be a string in dataset configuration")

    def _get_kind(kind_value: int | DatasetKind | None, default: DatasetKind) -> DatasetKind:
        """Convert kind value to DatasetKind enum."""
        if kind_value is None:
            return default
        if isinstance(kind_value, DatasetKind):
            return kind_value
        if isinstance(kind_value, int):
            return DatasetKind(kind_value)
        raise ValueError(f"kind must be int or DatasetKind, got {type(kind_value)}")

    match dataset_type:
        case "complex":
            return ComplexImageDataset(
                h5_path=h5_path,
                mode=config.get("mode", "combined"),
                augmentation_config=config.get("augmentation_config"),
                kind=_get_kind(config.get("kind"), DatasetKind.COMPLEX_T1W_COILWISE),
            )
        case "magnitude":
            return MagnitudeImageDataset(
                h5_path=h5_path,
                augmentation_config=config.get("augmentation_config"),
                kind=_get_kind(config.get("kind"), DatasetKind.T1W_MAGNITUDE),
            )
        case "mapping":
            return MappingDataset(
                h5_path=h5_path,
                scaling_factor=config.get("scaling_factor", 1.0),
                augmentation_config=config.get("augmentation_config"),
                kind=_get_kind(config.get("kind"), DatasetKind.T1MAP),
            )
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


def _linear_schedule(target_step: int) -> Callable[[int], float]:
    """Create linear schedule from 0 to 1 at target_step."""

    def schedule(step: int) -> float:
        return 1.0 if target_step <= 0 else min(1.0, step / target_step)

    return schedule


def _constant_schedule(weight: float) -> Callable[[int], float]:
    """Create constant schedule returning fixed weight."""

    def schedule(_: int) -> float:
        return weight

    return schedule


class MRIDataModule(LightningDataModule):
    """DataModule for MRI training datasets with scheduled sampling."""

    hparams: MRIDataModuleHparams  # type: ignore[assignment]

    def __init__(
        self,
        dataset_configs: list[dict[str, Any]] | None = None,
        schedule_ramp_steps: list[int] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        base_seed: int = 42,
    ) -> None:
        """
        Initialize MRI data module.

        Parameters
        ----------
        dataset_configs
            List of dataset configuration dicts with keys: type, h5_path, etc.
        schedule_ramp_steps
            Steps at which each dataset reaches full weight.
        batch_size
            Training batch size.
        num_workers
            DataLoader workers.
        base_seed
            Random seed for reproducibility.
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_configs = dataset_configs or []
        self.schedule_ramp_steps = schedule_ramp_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_seed = base_seed

        self.concat_ds: ConcatDataset | None = None
        self.size_labels: list[tuple[int, int]] | None = None

    def setup(self, stage: str) -> None:
        """Set up datasets for training."""
        if (stage == "fit" or stage is None) and self.dataset_configs:
            datasets = [create_dataset_from_config(cfg) for cfg in self.dataset_configs]
            self.concat_ds = create_concat_dataset(datasets)
            self.size_labels = get_size_labels(datasets)

    def _create_schedule_functions(self) -> list[Callable[[int], float]]:
        """Create schedule functions from ramp steps."""
        if self.schedule_ramp_steps is None:
            n = len(self.dataset_configs)
            weight = 1.0 / n
            return [_constant_schedule(weight) for _ in range(n)]
        return [_linear_schedule(steps) for steps in self.schedule_ramp_steps]

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with scheduled batch sampler."""
        if self.concat_ds is None or self.size_labels is None:
            raise RuntimeError("Must call setup() before train_dataloader()")

        rank = self.trainer.global_rank if self.trainer is not None else int(os.environ.get("LOCAL_RANK", 0))

        batch_sampler = ScheduledBatchSampler(
            self.concat_ds,
            self.size_labels,
            self._create_schedule_functions(),
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
            prefetch_factor=4 if self.num_workers > 0 else 2,
        )
