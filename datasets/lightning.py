"""PyTorch Lightning DataModule for MRI datasets."""

import os
from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from .batch_sampler import ScheduledBatchSampler
from .complex import ComplexImageDataset
from .concat import create_concat_dataset, get_size_labels
from .magnitude import MagnitudeImageDataset
from .mapping import MappingDataset


def create_dataset_from_config(
    config: dict[str, Any],
) -> ComplexImageDataset | MagnitudeImageDataset | MappingDataset:
    """Create a dataset from configuration dictionary."""
    dataset_type = config.get("type")
    h5_path = config.get("h5_path")

    if not isinstance(h5_path, str):
        raise ValueError("h5_path must be a string in dataset configuration")

    match dataset_type:
        case "complex":
            return ComplexImageDataset(
                h5_path=h5_path,
                mode=config.get("mode", "combined"),
                augmentation_config=config.get("augmentation_config"),
                kind=config.get("kind", 0),
            )
        case "magnitude":
            return MagnitudeImageDataset(
                h5_path=h5_path,
                dataset_path=config.get("dataset_path", "magnitude"),
                augmentation_config=config.get("augmentation_config"),
                kind=config.get("kind", 1),
            )
        case "mapping":
            return MappingDataset(
                h5_path=h5_path,
                dataset_path=config.get("dataset_path", "mapping"),
                scaling_factor=config.get("scaling_factor", 1.0),
                augmentation_config=config.get("augmentation_config"),
                kind=config.get("kind", 2),
            )
        case _:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'complex', 'magnitude', or 'mapping'")


def _linear_schedule(target_step: int) -> Callable[[int], float]:
    """Create a linear schedule that goes from 0 to 1 at target_step."""
    def schedule(step: int) -> float:
        if target_step <= 0:
            return 1.0
        return min(1.0, step / target_step)
    return schedule


class MRIDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for MRI datasets with scheduled sampling.

    All parameters are CLI-configurable via jsonargparse.
    """

    def __init__(
        self,
        dataset_configs: list[dict[str, Any]] | None = None,
        schedule_ramp_steps: list[int] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        base_seed: int = 42,
        val_dataset_configs: list[dict[str, Any]] | None = None,
        val_batch_size: int | None = None,
    ):
        """Initialize the DataModule.

        Args:
            dataset_configs: List of dataset configuration dicts with keys:
                - type: 'complex', 'magnitude', or 'mapping'
                - h5_path: Path to HDF5 file
                - Additional dataset-specific params
            schedule_ramp_steps: Steps at which each dataset reaches full weight.
                If None, all datasets are weighted equally.
            batch_size: Training batch size.
            num_workers: DataLoader workers.
            base_seed: Random seed for reproducibility.
            val_dataset_configs: Validation dataset configs (same format as above).
            val_batch_size: Validation batch size (defaults to batch_size).
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_configs = dataset_configs or []
        self.schedule_ramp_steps = schedule_ramp_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.val_dataset_configs = val_dataset_configs
        self.val_batch_size = val_batch_size or batch_size

        self.concat_ds: ConcatDataset | None = None
        self.size_labels: list[tuple[int, int]] | None = None
        self.val_concat_ds: ConcatDataset | None = None

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            if not self.dataset_configs:
                return

            datasets = [create_dataset_from_config(cfg) for cfg in self.dataset_configs]
            self.concat_ds = create_concat_dataset(datasets)
            self.size_labels = get_size_labels(datasets)

            if self.val_dataset_configs is not None:
                val_datasets = [create_dataset_from_config(cfg) for cfg in self.val_dataset_configs]
                self.val_concat_ds = create_concat_dataset(val_datasets)

    def _create_schedule_functions(self) -> list[Callable[[int], float]]:
        """Create schedule functions from ramp steps."""
        if self.schedule_ramp_steps is None:
            # Equal weighting for all datasets
            n_datasets = len(self.dataset_configs)
            return [lambda _: 1.0 / n_datasets] * n_datasets
        return [_linear_schedule(steps) for steps in self.schedule_ramp_steps]

    def train_dataloader(self) -> DataLoader:
        if self.concat_ds is None or self.size_labels is None:
            raise RuntimeError("Must call setup() before train_dataloader()")

        rank = self.trainer.global_rank if self.trainer is not None else int(os.environ.get("LOCAL_RANK", 0))
        schedule_functions = self._create_schedule_functions()

        batch_sampler = ScheduledBatchSampler(
            self.concat_ds,
            self.size_labels,
            schedule_functions,
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

    def val_dataloader(self) -> DataLoader | None:
        if self.val_concat_ds is None:
            return None

        return DataLoader(
            self.val_concat_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if self.num_workers > 0 else 2,
        )
