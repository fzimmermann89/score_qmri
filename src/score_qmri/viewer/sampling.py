"""Sample manager with DataLoader caching for fast sampling."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import Collate, ConstantSchedule

if TYPE_CHECKING:
    from score_qmri.datasets.lightning import DatasetConfig

logger = logging.getLogger(__name__)

# Output size after downsampling
OUTPUT_SIZE = 128


@dataclass
class SampleResult:
    """Result from sampling operation."""

    images: np.ndarray  # (n, 2, H, W) float32
    penumbra_masks: np.ndarray  # (n, H, W) uint8
    background_masks: np.ndarray  # (n, H, W) uint8
    labels: list[str]  # Dataset filenames
    kinds: list[str]  # DatasetKind names


class SampleManager:
    """Manages dataset sampling with DataLoader caching for speed."""

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config
            Config dict from config_loader.load_config().
        """
        self.config = config
        self.dataset_configs: list[DatasetConfig] = config["dataset_configs"]
        self.raw_configs: list[dict] = config["raw_configs"]
        self.data_settings = config["data_settings"]

        self.sample_size = self.data_settings["sample_size"]  # (256, 256)
        self.kind_normalization = self.data_settings["kind_normalization"]
        self.num_workers = min(self.data_settings.get("num_workers", 4), 4)
        self.base_seed = self.data_settings.get("base_seed", 42)

        # Collate function from training code (handles crop/pad + 2-channel)
        self.collate = Collate(*self.sample_size)

        # Caches
        self._datasets: dict[int, Dataset] = {}
        self._dataset_lengths: dict[int, int] = {}
        self._dataloaders: dict[int, DataLoader] = {}
        self._dataloader_iters: dict[int, Iterator] = {}

        # Per-kind caches
        self._kind_dataloaders: dict[str, DataLoader] = {}
        self._kind_dataloader_iters: dict[str, Iterator] = {}
        self._kind_dataset_indices: dict[str, list[int]] = {}

        self._request_count = 0

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all datasets with metadata."""
        result = []
        for idx, cfg in enumerate(self.dataset_configs):
            ds_len = self._get_dataset_length(idx)
            result.append(
                {
                    "dataset_idx": idx,
                    "total_samples": ds_len,
                    "h5_path": cfg.h5_path,
                    "h5_filename": Path(cfg.h5_path).stem,
                    "kind": cfg.kind.name,
                    "type": cfg.type,
                    "scaling_factor": getattr(cfg, "scaling_factor", 1.0),
                    "penumbra_width": cfg.penumbra_width,
                    "schedule": {
                        "type": cfg.schedule.type if cfg.schedule else "constant",
                        "weight": cfg.schedule.weight if cfg.schedule else 1.0,
                    },
                    "has_augmentation": cfg.augmentation_config is not None,
                }
            )
        return result

    def get_unique_kinds(self) -> list[dict[str, Any]]:
        """Get unique kinds with their dataset counts."""
        kind_info: dict[str, int] = {}
        for cfg in self.dataset_configs:
            name = cfg.kind.name
            kind_info[name] = kind_info.get(name, 0) + 1
        return [{"name": k, "count": v} for k, v in sorted(kind_info.items())]

    def sample_dataset(self, dataset_idx: int, n: int = 32) -> SampleResult:
        """Sample n random items from a single dataset.

        Uses DataLoader with caching for speed.
        """
        self._request_count += 1

        # Get or create dataloader
        loader = self._get_dataloader(dataset_idx)
        batch = self._get_batch_from_loader(dataset_idx, loader, n)

        cfg = self.dataset_configs[dataset_idx]
        filename = Path(cfg.h5_path).stem

        return self._process_batch(batch, [filename] * n, [cfg.kind.name] * n)

    def sample_kind(self, kind: DatasetKind, n: int = 32) -> SampleResult:
        """Sample n items from datasets of given kind using training distribution.

        Uses weighted sampling matching training ratios.
        """
        self._request_count += 1
        kind_name = kind.name

        # Get datasets for this kind
        if kind_name not in self._kind_dataset_indices:
            indices = [i for i, cfg in enumerate(self.dataset_configs) if cfg.kind == kind]
            if not indices:
                raise ValueError(f"No datasets found for kind {kind_name}")
            self._kind_dataset_indices[kind_name] = indices

        ds_indices = self._kind_dataset_indices[kind_name]

        # Compute sampling weights
        weights = self._compute_kind_weights(ds_indices)
        probs = weights / weights.sum()

        # Sample dataset indices according to probabilities
        selected_ds_indices = torch.multinomial(probs, n, replacement=True).tolist()

        # Sample from each dataset
        samples = []
        labels = []
        kinds = []

        for ds_idx in selected_ds_indices:
            ds = self._get_dataset(ds_idx)
            sample_idx = random.randint(0, len(ds) - 1)
            samples.append(ds[sample_idx])

            cfg = self.dataset_configs[ds_idx]
            labels.append(Path(cfg.h5_path).stem)
            kinds.append(cfg.kind.name)

        batch = self.collate(samples)
        return self._process_batch(batch, labels, kinds)

    def _compute_kind_weights(self, ds_indices: list[int]) -> torch.Tensor:
        """Compute sampling weights for datasets.

        Phase-in: use full weight (assume fully phased in)
        Phase-out: use full weight (assume not yet phased out)
        """
        weights = []
        for idx in ds_indices:
            cfg = self.dataset_configs[idx]
            schedule = cfg.schedule or ConstantSchedule()
            w = schedule.weight
            ds_len = self._get_dataset_length(idx)
            weights.append(w * ds_len)

        return torch.tensor(weights, dtype=torch.float32)

    def _process_batch(
        self, batch: dict[str, torch.Tensor], labels: list[str], kinds: list[str]
    ) -> SampleResult:
        """Process batch: downsample and convert to numpy."""
        images = batch["image"]  # (n, 2, H, W)
        penumbra = batch["penumbra_mask"]  # (n, H, W)
        background = batch["background_mask"]  # (n, H, W)

        # Downsample images by factor 2 using area interpolation
        images_down = F.interpolate(images, size=OUTPUT_SIZE, mode="area")

        # Downsample masks using nearest neighbor
        penumbra_down = F.interpolate(
            penumbra.unsqueeze(1).float(), size=OUTPUT_SIZE, mode="nearest"
        ).squeeze(1)
        background_down = F.interpolate(
            background.unsqueeze(1).float(), size=OUTPUT_SIZE, mode="nearest"
        ).squeeze(1)

        return SampleResult(
            images=images_down.numpy().astype(np.float32),
            penumbra_masks=(penumbra_down > 0.5).numpy().astype(np.uint8),
            background_masks=(background_down > 0.5).numpy().astype(np.uint8),
            labels=labels,
            kinds=kinds,
        )

    def _get_dataset(self, idx: int) -> Dataset:
        """Get or create dataset instance."""
        if idx not in self._datasets:
            cfg = self.dataset_configs[idx]
            self._datasets[idx] = cfg.create()
            self._dataset_lengths[idx] = len(self._datasets[idx])
        return self._datasets[idx]

    def _get_dataset_length(self, idx: int) -> int:
        """Get dataset length (cached)."""
        if idx not in self._dataset_lengths:
            self._get_dataset(idx)
        return self._dataset_lengths[idx]

    def _get_dataloader(self, dataset_idx: int) -> DataLoader:
        """Get or create DataLoader for dataset."""
        if dataset_idx not in self._dataloaders:
            ds = self._get_dataset(dataset_idx)
            self._dataloaders[dataset_idx] = DataLoader(
                ds,
                batch_size=32,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collate,
                persistent_workers=self.num_workers > 0,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4 if self.num_workers > 0 else None,
                drop_last=False,
            )
        return self._dataloaders[dataset_idx]

    def _get_batch_from_loader(
        self, dataset_idx: int, loader: DataLoader, n: int
    ) -> dict[str, torch.Tensor]:
        """Get a batch from dataloader, handling iterator exhaustion."""
        if dataset_idx not in self._dataloader_iters:
            self._dataloader_iters[dataset_idx] = iter(loader)

        try:
            batch = next(self._dataloader_iters[dataset_idx])
        except StopIteration:
            # Recreate iterator
            self._dataloader_iters[dataset_idx] = iter(loader)
            batch = next(self._dataloader_iters[dataset_idx])

        # If batch is smaller than n, pad with more samples
        actual_n = batch["image"].shape[0]
        if actual_n < n:
            # Just return what we have - this is fine for viewer
            pass

        return batch

    def clear_cache(self) -> None:
        """Clear all caches (called on config reload)."""
        self._datasets.clear()
        self._dataset_lengths.clear()
        self._dataloaders.clear()
        self._dataloader_iters.clear()
        self._kind_dataloaders.clear()
        self._kind_dataloader_iters.clear()
        self._kind_dataset_indices.clear()
