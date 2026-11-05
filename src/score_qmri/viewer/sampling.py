"""Sample manager with optimized sampling for the viewer."""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import Collate, ConstantSchedule
from score_qmri.models.diffusion import DEFAULT_KIND_SCALES

if TYPE_CHECKING:
    from score_qmri.datasets.lightning import DatasetConfig

logger = logging.getLogger(__name__)

# Output size after downsampling (smaller = faster network transfer)
OUTPUT_SIZE = 96

# Default number of samples per request (3 rows x 7 cols)
DEFAULT_N = 21


@dataclass
class SampleResult:
    """Result from sampling operation."""

    images: np.ndarray  # (n, 2, H, W) float32
    penumbra_masks: np.ndarray  # (n, H, W) uint8
    background_masks: np.ndarray  # (n, H, W) uint8
    labels: list[str]  # Dataset filenames
    kinds: list[str]  # DatasetKind names


class SampleManager:
    """Manages dataset sampling using training-realistic settings.

    Uses the same dataset and augmentation pipeline as training, with
    DataLoader workers enabled. Pre-warming can be done in background.
    """

    def __init__(self, config: dict, prewarm: bool = True) -> None:
        """
        Parameters
        ----------
        config
            Config dict from config_loader.load_config().
        prewarm
            If True, pre-create all datasets on init for faster first requests.
        """
        self.config = config
        self.dataset_configs: list[DatasetConfig] = config["dataset_configs"]
        self.raw_configs: list[dict] = config["raw_configs"]
        self.data_settings = config["data_settings"]

        self.sample_size = self.data_settings["sample_size"]  # (256, 256)
        self.kind_normalization = self.data_settings["kind_normalization"]
        self.base_seed = self.data_settings.get("base_seed", 42)
        num_workers = int(self.data_settings.get("num_workers", 4))
        self.num_workers = max(0, min(num_workers, 2))

        # Collate function from training code (handles crop/pad + 2-channel)
        self.collate = Collate(*self.sample_size)

        # Caches
        self._datasets: dict[int, Dataset[Any]] = {}
        self._dataset_lengths: dict[int, int] = {}
        self._dataloaders: dict[int, DataLoader[Any]] = {}
        self._dataloader_iters: dict[int, Any] = {}
        self._prefetch_results: dict[int, SampleResult] = {}
        self._prefetch_lock = threading.Lock()

        # Per-kind caches
        self._kind_dataset_indices: dict[str, list[int]] = {}

        self._request_count = 0
        self._is_prewarmed = False

        # Pre-warm datasets if requested
        if prewarm:
            self.prewarm_datasets()

    def prewarm_datasets(self, indices: list[int] | None = None, warm_dataloaders: bool = True) -> None:
        """Pre-create datasets and optionally warm DataLoader workers.

        Parameters
        ----------
        indices
            Specific dataset indices to prewarm. If None, prewarm all.
        warm_dataloaders
            If True, initialize DataLoader workers and fetch a warm batch.
        """
        if indices is None:
            if self._is_prewarmed:
                return
            indices = list(range(len(self.dataset_configs)))
            mark_complete = True
        else:
            mark_complete = False

        logger.info(f"Pre-warming {len(indices)} datasets (workers={self.num_workers})...")

        for idx in indices:
            try:
                self._get_dataset(idx)
            except Exception as e:
                logger.warning(f"Failed to prewarm dataset {idx}: {e}")

        if warm_dataloaders:
            for idx in indices:
                try:
                    self._warm_dataloader(idx)
                except Exception as e:
                    logger.warning(f"Failed to warm dataloader for dataset {idx}: {e}")

        if mark_complete:
            self._is_prewarmed = True
            logger.info(f"Pre-warming complete: {len(self._datasets)}/{len(self.dataset_configs)} datasets ready")

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

    def sample_dataset(self, dataset_idx: int, n: int = DEFAULT_N) -> SampleResult:
        """Sample n random items from a single dataset.

        Uses DataLoader with caching for speed.
        """
        self._request_count += 1

        # Serve prefetched result when available
        if n == DEFAULT_N:
            with self._prefetch_lock:
                cached = self._prefetch_results.pop(dataset_idx, None)
            if cached is not None:
                return cached

        # Get or create dataloader
        loader = self._get_dataloader(dataset_idx)
        batch = self._get_batch_from_loader(dataset_idx, loader, n)

        cfg = self.dataset_configs[dataset_idx]
        filename = Path(cfg.h5_path).stem
        actual_n = batch["image"].shape[0]

        return self._process_batch(batch, [filename] * actual_n, [cfg.kind.name] * actual_n)

    def sample_kind(self, kind: DatasetKind, n: int = DEFAULT_N) -> SampleResult:
        """Sample n items from datasets of given kind using training distribution.

        Uses weighted sampling matching training ratios.
        """
        self._request_count += 1
        ds_indices = self.get_kind_dataset_indices(kind)
        if not ds_indices:
            raise ValueError(f"No datasets found for kind {kind.name}")

        # Compute sampling weights
        weights = self._compute_kind_weights(ds_indices)
        probs = weights / weights.sum()

        # Sample dataset indices according to probabilities (local indices into ds_indices)
        selected_local_indices = torch.multinomial(probs, n, replacement=True).tolist()

        # Sample from each dataset
        samples = []
        labels = []
        kinds = []

        for local_idx in selected_local_indices:
            ds_idx = ds_indices[local_idx]  # Map local index back to actual dataset index
            ds = self._get_dataset(ds_idx)
            sample_idx = random.randint(0, self._get_dataset_length(ds_idx) - 1)
            samples.append(ds[sample_idx])

            cfg = self.dataset_configs[ds_idx]
            labels.append(Path(cfg.h5_path).stem)
            kinds.append(cfg.kind.name)

        batch = self.collate(samples)
        return self._process_batch(batch, labels, kinds)

    def get_kind_dataset_indices(self, kind: DatasetKind) -> list[int]:
        """Get dataset indices for a kind (cached)."""
        kind_name = kind.name
        if kind_name not in self._kind_dataset_indices:
            indices = [i for i, cfg in enumerate(self.dataset_configs) if cfg.kind == kind]
            self._kind_dataset_indices[kind_name] = indices
        return self._kind_dataset_indices[kind_name]

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

    def _process_batch(self, batch: dict[str, torch.Tensor], labels: list[str], kinds: list[str]) -> SampleResult:
        """Process batch: downsample and convert to numpy."""
        images = batch["image"]  # (n, 2, H, W)
        penumbra = batch["penumbra_mask"]  # (n, H, W)
        background = batch["background_mask"]  # (n, H, W)

        # Downsample images using area interpolation
        images_down = F.interpolate(images, size=OUTPUT_SIZE, mode="area")

        # Apply per-kind denormalization for mapping kinds (viewer should show full scale)
        images_down = self._apply_kind_scales(images_down, kinds)

        # Downsample masks using nearest neighbor
        penumbra_down = F.interpolate(penumbra.unsqueeze(1).float(), size=OUTPUT_SIZE, mode="nearest").squeeze(1)
        background_down = F.interpolate(background.unsqueeze(1).float(), size=OUTPUT_SIZE, mode="nearest").squeeze(1)

        return SampleResult(
            images=images_down.numpy().astype(np.float32),
            penumbra_masks=(penumbra_down > 0.5).numpy().astype(np.uint8),
            background_masks=(background_down > 0.5).numpy().astype(np.uint8),
            labels=labels,
            kinds=kinds,
        )

    def _apply_kind_scales(self, images: torch.Tensor, kinds: list[str]) -> torch.Tensor:
        """Apply per-kind normalization for mapping kinds only."""
        if not kinds:
            return images
        scales = torch.ones(len(kinds), device=images.device, dtype=images.dtype)
        for idx, kind_name in enumerate(kinds):
            scale = DEFAULT_KIND_SCALES.get(kind_name)
            if scale is not None:
                scales[idx] = float(scale)
        if torch.all(scales == 1):
            return images
        return images * scales.view(-1, 1, 1, 1)

    def _get_dataset(self, idx: int) -> Dataset[Any]:
        """Get or create dataset instance with training augmentations."""
        if idx not in self._datasets:
            cfg = self.dataset_configs[idx]
            self._datasets[idx] = cfg.create()
            self._dataset_lengths[idx] = self._dataset_len(self._datasets[idx])
        return self._datasets[idx]

    @staticmethod
    def _dataset_len(dataset: Dataset[Any]) -> int:
        """Get length from dataset with explicit Sized cast."""
        return len(cast(Sized, dataset))

    def _get_dataset_length(self, idx: int) -> int:
        """Get dataset length (cached)."""
        if idx not in self._dataset_lengths:
            self._get_dataset(idx)
        return self._dataset_lengths[idx]

    def _warm_dataloader(self, dataset_idx: int) -> None:
        """Warm DataLoader workers by fetching one batch."""
        loader = self._get_dataloader(dataset_idx)
        batch = self._get_batch_from_loader(dataset_idx, loader, DEFAULT_N)

        cfg = self.dataset_configs[dataset_idx]
        filename = Path(cfg.h5_path).stem
        actual_n = batch["image"].shape[0]
        result = self._process_batch(batch, [filename] * actual_n, [cfg.kind.name] * actual_n)

        with self._prefetch_lock:
            self._prefetch_results[dataset_idx] = result

    def _get_dataloader(self, dataset_idx: int) -> DataLoader[Any]:
        """Get or create DataLoader for dataset."""
        if dataset_idx not in self._dataloaders:
            ds = self._get_dataset(dataset_idx)
            self._dataloaders[dataset_idx] = DataLoader(
                ds,
                batch_size=DEFAULT_N,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collate,
                persistent_workers=self.num_workers > 0,
                pin_memory=False,
                prefetch_factor=4 if self.num_workers > 0 else None,
                drop_last=False,
            )
        return self._dataloaders[dataset_idx]

    def _get_batch_from_loader(self, dataset_idx: int, loader: DataLoader[Any], n: int) -> dict[str, torch.Tensor]:
        """Get a batch from dataloader, handling iterator exhaustion."""
        if n < 1:
            raise ValueError("Sample count must be positive")
        batches = []
        count = 0
        while count < n:
            if dataset_idx not in self._dataloader_iters:
                self._dataloader_iters[dataset_idx] = iter(loader)
            try:
                batch = next(self._dataloader_iters[dataset_idx])
            except StopIteration:
                self._dataloader_iters[dataset_idx] = iter(loader)
                batch = next(self._dataloader_iters[dataset_idx])
            batches.append(batch)
            count += batch["image"].shape[0]
        return {key: torch.cat([batch[key] for batch in batches], dim=0)[:n] for key in ("image", "penumbra_mask", "background_mask")}

    def partial_reload(self, new_config: dict, changed_indices: set[int]) -> None:
        """Reload only the datasets that changed.

        Parameters
        ----------
        new_config
            New config dict from load_config().
        changed_indices
            Set of dataset indices that changed.
        """
        # Update config references
        self.config = new_config
        self.dataset_configs = new_config["dataset_configs"]
        self.raw_configs = new_config["raw_configs"]
        self.data_settings = new_config["data_settings"]

        # Clear caches only for changed datasets
        for idx in changed_indices:
            self._datasets.pop(idx, None)
            self._dataset_lengths.pop(idx, None)
            self._dataloaders.pop(idx, None)
            self._dataloader_iters.pop(idx, None)
            self._prefetch_results.pop(idx, None)

        # Clear kind caches if any dataset of that kind changed
        kinds_to_clear = set()
        for idx in changed_indices:
            if idx < len(self.dataset_configs):
                kinds_to_clear.add(self.dataset_configs[idx].kind.name)

        for kind_name in kinds_to_clear:
            self._kind_dataset_indices.pop(kind_name, None)

        logger.info(f"Partial reload complete: {len(changed_indices)} datasets reloaded")

    def clear_cache(self) -> None:
        """Clear all caches (called on full config reload)."""
        self._datasets.clear()
        self._dataset_lengths.clear()
        self._dataloaders.clear()
        self._dataloader_iters.clear()
        self._kind_dataset_indices.clear()
        self._prefetch_results.clear()
        self._is_prewarmed = False
