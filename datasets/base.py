"""Base dataset class for MRI data with H5 file handling."""

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .augmentation import AugmentationPipeline
from .config import AugmentationConfig
from .kind import DatasetKind


class BaseMRIDataset(Dataset[dict[str, Any]], ABC):
    """Base class for MRI datasets with H5 file handling.

    Handles H5 file lifecycle: counts slices in __init__ and closes file,
    then opens lazily in __getitem__ with per-worker caching.
    """

    def __init__(
        self,
        h5_path: str,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.COMPLEX_T1W_COILWISE,
    ) -> None:
        self.h5_path = str(Path(h5_path).resolve())
        self.kind = kind
        self.augmentation = AugmentationPipeline(augmentation_config) if augmentation_config else None
        self._h5_file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as f:
            self.num_slices = self._count_slices_in_file(f)
            self.slice_shape = self._get_slice_shape(f)

    @abstractmethod
    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count total slices in H5 file."""
        ...

    @abstractmethod
    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get shape of a single slice (H, W)."""
        ...

    @abstractmethod
    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        """Load single slice by index as a float Tensor."""
        ...

    def _get_h5_file(self) -> h5py.File:
        """Get cached H5 file handle, opening if needed."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return self.num_slices

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample with image, mask, and kind tensors."""
        image = self._load_slice(self._get_h5_file(), idx)
        mask = image != 0

        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        return {
            "image": image,
            "mask": mask,
            "kind": torch.tensor(self.kind.value, dtype=torch.long),
        }

    def __del__(self) -> None:
        if self._h5_file is not None:
            with contextlib.suppress(Exception):
                self._h5_file.close()
