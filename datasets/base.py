"""Base dataset class for MRI data with H5 file handling."""

import os
from abc import ABC, abstractmethod
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .augmentation import AugmentationPipeline
from .config import AugmentationConfig


class BaseMRIDataset(Dataset, ABC):
    """Base class for MRI datasets with H5 file handling."""

    def __init__(
        self,
        h5_path: str,
        augmentation_config: AugmentationConfig | None = None,
        kind: int = 0,
    ):
        """Initialize base MRI dataset.

        Args:
            h5_path: Path to H5 file
            augmentation_config: Augmentation configuration. If None, no augmentation is applied.
            kind: Integer value for conditioning (returned with each sample)
        """
        self.h5_path = os.path.abspath(h5_path)
        self.kind = kind
        self.augmentation = (
            AugmentationPipeline(augmentation_config)
            if augmentation_config is not None
            else None
        )

        # Count slices and store metadata
        self._count_slices()

        # File handle will be cached in __getitem__
        self._h5_file: h5py.File | None = None

    def _count_slices(self) -> None:
        """Count total number of slices and store metadata.

        Opens H5 file, counts slices, stores shape information, then closes file.
        """
        with h5py.File(self.h5_path, "r") as f:
            self.num_slices = self._count_slices_in_file(f)
            self.slice_shape = self._get_slice_shape(f)

    @abstractmethod
    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count number of slices in H5 file.

        Args:
            h5_file: Open H5 file handle

        Returns:
            Total number of slices
        """
        pass

    @abstractmethod
    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get shape of a single slice.

        Args:
            h5_file: Open H5 file handle

        Returns:
            Shape tuple for a single slice
        """
        pass

    @abstractmethod
    def _load_slice(self, h5_file: h5py.File, idx: int) -> np.ndarray:
        """Load a single slice from H5 file.

        Args:
            h5_file: Open H5 file handle
            idx: Slice index

        Returns:
            Slice data as numpy array
        """
        pass

    def _get_mask(self, data: np.ndarray) -> np.ndarray:
        """Generate mask from data (data != 0).

        Args:
            data: Data array

        Returns:
            Boolean mask array
        """
        return data != 0

    def _get_h5_file(self) -> h5py.File:
        """Get or create cached H5 file handle.

        Returns:
            Open H5 file handle
        """
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        """Return total number of slices."""
        return self.num_slices

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys: 'image', 'mask', 'kind'
        """
        h5_file = self._get_h5_file()
        data = self._load_slice(h5_file, idx)

        # Convert to tensor
        image = torch.from_numpy(data).float()

        # Generate mask
        mask = torch.from_numpy(self._get_mask(data)).bool()

        # Apply augmentation if configured
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        return {
            "image": image,
            "mask": mask,
            "kind": torch.tensor(self.kind, dtype=torch.long),
        }

    def __del__(self) -> None:
        """Close H5 file handle if open."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

