"""Base dataset class for MRI data with H5 file handling."""

import contextlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor
from torch.utils.data import Dataset

from .augmentation import AugmentationPipeline
from .config import AugmentationConfig
from .kind import DatasetKind


def _compute_penumbra(signal_mask: np.ndarray, width: int) -> np.ndarray:
    """Compute penumbra mask using distance transform (O(N) vs O(k*N) for dilation)."""
    dist: np.ndarray = distance_transform_edt(~signal_mask)
    return (dist > 0) & (dist <= width)


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
        scaling_factor: float = 1.0,
        penumbra_width: int = 30,
    ) -> None:
        self.h5_path = str(Path(h5_path).resolve())
        self.kind = kind
        self.scaling_factor = scaling_factor
        self.penumbra_width = penumbra_width
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
        """Get sample with image, masks, and kind tensors."""
        image = self._load_slice(self._get_h5_file(), idx)

        if image.is_complex() != self.kind.is_complex:
            raise ValueError(
                f"Kind/dtype mismatch: kind={self.kind.name} (complex={self.kind.is_complex}) "
                f"but loaded image is_complex={image.is_complex()} from {self.h5_path}"
            )

        signal_mask = image.abs() > 1e-6

        if self.scaling_factor != 1.0:
            image = image * self.scaling_factor

        if self.augmentation is not None:
            image, signal_mask = self.augmentation(image, signal_mask)

        background_mask = ~signal_mask

        if self.penumbra_width < 0:
            penumbra_mask = background_mask
        elif self.penumbra_width == 0:
            penumbra_mask = torch.zeros_like(signal_mask)
        else:
            penumbra_mask = torch.from_numpy(_compute_penumbra(signal_mask.numpy(), self.penumbra_width))

        return {
            "image": image,
            "penumbra_mask": penumbra_mask,
            "background_mask": background_mask,
            "kind": torch.tensor(self.kind.value, dtype=torch.long),
            "h5_path": self.h5_path,
        }

    def __del__(self) -> None:
        if self._h5_file is not None:
            with contextlib.suppress(Exception):
                self._h5_file.close()
