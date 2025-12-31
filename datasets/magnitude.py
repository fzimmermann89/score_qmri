"""Magnitude image dataset for MRI data.

Handles H5 files with flat subject datasets at root level:
- <subject_001>: (N_z, H, W) float32
- <subject_002>: (N_z, H, W) float32
"""

import h5py
import numpy as np
import torch
from torch import Tensor

from .base import BaseMRIDataset
from .config import AugmentationConfig
from .kind import DatasetKind


class MagnitudeImageDataset(BaseMRIDataset):
    """Dataset for magnitude MRI images.

    Expects H5 files with subject datasets at root level (no groups).
    Each dataset has shape (N_z, H, W) float32.
    """

    def __init__(
        self,
        h5_path: str,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.T1W_MAGNITUDE,
    ):
        """
        Parameters
        ----------
        h5_path
            Path to H5 file with subject datasets at root.
        augmentation_config
            Augmentation configuration.
        kind
            DatasetKind for conditioning.
        """
        self._subject_keys: list[str] = []
        self._slice_index: list[tuple[str, int]] = []  # (subject, z_idx)
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count total slices across all subjects."""
        self._subject_keys = [k for k in h5_file.keys() if isinstance(h5_file[k], h5py.Dataset)]
        if not self._subject_keys:
            raise ValueError(f"No datasets found in {self.h5_path}")

        self._slice_index = []
        for subject_key in self._subject_keys:
            data = h5_file[subject_key]
            n_z = data.shape[0]
            for z_idx in range(n_z):
                self._slice_index.append((subject_key, z_idx))

        return len(self._slice_index)

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get spatial dimensions (H, W) from first subject."""
        first_key = self._subject_keys[0]
        data = h5_file[first_key]
        return (data.shape[1], data.shape[2])

    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        """Load a single 2D slice."""
        subject_key, z_idx = self._slice_index[idx]
        data = h5_file[subject_key][z_idx, :, :]
        arr = np.array(data)
        return torch.from_numpy(arr).float()
