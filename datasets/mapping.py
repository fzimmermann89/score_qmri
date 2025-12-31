"""Quantitative mapping dataset for MRI data.

Handles H5 files with flat subject datasets at root level:
- <subject_001>: (N_z, H, W) float32
- <subject_002>: (N_z, H, W) float32

Applies scaling factor to normalize quantitative values (e.g., T1 in ms) to ~[0, 1].
"""

import h5py
import numpy as np
import torch
from torch import Tensor

from .base import BaseMRIDataset
from .config import AugmentationConfig
from .kind import DatasetKind


class MappingDataset(BaseMRIDataset):
    """Dataset for quantitative MRI parameter maps (T1, T2, etc.).

    Expects H5 files with subject datasets at root level (no groups).
    Each dataset has shape (N_z, H, W) float32.

    Applies a scaling factor to normalize values to ~[0, 1] range.
    For example, T1 values in ms (0-3000) can be scaled by 1/3000.
    """

    def __init__(
        self,
        h5_path: str,
        scaling_factor: float,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.T1MAP,
    ):
        """
        Parameters
        ----------
        h5_path
            Path to H5 file with subject datasets at root.
        scaling_factor
            Factor to multiply values by (e.g., 1/3000 for T1 in ms).
        augmentation_config
            Augmentation configuration.
        kind
            DatasetKind for conditioning.
        """
        self.scaling_factor = scaling_factor
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
        """Load a single 2D slice with scaling applied."""
        subject_key, z_idx = self._slice_index[idx]
        data = h5_file[subject_key][z_idx, :, :]
        arr = np.array(data)
        return torch.from_numpy(arr).float() * self.scaling_factor
