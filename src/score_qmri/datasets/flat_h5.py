"""Base class for flat H5 datasets with subject datasets at root level.

Handles H5 files with structure:
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


class FlatH5Dataset(BaseMRIDataset):
    """Base class for flat H5 files with subject datasets at root.

    Subclasses can override `_process_slice` to apply custom processing
    (e.g., scaling for quantitative maps).
    """

    def __init__(
        self,
        h5_path: str,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.T1W_MAGNITUDE,
        scaling_factor: float = 1.0,
        penumbra_width: int = 30,
    ) -> None:
        """
        Parameters
        ----------
        h5_path
            Path to H5 file with subject datasets at root.
        augmentation_config
            Augmentation configuration.
        kind
            DatasetKind for conditioning.
        scaling_factor
            Factor to multiply values by for normalization.
        penumbra_width
            Penumbra dilation width: >0 for ring, 0 for none, <0 for infinite.
        """
        self._subject_keys: list[str] = []
        self._slice_index: list[tuple[str, int]] = []
        super().__init__(h5_path, augmentation_config, kind, scaling_factor, penumbra_width)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count total slices across all subjects."""
        self._subject_keys = [k for k in h5_file if isinstance(h5_file[k], h5py.Dataset)]
        if not self._subject_keys:
            raise ValueError(f"No datasets found in {self.h5_path}")

        self._slice_index = [(subject_key, z_idx) for subject_key in self._subject_keys for z_idx in range(h5_file[subject_key].shape[0])]
        return len(self._slice_index)

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get spatial dimensions (H, W) from first subject."""
        data = h5_file[self._subject_keys[0]]
        return (data.shape[1], data.shape[2])

    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        """Load a single 2D slice."""
        subject_key, z_idx = self._slice_index[idx]
        data = h5_file[subject_key][z_idx, :, :]
        arr = np.array(data)
        return self._process_slice(torch.from_numpy(arr).float())

    def _process_slice(self, tensor: Tensor) -> Tensor:
        """Process loaded slice. Override in subclasses for custom processing."""
        return tensor
