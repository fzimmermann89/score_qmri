"""Complex image dataset for combined or coilwise MRI data.

Handles H5 files with coilwise/combined groups:
- coilwise/<subject>: (N_z, N_coils, H, W) complex64
- combined/<subject>: (N_z, H, W) complex64
"""

from typing import Literal

import h5py
import numpy as np
import torch
from torch import Tensor

from .base import BaseMRIDataset
from .config import AugmentationConfig
from .kind import DatasetKind


class ComplexImageDataset(BaseMRIDataset):
    """Dataset for complex MRI images (combined or coilwise).

    Expects H5 files with 'coilwise' and/or 'combined' groups containing
    subject datasets.

    For coilwise mode, randomly samples coils during training to learn
    a prior over individual coil images.
    """

    def __init__(
        self,
        h5_path: str,
        mode: Literal["combined", "coilwise"],
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.COMPLEX_T1W_COILWISE,
    ):
        """
        Parameters
        ----------
        h5_path
            Path to H5 file with coilwise/combined groups.
        mode
            'combined' for coil-combined images, 'coilwise' for per-coil images.
        augmentation_config
            Augmentation configuration.
        kind
            DatasetKind for conditioning.
        """
        self.mode = mode
        self._subject_keys: list[str] = []
        self._slice_index: list[tuple[str, int, int]] = []  # (subject, z_idx, coil_idx)
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count total slices, building index mapping."""
        group_name = self.mode
        if group_name not in h5_file:
            raise ValueError(f"Group '{group_name}' not found in {self.h5_path}")

        group = h5_file[group_name]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"'{group_name}' must be a group containing subject datasets")

        self._subject_keys = list(group.keys())
        self._slice_index = []

        for subject_key in self._subject_keys:
            data = group[subject_key]
            if self.mode == "combined":
                # Shape: (N_z, H, W)
                n_z = data.shape[0]
                for z_idx in range(n_z):
                    self._slice_index.append((subject_key, z_idx, 0))
            else:
                # Shape: (N_z, N_coils, H, W)
                n_z, n_coils = data.shape[0], data.shape[1]
                for z_idx in range(n_z):
                    for coil_idx in range(n_coils):
                        self._slice_index.append((subject_key, z_idx, coil_idx))

        return len(self._slice_index)

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get spatial dimensions (H, W) from first subject."""
        group = h5_file[self.mode]
        first_key = self._subject_keys[0]
        data = group[first_key]
        if self.mode == "combined":
            return (data.shape[1], data.shape[2])
        else:
            return (data.shape[2], data.shape[3])

    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        """Load a single 2D slice as complex tensor."""
        subject_key, z_idx, coil_idx = self._slice_index[idx]
        group = h5_file[self.mode]
        data = group[subject_key]

        if self.mode == "combined":
            # Shape: (N_z, H, W)
            slice_data = data[z_idx, :, :]
        else:
            # Shape: (N_z, N_coils, H, W)
            if coil_idx == -1:
                # Random coil selection
                n_coils = data.shape[1]
                coil_idx = int(np.random.randint(0, n_coils))
            slice_data = data[z_idx, coil_idx, :, :]

        # Convert to torch complex tensor
        arr = np.array(slice_data)
        return torch.from_numpy(arr).to(torch.complex64)
