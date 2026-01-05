"""Complex image dataset for flat, combined or coilwise MRI data."""

from typing import Literal

import h5py
import numpy as np
import torch
from torch import Tensor

from .base import BaseMRIDataset
from .config import AugmentationConfig
from .kind import DatasetKind


class ComplexImageDataset(BaseMRIDataset):
    """Dataset for complex MRI images (flat, combined, or coilwise)."""

    def __init__(
        self,
        h5_path: str,
        mode: Literal["flat", "combined", "coilwise"] = "flat",
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.COMPLEX_T1W_COILWISE,
    ):
        self.mode = mode
        self._subject_keys: list[str] = []
        self._slice_index: list[tuple[str, int, int]] = []
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        container: h5py.File | h5py.Group = h5_file
        match self.mode:
            case "flat":
                self._subject_keys = [k for k in h5_file if isinstance(h5_file[k], h5py.Dataset)]
                container = h5_file
            case "combined" | "coilwise":
                if self.mode not in h5_file:
                    raise ValueError(f"Group '{self.mode}' not found in {self.h5_path}")
                container = h5_file[self.mode]
                if not isinstance(container, h5py.Group):
                    raise ValueError(f"'{self.mode}' must be a group")
                self._subject_keys = list(container.keys())

        self._slice_index = []
        for subject_key in self._subject_keys:
            data = container[subject_key]
            match self.mode:
                case "flat" | "combined":
                    for z_idx in range(data.shape[0]):
                        self._slice_index.append((subject_key, z_idx, 0))
                case "coilwise":
                    for z_idx in range(data.shape[0]):
                        for coil_idx in range(data.shape[1]):
                            self._slice_index.append((subject_key, z_idx, coil_idx))

        return len(self._slice_index)

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        first_key = self._subject_keys[0]
        match self.mode:
            case "flat":
                data = h5_file[first_key]
                return (data.shape[1], data.shape[2])
            case "combined":
                data = h5_file[self.mode][first_key]
                return (data.shape[1], data.shape[2])
            case "coilwise":
                data = h5_file[self.mode][first_key]
                return (data.shape[2], data.shape[3])

    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        subject_key, z_idx, coil_idx = self._slice_index[idx]
        match self.mode:
            case "flat":
                slice_data = h5_file[subject_key][z_idx, :, :]
            case "combined":
                slice_data = h5_file[self.mode][subject_key][z_idx, :, :]
            case "coilwise":
                slice_data = h5_file[self.mode][subject_key][z_idx, coil_idx, :, :]
        arr = np.array(slice_data)
        return torch.from_numpy(arr).to(torch.complex64)
