"""Complex image dataset for combined or coilwise MRI data."""

from typing import Literal

import h5py
import torch
from torch import Tensor

from .base import BaseMRIDataset
from .config import AugmentationConfig
from .kind import DatasetKind


class ComplexImageDataset(BaseMRIDataset):
    """Dataset for complex MRI images (combined or coilwise)."""

    def __init__(
        self,
        h5_path: str,
        mode: Literal["combined", "coilwise"],
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.COMPLEX_T1W_COILWISE,
    ):
        self.mode = mode
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        if self.mode == "combined":
            if "combined" not in h5_file:
                raise ValueError(f"Dataset 'combined' not found in {self.h5_path}")
            return int(h5_file["combined"].shape[0])
        else:
            if "coilwise" not in h5_file:
                raise ValueError(f"Dataset 'coilwise' not found in {self.h5_path}")
            data = h5_file["coilwise"]
            return int(data.shape[0]) * int(data.shape[1])

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        if self.mode == "combined":
            data = h5_file["combined"]
            return data.shape[1], data.shape[2]
        else:
            data = h5_file["coilwise"]
            return data.shape[2], data.shape[3]

    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        if self.mode == "combined":
            data = h5_file["combined"][idx, :, :]
        else:
            coilwise = h5_file["coilwise"]
            num_coils = int(coilwise.shape[0])
            num_z = int(coilwise.shape[1])
            coil_idx = idx % num_coils
            z_idx = (idx // num_coils) % num_z
            data = coilwise[coil_idx, z_idx, :, :]

        return torch.from_numpy(data).float()
