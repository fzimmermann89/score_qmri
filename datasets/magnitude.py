"""Magnitude image dataset for MRI data."""

import h5py
import torch
from torch import Tensor

from .base import BaseMRIDataset
from .config import AugmentationConfig
from .kind import DatasetKind


class MagnitudeImageDataset(BaseMRIDataset):
    """Dataset for magnitude MRI images."""

    def __init__(
        self,
        h5_path: str,
        dataset_path: str,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.T1W_MAGNITUDE,
    ):
        self.dataset_path = dataset_path
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        if self.dataset_path not in h5_file:
            raise ValueError(f"Dataset '{self.dataset_path}' not found in {self.h5_path}")
        return int(h5_file[self.dataset_path].shape[0])

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        data = h5_file[self.dataset_path]
        return data.shape[1], data.shape[2]

    def _load_slice(self, h5_file: h5py.File, idx: int) -> Tensor:
        data = h5_file[self.dataset_path][idx, :, :]
        return torch.from_numpy(data).float()
