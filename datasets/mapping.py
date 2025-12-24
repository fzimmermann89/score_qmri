"""Mapping dataset for MRI data with custom scaling."""

import h5py
import numpy as np

from .base import BaseMRIDataset
from .config import AugmentationConfig


class MappingDataset(BaseMRIDataset):
    """Dataset for mapping MRI data with custom scaling factor."""

    def __init__(
        self,
        h5_path: str,
        dataset_path: str,
        scaling_factor: float,
        augmentation_config: AugmentationConfig | None = None,
        kind: int = 2,
    ):
        """Initialize mapping dataset.

        Args:
            h5_path: Path to H5 file
            dataset_path: Path to dataset within H5 file
            scaling_factor: Factor to scale values from H5 to sample values
            augmentation_config: Augmentation configuration
            kind: Integer value for conditioning (default: 2 for mapping)
        """
        self.dataset_path = dataset_path
        self.scaling_factor = scaling_factor
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count number of slices in H5 file.

        Args:
            h5_file: Open H5 file handle

        Returns:
            Total number of slices
        """
        if self.dataset_path not in h5_file:
            raise ValueError(
                f"Dataset '{self.dataset_path}' not found in {self.h5_path}"
            )
        data = h5_file[self.dataset_path]
        # Shape is (z, y, x), so num_slices = z
        return data.shape[0]

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get shape of a single slice.

        Args:
            h5_file: Open H5 file handle

        Returns:
            Shape tuple for a single slice (y, x)
        """
        data = h5_file[self.dataset_path]
        # Shape is (z, y, x), slice shape is (y, x)
        return data.shape[1], data.shape[2]

    def _load_slice(self, h5_file: h5py.File, idx: int) -> np.ndarray:
        """Load a single slice from H5 file and apply scaling.

        Args:
            h5_file: Open H5 file handle
            idx: Slice index

        Returns:
            Scaled slice data as numpy array of shape (y, x)
        """
        data = h5_file[self.dataset_path]
        slice_data = data[idx, :, :]
        # Apply scaling factor
        return slice_data * self.scaling_factor

