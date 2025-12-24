"""Complex image dataset for combined or coilwise MRI data."""

from typing import Literal

import h5py
import numpy as np

from .base import BaseMRIDataset
from .config import AugmentationConfig


class ComplexImageDataset(BaseMRIDataset):
    """Dataset for complex MRI images (combined or coilwise)."""

    def __init__(
        self,
        h5_path: str,
        mode: Literal["combined", "coilwise"],
        augmentation_config: AugmentationConfig | None = None,
        kind: int = 0,
    ):
        """Initialize complex image dataset.

        Args:
            h5_path: Path to H5 file
            mode: "combined" for (z, y, x) or "coilwise" for (coil, z, y, x)
            augmentation_config: Augmentation configuration
            kind: Integer value for conditioning (default: 0 for complex images)
        """
        self.mode = mode
        super().__init__(h5_path, augmentation_config, kind)

    def _count_slices_in_file(self, h5_file: h5py.File) -> int:
        """Count number of slices in H5 file.

        Args:
            h5_file: Open H5 file handle

        Returns:
            Total number of slices
        """
        if self.mode == "combined":
            if "combined" not in h5_file:
                raise ValueError(
                    f"Dataset 'combined' not found in {self.h5_path}"
                )
            data = h5_file["combined"]
            # Shape is (z, y, x), so num_slices = z
            return data.shape[0]
        else:  # coilwise
            if "coilwise" not in h5_file:
                raise ValueError(
                    f"Dataset 'coilwise' not found in {self.h5_path}"
                )
            data = h5_file["coilwise"]
            # Shape is (coil, z, y, x)
            # Total slices = num_coils * num_z_slices
            num_coils, num_z_slices = data.shape[0], data.shape[1]
            return num_coils * num_z_slices

    def _get_slice_shape(self, h5_file: h5py.File) -> tuple[int, ...]:
        """Get shape of a single slice.

        Args:
            h5_file: Open H5 file handle

        Returns:
            Shape tuple for a single slice (y, x)
        """
        if self.mode == "combined":
            data = h5_file["combined"]
            # Shape is (z, y, x), slice shape is (y, x)
            return data.shape[1], data.shape[2]
        else:  # coilwise
            data = h5_file["coilwise"]
            # Shape is (coil, z, y, x), slice shape is (y, x)
            return data.shape[2], data.shape[3]

    def _load_slice(self, h5_file: h5py.File, idx: int) -> np.ndarray:
        """Load a single slice from H5 file.

        Args:
            h5_file: Open H5 file handle
            idx: Slice index

        Returns:
            Slice data as numpy array of shape (y, x)
        """
        if self.mode == "combined":
            data = h5_file["combined"]
            # Direct slice access: data[z_idx, :, :]
            return data[idx, :, :]
        else:  # coilwise
            data = h5_file["coilwise"]
            # Sequential coil selection: round-robin
            # idx maps to (coil_idx, z_idx) where coil_idx = idx % num_coils
            num_coils = data.shape[0]
            num_z_slices = data.shape[1]

            coil_idx = idx % num_coils
            z_idx = idx // num_coils

            # Ensure z_idx is within bounds
            if z_idx >= num_z_slices:
                z_idx = z_idx % num_z_slices

            return data[coil_idx, z_idx, :, :]

