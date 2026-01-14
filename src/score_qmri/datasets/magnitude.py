"""Magnitude image dataset for MRI data.

Handles H5 files with flat subject datasets at root level:
- <subject_001>: (N_z, H, W) float32
- <subject_002>: (N_z, H, W) float32
"""

from .config import AugmentationConfig
from .flat_h5 import FlatH5Dataset
from .kind import DatasetKind


class MagnitudeImageDataset(FlatH5Dataset):
    """Dataset for magnitude MRI images.

    Expects H5 files with subject datasets at root level (no groups).
    Each dataset has shape (N_z, H, W) float32.
    """

    def __init__(
        self,
        h5_path: str,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.T1W_MAGNITUDE,
        scaling_factor: float = 1.0,
        penumbra_width: int = 30,
    ) -> None:
        super().__init__(h5_path, augmentation_config, kind, scaling_factor, penumbra_width)
