"""Quantitative mapping dataset for MRI data.

Handles H5 files with flat subject datasets at root level:
- <subject_001>: (N_z, H, W) float32
- <subject_002>: (N_z, H, W) float32

Applies scaling factor to normalize quantitative values (e.g., T1 in ms) to ~[0, 1].
"""

from .config import AugmentationConfig
from .flat_h5 import FlatH5Dataset
from .kind import DatasetKind


class MappingDataset(FlatH5Dataset):
    """Dataset for quantitative MRI parameter maps (T1, T2, etc.).

    Expects H5 files with subject datasets at root level (no groups).
    Each dataset has shape (N_z, H, W) float32.

    Applies a scaling factor to normalize values to ~[0, 1] range.
    For example, T1 values in ms (0-3000) can be scaled by 1/3000.
    """

    def __init__(
        self,
        h5_path: str,
        scaling_factor: float = 1.0,
        augmentation_config: AugmentationConfig | None = None,
        kind: DatasetKind = DatasetKind.T1MAP,
        penumbra_width: int = -1,
    ) -> None:
        super().__init__(h5_path, augmentation_config, kind, scaling_factor, penumbra_width)
