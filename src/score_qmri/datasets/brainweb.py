"""Brainweb quantitative dataset for T1, T2 maps and complex M0.

Reads Brainweb HDF5 class volumes and produces quantitative parameter maps
with the same output format as other MRI datasets for diffusion training.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .augmentation import AugmentationPipeline
from .base import _compute_penumbra
from .config import AugmentationConfig
from .kind import DatasetKind

# Tissue class names (excluding background)
BRAINWEB_CLASSES = ("skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves")


@dataclass
class BrainwebTissue:
    """Container for tissue-specific MRI parameters.

    Each parameter can be a single value or a (min, max) tuple for random sampling.
    T1/T2 are in seconds. M0 magnitude is relative (0-1), phase in radians.
    """

    t1: float | tuple[float, float]
    """T1 relaxation time in seconds, or (min, max) range."""

    t2: float | tuple[float, float]
    """T2 relaxation time in seconds, or (min, max) range."""

    m0_abs: float | tuple[float, float]
    """M0 magnitude (0-1 relative), or (min, max) range."""

    m0_phase: float | tuple[float, float] = 0.0
    """M0 phase in radians, or (min, max) range."""

    def sample_r1(self, rng: torch.Generator | None = None) -> Tensor:
        """Sample R1 = 1/T1 value."""
        if isinstance(self.t1, tuple):
            t1 = torch.empty(1).uniform_(*self.t1, generator=rng)
            return 1.0 / t1
        return torch.tensor([1.0 / self.t1])

    def sample_r2(self, rng: torch.Generator | None = None) -> Tensor:
        """Sample R2 = 1/T2 value."""
        if isinstance(self.t2, tuple):
            t2 = torch.empty(1).uniform_(*self.t2, generator=rng)
            return 1.0 / t2
        return torch.tensor([1.0 / self.t2])

    def sample_m0(self, rng: torch.Generator | None = None) -> Tensor:
        """Sample complex M0 value."""
        magnitude = torch.empty(1).uniform_(*self.m0_abs, generator=rng) if isinstance(self.m0_abs, tuple) else torch.tensor([self.m0_abs])
        phase = (
            torch.empty(1).uniform_(*self.m0_phase, generator=rng) if isinstance(self.m0_phase, tuple) else torch.tensor([self.m0_phase])
        )

        return torch.polar(magnitude, phase)


# Default tissue parameters at 3T with randomization ranges
VALUES_3T_RANDOMIZED: Mapping[str, BrainwebTissue] = MappingProxyType(
    {
        "skl": BrainwebTissue((0.001, 2.000), (0.001, 0.010), (0.00, 0.05), (-0.2, 0.2)),
        "gry": BrainwebTissue((1.200, 2.000), (0.080, 0.120), (0.70, 1.00), (-0.2, 0.2)),
        "wht": BrainwebTissue((0.800, 1.500), (0.060, 0.100), (0.50, 0.90), (-0.2, 0.2)),
        "csf": BrainwebTissue((2.000, 4.000), (1.300, 2.000), (0.90, 1.00), (-0.2, 0.2)),
        "mrw": BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.70, 1.00), (-0.2, 0.2)),
        "dura": BrainwebTissue((2.000, 2.800), (0.200, 0.500), (0.90, 1.00), (-0.2, 0.2)),
        "fat": BrainwebTissue((0.300, 0.500), (0.060, 0.100), (0.90, 1.00), (-0.2, 0.2)),
        "fat2": BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.60, 0.90), (-0.2, 0.2)),
        "mus": BrainwebTissue((1.200, 1.500), (0.040, 0.060), (0.90, 1.00), (-0.2, 0.2)),
        "m-s": BrainwebTissue((0.500, 0.900), (0.300, 0.500), (0.90, 1.00), (-0.2, 0.2)),
        "ves": BrainwebTissue((1.700, 2.100), (0.200, 0.400), (0.80, 1.00), (-0.2, 0.2)),
    }
)


class BrainwebQuantitativeSampler(Dataset[dict[str, Any]]):
    """Dataset for Brainweb quantitative maps (T1, T2) and complex M0.

    Reads tissue class volumes from Brainweb HDF5 and synthesizes quantitative
    parameter maps by weighted combination of per-tissue values.

    Only axial slices are supported.
    """

    def __init__(
        self,
        h5_path: str,
        kind: DatasetKind,
        parameters: Mapping[str, BrainwebTissue] = VALUES_3T_RANDOMIZED,
        augmentation_config: AugmentationConfig | None = None,
        scaling_factor: float = 1.0,
        step: int = 1,
        skip_slices: tuple[int, int] = (80, 80),
        seed: int | Literal["index", "random"] = "random",
        penumbra_width: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        h5_path
            Path to Brainweb HDF5 file with `classes` dataset.
        kind
            Output kind: T1MAP, T2MAP, or M0_COMPLEX.
        parameters
            Tissue parameter mapping for T1/T2/M0 sampling.
        augmentation_config
            Spatial augmentation configuration.
        scaling_factor
            Factor to multiply output values by (for normalization).
        step
            Step size between slices.
        skip_slices
            (skip_start, skip_end) for axial slices.
        seed
            Random seed: int for fixed, "random" for default generator,
            "index" for per-slice deterministic seed.
        penumbra_width
            Penumbra dilation: >0 for ring, 0 for none, <0 for full background.
        """
        if kind not in (DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.M0_COMPLEX):
            raise ValueError(f"kind must be T1MAP, T2MAP, or M0_COMPLEX, got {kind}")

        self.h5_path = str(Path(h5_path).resolve())
        self.kind = kind
        self.parameters = parameters
        self.scaling_factor = scaling_factor
        self.step = step
        self.skip_slices = skip_slices
        self.penumbra_width = penumbra_width
        self.augmentation = AugmentationPipeline(augmentation_config) if augmentation_config else None

        # Set up RNG
        if seed == "random":
            self._rng: torch.Generator | None = torch.default_generator
        elif isinstance(seed, int):
            self._rng = torch.Generator().manual_seed(seed)
        else:  # "index"
            self._rng = None

        self._h5_file: h5py.File | None = None

        # Count slices
        with h5py.File(self.h5_path, "r") as f:
            if "classes" not in f:
                raise ValueError(f"Dataset 'classes' not found in {self.h5_path}")
            dataset = f["classes"]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"'classes' must be a dataset in {self.h5_path}")
            classes_shape = dataset.shape  # (Z, Y, X, C)
            raw_attr = f.attrs.get("classnames", None)
            if raw_attr is None:
                raise ValueError(f"Attribute 'classnames' not found in {self.h5_path}")
            if not isinstance(raw_attr, (list, tuple, np.ndarray)):
                raise TypeError(f"Attribute 'classnames' must be array-like in {self.h5_path}")
            raw_classnames = list(raw_attr)

        self._classnames = [c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c) for c in raw_classnames]
        missing = sorted(set(self._classnames) - set(self.parameters.keys()))
        if missing:
            raise ValueError(f"Missing tissue parameters for classes: {missing}")

        self._n_axial = classes_shape[0]
        skip_start, skip_end = skip_slices
        self._valid_slices = max(0, self._n_axial - skip_start - skip_end)
        self._skip_start = skip_start
        self._spatial_shape = (classes_shape[1], classes_shape[2])  # (Y, X)
        self._mask_kernel = torch.ones(1, 1, 3, 3)

    def _get_h5_file(self) -> h5py.File:
        """Get cached H5 file handle."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return self._valid_slices // self.step

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = len(self) + index
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        # Map index to actual slice
        slice_idx = index * self.step + self._skip_start

        # Load class weights for this slice
        f = self._get_h5_file()
        dataset = f["classes"]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"'classes' must be a dataset in {self.h5_path}")
        classes_uint8 = np.array(dataset[slice_idx])  # (Y, X, C)
        classes = torch.from_numpy(classes_uint8).float() / 255.0  # Normalize to [0, 1]

        # Set up RNG for this slice
        rng = torch.Generator().manual_seed(index) if self._rng is None else self._rng

        # Sample tissue parameters and compute weighted maps
        r1_values = torch.stack([self.parameters[c].sample_r1(rng) for c in self._classnames])  # (C, 1)
        r2_values = torch.stack([self.parameters[c].sample_r2(rng) for c in self._classnames])  # (C, 1)
        m0_values = torch.stack([self.parameters[c].sample_m0(rng) for c in self._classnames])  # (C, 1)

        # Weighted sum: classes @ values -> (Y, X, 1) -> squeeze to (Y, X)
        class_sum = classes.sum(dim=-1)
        weights = classes / (class_sum[..., None] + 1e-8)

        r1_map = (weights @ r1_values)[..., 0]  # (Y, X)
        r2_map = (weights @ r2_values)[..., 0]  # (Y, X)
        m0_map = (classes.to(m0_values.dtype) @ m0_values)[..., 0]  # (Y, X) complex

        # Convert R1/R2 to T1/T2 (seconds)
        t1_map = 1.0 / (r1_map + 1e-8)
        t2_map = 1.0 / (r2_map + 1e-8)

        # Compute mask from class weights (sum > threshold)
        signal_mask = class_sum > 0.5

        # Clean mask with 2D conv (remove isolated holes)
        mask_float = signal_mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, Y, X)
        cleaned = torch.nn.functional.conv2d(mask_float, self._mask_kernel, padding=1)
        signal_mask = (cleaned[0, 0] >= 1).bool()

        # Select output based on kind
        if self.kind == DatasetKind.T1MAP:
            image: Tensor = t1_map
        elif self.kind == DatasetKind.T2MAP:
            image = t2_map
        else:  # M0_COMPLEX
            image = m0_map

        # Apply scaling factor
        if self.scaling_factor != 1.0:
            image = image * self.scaling_factor

        # Apply augmentations
        if self.augmentation is not None:
            image, signal_mask = self.augmentation(image, signal_mask)

        if self.kind in (DatasetKind.T1MAP, DatasetKind.T2MAP):
            image = image.masked_fill(~signal_mask, 0.0)

        # Compute background and penumbra masks
        background_mask = ~signal_mask

        if self.penumbra_width < 0:
            penumbra_mask = background_mask
        elif self.penumbra_width == 0:
            penumbra_mask = torch.zeros_like(signal_mask)
        else:
            penumbra_mask = torch.from_numpy(_compute_penumbra(signal_mask.numpy(), self.penumbra_width))

        return {
            "image": image,
            "penumbra_mask": penumbra_mask,
            "background_mask": background_mask,
            "kind": torch.tensor(self.kind.value, dtype=torch.long),
            "h5_path": self.h5_path,
        }

    def __del__(self) -> None:
        if self._h5_file is not None:
            with contextlib.suppress(Exception):
                self._h5_file.close()
