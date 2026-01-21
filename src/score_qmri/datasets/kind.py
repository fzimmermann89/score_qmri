"""Dataset kind enumeration for MRI data types."""

from enum import IntEnum
from typing import overload

import torch
from torch import Tensor


class DatasetKind(IntEnum):
    """Enumeration of different MRI dataset types for conditioning."""

    COMPLEX_T1W_COILWISE = 0
    COMPLEX_T1W_COIL_COMBINED = 1
    COMPLEX_T2W_COILWISE = 2
    COMPLEX_T2W_COIL_COMBINED = 3
    COMPLEX_FLAIR_COILWISE = 4
    COMPLEX_FLAIR_COIL_COMBINED = 5
    T1MAP = 6
    T2MAP = 7
    T2STARMAP = 8
    T1W_MAGNITUDE = 9
    T2W_MAGNITUDE = 10
    FLAIR_MAGNITUDE = 11
    M0_COMPLEX = 12
    M0_MAGNITUDE = 13

    @property
    def is_complex(self) -> bool:
        """Check if this kind represents complex-valued data."""
        return self in _COMPLEX_KINDS


_COMPLEX_KINDS = frozenset(
    {
        DatasetKind.COMPLEX_T1W_COILWISE,
        DatasetKind.COMPLEX_T1W_COIL_COMBINED,
        DatasetKind.COMPLEX_T2W_COILWISE,
        DatasetKind.COMPLEX_T2W_COIL_COMBINED,
        DatasetKind.COMPLEX_FLAIR_COILWISE,
        DatasetKind.COMPLEX_FLAIR_COIL_COMBINED,
        DatasetKind.M0_COMPLEX,
    }
)

_MAGNITUDE_KINDS = frozenset(
    {
        DatasetKind.T1W_MAGNITUDE,
        DatasetKind.T2W_MAGNITUDE,
        DatasetKind.FLAIR_MAGNITUDE,
        DatasetKind.M0_MAGNITUDE,
    }
)

_MAPPING_KINDS = frozenset(
    {
        DatasetKind.T1MAP,
        DatasetKind.T2MAP,
        DatasetKind.T2STARMAP,
    }
)

NUM_DATASET_KINDS: int = len(DatasetKind)


# Vectorized lookup table (CPU): kind_index -> is_complex.
COMPLEX_KIND_LUT: Tensor = torch.tensor(
    [i in _COMPLEX_KINDS for i in range(NUM_DATASET_KINDS)],
    dtype=torch.bool,
)


@overload
def is_complex_kind(kind_index: int) -> bool: ...


@overload
def is_complex_kind(kind_index: Tensor) -> Tensor: ...


def is_complex_kind(kind_index: int | Tensor) -> bool | Tensor:
    """Check whether dataset kind(s) represent complex-valued data.

    For tensor inputs, this is a vectorized lookup without per-element `.item()`
    calls.
    """
    if isinstance(kind_index, int):
        if not (0 <= kind_index < NUM_DATASET_KINDS):
            raise IndexError(f"kind_index {kind_index} out of range [0, {NUM_DATASET_KINDS})")
        return bool(COMPLEX_KIND_LUT[kind_index].item())
    return COMPLEX_KIND_LUT.to(device=kind_index.device)[kind_index]


def is_magnitude_kind(kind_index: int) -> bool:
    return kind_index in _MAGNITUDE_KINDS


def is_mapping_kind(kind_index: int) -> bool:
    return kind_index in _MAPPING_KINDS
