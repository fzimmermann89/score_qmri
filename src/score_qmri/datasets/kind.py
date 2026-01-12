"""Dataset kind enumeration for MRI data types."""

from enum import IntEnum


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

NUM_DATASET_KINDS: int = len(DatasetKind)


def is_complex_kind(kind_index: int) -> bool:
    return kind_index in _COMPLEX_KINDS
