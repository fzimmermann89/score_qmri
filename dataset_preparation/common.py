"""Shared HDF5 creation and compression helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np


@contextmanager
def exclusive_h5(path: str | Path) -> Iterator[h5py.File]:
    """Refuse existing outputs, including dangling symlinks."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as handle:
        yield handle


def axial_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Use the historical cleanup layout for image volumes."""
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"Expected a nonempty array, got shape {shape}")
    if len(shape) == 4:
        return (1, 1, *shape[2:])
    return (1, *shape[1:])


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    """Write an unchanged array using axial LZF compression and shuffle."""
    array = np.asarray(array)
    return handle.create_dataset(name, data=array, chunks=axial_chunks(array.shape), compression="lzf", shuffle=True)
