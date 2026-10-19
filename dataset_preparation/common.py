"""Small, CPU-only I/O helpers shared by standalone preparation recipes."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np


@contextmanager
def exclusive_h5(path: str | Path) -> Iterator[h5py.File]:
    """Create a new HDF5 file without replacing files or following symlinks.

    A failed conversion leaves its new, incomplete output for inspection. It
    never removes an output or attempts to resume one automatically.
    """
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


def selected_paths(paths: Iterable[str | Path], subjects: Iterable[str] | None = None, limit: int | None = None) -> list[Path]:
    """Select deterministically by exact filename or stem, then apply a limit."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = sorted({Path(path) for path in paths})
    if subjects is not None:
        wanted = set(subjects)
        selected = [path for path in selected if path.stem in wanted or path.name in wanted]
    return selected if limit is None else selected[:limit]
