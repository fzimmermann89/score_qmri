"""Tests for BaseMRIDataset."""


import h5py
import numpy as np
import torch

from datasets.base import BaseMRIDataset
from datasets.kind import DatasetKind


class DummyDataset(BaseMRIDataset):
    """Dummy dataset for testing base class. Expects flat root datasets."""

    def _count_slices_in_file(self, h5_file):
        keys = [k for k in h5_file if isinstance(h5_file[k], h5py.Dataset)]
        return sum(h5_file[k].shape[0] for k in keys)

    def _get_slice_shape(self, h5_file):
        first_key = next(k for k in h5_file if isinstance(h5_file[k], h5py.Dataset))
        return h5_file[first_key].shape[1], h5_file[first_key].shape[2]

    def _load_slice(self, h5_file, idx):
        # Simple sequential access
        keys = [k for k in h5_file if isinstance(h5_file[k], h5py.Dataset)]
        offset = 0
        for k in keys:
            n_slices = h5_file[k].shape[0]
            if idx < offset + n_slices:
                return torch.from_numpy(h5_file[k][idx - offset, :, :].astype(np.float32))
            offset += n_slices
        raise IndexError(f"Index {idx} out of bounds")


def test_base_dataset_init(sample_h5_file):
    """Test BaseMRIDataset initialization counts slices correctly."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.T1W_MAGNITUDE)
    # sample_h5_file: subject_001 has 5 slices, subject_002 has 3 slices = 8
    assert len(dataset) == 8
    assert dataset.slice_shape == (64, 64)
    assert dataset.kind == DatasetKind.T1W_MAGNITUDE


def test_base_dataset_getitem_returns_correct_keys(sample_h5_file):
    """Test __getitem__ returns dict with expected keys and types."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.T1W_MAGNITUDE)
    sample = dataset[0]

    assert set(sample.keys()) == {"image", "mask", "kind"}
    assert sample["image"].shape == (64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["mask"].dtype == torch.bool
    assert sample["kind"].item() == DatasetKind.T1W_MAGNITUDE.value


def test_base_dataset_mask_is_nonzero_pixels(sample_h5_file):
    """Test mask correctly identifies non-zero pixels."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.T1W_MAGNITUDE)
    sample = dataset[0]

    image, mask = sample["image"], sample["mask"]
    assert torch.all((image != 0) == mask)


def test_base_dataset_h5_caching(sample_h5_file):
    """Test H5 file handle is cached after first access."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.T1W_MAGNITUDE)

    # Initially no file handle
    assert dataset._h5_file is None

    # Access triggers lazy loading
    _ = dataset[0]
    first_handle = dataset._h5_file

    # Second access reuses same handle
    _ = dataset[1]
    assert dataset._h5_file is first_handle
