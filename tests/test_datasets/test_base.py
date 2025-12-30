"""Tests for BaseMRIDataset."""

import torch

from datasets.base import BaseMRIDataset
from datasets.kind import DatasetKind


class DummyDataset(BaseMRIDataset):
    """Dummy dataset for testing base class."""

    def _count_slices_in_file(self, h5_file):
        return h5_file["combined"].shape[0]

    def _get_slice_shape(self, h5_file):
        return h5_file["combined"].shape[1], h5_file["combined"].shape[2]

    def _load_slice(self, h5_file, idx):
        return torch.from_numpy(h5_file["combined"][idx, :, :]).float()


def test_base_dataset_init(sample_h5_file):
    """Test BaseMRIDataset initialization counts slices correctly."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.COMPLEX_T1W_COILWISE)
    assert len(dataset) == 5
    assert dataset.slice_shape == (64, 64)
    assert dataset.kind == DatasetKind.COMPLEX_T1W_COILWISE


def test_base_dataset_getitem_returns_correct_keys(sample_h5_file):
    """Test __getitem__ returns dict with expected keys and types."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED)
    sample = dataset[0]

    assert set(sample.keys()) == {"image", "mask", "kind"}
    assert sample["image"].shape == (64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["mask"].dtype == torch.bool
    assert sample["kind"].item() == DatasetKind.COMPLEX_T1W_COIL_COMBINED.value


def test_base_dataset_mask_is_nonzero_pixels(sample_h5_file):
    """Test mask correctly identifies non-zero pixels."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.COMPLEX_T1W_COILWISE)
    sample = dataset[0]

    image, mask = sample["image"], sample["mask"]
    assert torch.all((image != 0) == mask)


def test_base_dataset_h5_caching(sample_h5_file):
    """Test H5 file handle is cached after first access."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.COMPLEX_T1W_COILWISE)

    # Initially no file handle
    assert dataset._h5_file is None

    # Access triggers lazy loading
    _ = dataset[0]
    first_handle = dataset._h5_file

    # Second access reuses same handle
    _ = dataset[1]
    assert dataset._h5_file is first_handle
