"""Tests for MagnitudeImageDataset."""

import tempfile

import h5py
import pytest

from datasets.kind import DatasetKind
from datasets.magnitude import MagnitudeImageDataset


def test_magnitude_dataset_counts_all_subjects(sample_h5_file_magnitude):
    """Test MagnitudeImageDataset counts slices across all subjects."""
    dataset = MagnitudeImageDataset(sample_h5_file_magnitude, kind=DatasetKind.T1W_MAGNITUDE)
    # Subject 1: 5 slices, Subject 2: 3 slices = 8 total
    assert len(dataset) == 8
    assert dataset.slice_shape == (64, 64)


def test_magnitude_getitem_returns_correct_shape(sample_h5_file_magnitude):
    """Test __getitem__ returns correct shape and type."""
    dataset = MagnitudeImageDataset(sample_h5_file_magnitude, kind=DatasetKind.T1W_MAGNITUDE)
    sample = dataset[0]
    assert sample["image"].shape == (64, 64)
    assert sample["image"].dtype.is_floating_point
    assert sample["kind"].item() == DatasetKind.T1W_MAGNITUDE.value


def test_magnitude_iterates_all_slices(sample_h5_file_magnitude):
    """Test we can iterate through all slices."""
    dataset = MagnitudeImageDataset(sample_h5_file_magnitude, kind=DatasetKind.T1W_MAGNITUDE)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["image"].shape == (64, 64)


def test_magnitude_empty_file_raises():
    """Test error when H5 file has no datasets."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            # Create a group but no datasets
            f.create_group("some_group")

        with pytest.raises(ValueError, match="No datasets found"):
            MagnitudeImageDataset(temp_path, kind=DatasetKind.T1W_MAGNITUDE)
    finally:
        import os

        os.unlink(temp_path)
