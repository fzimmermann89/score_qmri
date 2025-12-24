"""Tests for MagnitudeImageDataset."""

import pytest

from datasets.magnitude import MagnitudeImageDataset


def test_magnitude_dataset_loads_slices(sample_h5_file):
    """Test MagnitudeImageDataset loads slices correctly."""
    dataset = MagnitudeImageDataset(sample_h5_file, dataset_path="magnitude", kind=1)
    assert len(dataset) == 5
    assert dataset.slice_shape == (64, 64)

    sample = dataset[0]
    assert sample["image"].shape == (64, 64)
    assert sample["kind"].item() == 1


def test_magnitude_missing_dataset_raises(sample_h5_file):
    """Test error when specified dataset path not in H5."""
    with pytest.raises(ValueError, match="not found"):
        MagnitudeImageDataset(sample_h5_file, dataset_path="nonexistent", kind=1)
