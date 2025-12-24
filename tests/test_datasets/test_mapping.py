"""Tests for MappingDataset."""

import h5py
import numpy as np

from datasets.mapping import MappingDataset


def test_mapping_dataset_applies_scaling(sample_h5_file):
    """Test MappingDataset applies scaling factor correctly."""
    scaling_factor = 2.5
    dataset = MappingDataset(
        sample_h5_file, dataset_path="mapping", scaling_factor=scaling_factor, kind=2
    )

    assert len(dataset) == 5
    assert dataset.slice_shape == (64, 64)

    sample = dataset[0]
    assert sample["image"].shape == (64, 64)
    assert sample["kind"].item() == 2


def test_mapping_scaling_factor_applied_to_data(sample_h5_file):
    """Test that scaling factor is applied to slice data."""
    with h5py.File(sample_h5_file, "r") as f:
        original = f["mapping"][0, :, :]

    scaling_factor = 3.0
    dataset = MappingDataset(
        sample_h5_file, dataset_path="mapping", scaling_factor=scaling_factor, kind=2
    )

    sample = dataset[0]
    expected = original * scaling_factor
    np.testing.assert_allclose(sample["image"].numpy(), expected, rtol=1e-5)
