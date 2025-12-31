"""Tests for MappingDataset."""

import tempfile

import h5py
import pytest

from datasets.kind import DatasetKind
from datasets.mapping import MappingDataset


def test_mapping_dataset_counts_all_subjects(sample_h5_file_mapping):
    """Test MappingDataset counts slices across all subjects."""
    dataset = MappingDataset(
        sample_h5_file_mapping,
        scaling_factor=1 / 3000,
        kind=DatasetKind.T1MAP,
    )
    # Subject 1: 4 slices, Subject 2: 3 slices = 7 total
    assert len(dataset) == 7
    assert dataset.slice_shape == (64, 64)


def test_mapping_applies_scaling(sample_h5_file_mapping):
    """Test scaling factor is applied correctly."""
    scaling_factor = 1 / 3000
    dataset = MappingDataset(
        sample_h5_file_mapping,
        scaling_factor=scaling_factor,
        kind=DatasetKind.T1MAP,
    )
    sample = dataset[0]

    # Values should be scaled to ~[0, 1] range
    # Original values are uniform(500, 2000), scaled by 1/3000 gives ~[0.17, 0.67]
    assert sample["image"].min() >= 0
    assert sample["image"].max() <= 1
    assert sample["kind"].item() == DatasetKind.T1MAP.value


def test_mapping_getitem_shape(sample_h5_file_mapping):
    """Test __getitem__ returns correct shape."""
    dataset = MappingDataset(
        sample_h5_file_mapping,
        scaling_factor=0.001,
        kind=DatasetKind.T2MAP,
    )
    sample = dataset[0]
    assert sample["image"].shape == (64, 64)
    assert sample["image"].dtype.is_floating_point


def test_mapping_empty_file_raises():
    """Test error when H5 file has no datasets."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            f.create_group("empty_group")

        with pytest.raises(ValueError, match="No datasets found"):
            MappingDataset(temp_path, scaling_factor=1.0, kind=DatasetKind.T1MAP)
    finally:
        import os

        os.unlink(temp_path)
