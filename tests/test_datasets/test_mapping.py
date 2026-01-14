"""Tests for MappingDataset."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.mapping import MappingDataset


def test_mapping_dataset_counts_all_subjects(sample_h5_file_mapping: str) -> None:
    """Test MappingDataset counts slices across all subjects."""
    dataset = MappingDataset(sample_h5_file_mapping, scaling_factor=2000.0, kind=DatasetKind.T1MAP)
    # Subject 1: 4 slices, Subject 2: 3 slices = 7 total
    assert len(dataset) == 7
    assert dataset.slice_shape == (64, 64)


def test_mapping_getitem_returns_correct_shape(sample_h5_file_mapping: str) -> None:
    """Test __getitem__ returns correct shape and type."""
    dataset = MappingDataset(sample_h5_file_mapping, scaling_factor=2000.0, kind=DatasetKind.T1MAP)
    sample = dataset[0]
    assert sample["image"].shape == (64, 64)
    assert sample["image"].dtype.is_floating_point
    assert sample["kind"].item() == DatasetKind.T1MAP.value


def test_mapping_iterates_all_slices(sample_h5_file_mapping: str) -> None:
    """Test we can iterate through all slices."""
    dataset = MappingDataset(sample_h5_file_mapping, scaling_factor=2000.0, kind=DatasetKind.T1MAP)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample["image"].shape == (64, 64)


def test_scaling_factor_applied(sample_h5_file_mapping: str) -> None:
    """Test scaling factor is correctly applied to mapping data."""
    # Use 1/2000.0 to divide values to ~[0.25, 1.0] range
    dataset = MappingDataset(sample_h5_file_mapping, scaling_factor=1.0 / 2000.0, kind=DatasetKind.T1MAP)
    sample = dataset[0]

    # Verify values are in expected range (~[0, 1] after scaling)
    # Original data is 500-2000, after dividing by 2000 should be ~[0.25, 1.0]
    assert sample["image"].min() >= 0.0
    assert sample["image"].max() <= 1.1


def test_scaling_factor_zero(sample_h5_file_mapping: str, tmp_path: Path) -> None:
    """Test scaling factor of zero returns zero values."""
    # Create a temporary H5 file with known values
    with h5py.File(str(tmp_path / "test_zero_scale.h5"), "w") as f:
        f.create_dataset("subject_001", data=np.ones((2, 64, 64), dtype=np.float32) * 1000.0)

    dataset = MappingDataset(str(tmp_path / "test_zero_scale.h5"), scaling_factor=0.0, kind=DatasetKind.T1MAP)
    sample = dataset[0]

    # Values should be zero when scaling_factor=0.0
    assert sample["image"].max() == 0.0
    assert sample["image"].min() == 0.0


def test_scaling_factor_negative(sample_h5_file_mapping: str, tmp_path: Path) -> None:
    """Test negative scaling factor inverts values."""
    # Create a temporary H5 file with known values
    with h5py.File(str(tmp_path / "test_neg_scale.h5"), "w") as f:
        f.create_dataset("subject_001", data=np.ones((2, 64, 64), dtype=np.float32) * 1000.0)

    dataset = MappingDataset(str(tmp_path / "test_neg_scale.h5"), scaling_factor=-0.001, kind=DatasetKind.T1MAP)
    sample = dataset[0]

    # Values should be negative and large magnitude
    assert sample["image"].max() < 0


def test_mapping_empty_file_raises(tmp_path: Path) -> None:
    """Test error when H5 file has no datasets."""
    with h5py.File(str(tmp_path / "empty.h5"), "w") as f:
        # Create a group but no datasets
        f.create_group("some_group")

    with pytest.raises(ValueError, match="No datasets found"):
        MappingDataset(str(tmp_path / "empty.h5"), scaling_factor=2000.0, kind=DatasetKind.T1MAP)


def test_mapping_mask_is_nonzero_pixels(sample_h5_file_mapping: str) -> None:
    """Test background_mask correctly identifies zero pixels (inverse of signal)."""
    dataset = MappingDataset(sample_h5_file_mapping, scaling_factor=2000.0, kind=DatasetKind.T1MAP)
    sample = dataset[0]

    image, background_mask = sample["image"], sample["background_mask"]
    signal_mask = ~background_mask
    assert torch.all((image > 1e-6) == signal_mask)


def test_mapping_different_kinds(sample_h5_file_mapping: str) -> None:
    """Test MappingDataset works with different DatasetKind values."""
    for kind in [DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.T2STARMAP]:
        dataset = MappingDataset(sample_h5_file_mapping, scaling_factor=2000.0, kind=kind)
        assert dataset.kind == kind
        sample = dataset[0]
        assert sample["kind"].item() == kind.value
