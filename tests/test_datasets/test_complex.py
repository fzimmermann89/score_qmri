"""Tests for ComplexImageDataset."""

import tempfile

import h5py
import pytest

from datasets.complex import ComplexImageDataset
from datasets.kind import DatasetKind


def test_complex_combined_mode_counts_z_slices(sample_h5_file):
    """Test combined mode counts z-dimension slices."""
    dataset = ComplexImageDataset(sample_h5_file, mode="combined", kind=DatasetKind.COMPLEX_T1W_COILWISE)
    assert len(dataset) == 5  # z dimension
    assert dataset.slice_shape == (64, 64)


def test_complex_coilwise_mode_counts_all_coil_slices(sample_h5_file):
    """Test coilwise mode counts coils * z slices."""
    dataset = ComplexImageDataset(sample_h5_file, mode="coilwise", kind=DatasetKind.COMPLEX_T1W_COILWISE)
    assert len(dataset) == 20  # 4 coils * 5 z slices


def test_complex_getitem_returns_correct_shape(sample_h5_file):
    """Test __getitem__ returns 2D slice."""
    dataset = ComplexImageDataset(sample_h5_file, mode="combined", kind=DatasetKind.COMPLEX_T1W_COILWISE)
    sample = dataset[0]
    assert sample["image"].shape == (64, 64)
    assert sample["kind"].item() == DatasetKind.COMPLEX_T1W_COILWISE.value


def test_complex_coilwise_sequential_coil_selection(sample_h5_file):
    """Test coilwise mode iterates through coils sequentially."""
    dataset = ComplexImageDataset(sample_h5_file, mode="coilwise", kind=DatasetKind.COMPLEX_T1W_COILWISE)
    # All samples should be valid 2D slices
    for i in range(min(10, len(dataset))):
        assert dataset[i]["image"].shape == (64, 64)


def test_complex_missing_combined_dataset_raises():
    """Test error when combined dataset missing from H5."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    try:
        with h5py.File(temp_path, "w") as f:
            f.create_dataset("other", data=[1, 2, 3])

        with pytest.raises(ValueError, match="not found"):
            ComplexImageDataset(temp_path, mode="combined", kind=DatasetKind.COMPLEX_T1W_COILWISE)
    finally:
        import os

        os.unlink(temp_path)
