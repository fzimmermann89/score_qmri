"""Tests for ComplexImageDataset."""

import h5py
import pytest
import torch

from datasets.complex import ComplexImageDataset
from datasets.kind import DatasetKind


class TestComplexDataset:
    """Tests for complex data with grouped format."""

    def test_combined_mode_counts_all_slices(self, sample_h5_file_complex):
        """Test combined mode counts slices across all subjects."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="combined",
            kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
        )
        # Subject 1: 3 slices, Subject 2: 2 slices = 5 total
        assert len(dataset) == 5
        assert dataset.slice_shape == (64, 64)

    def test_coilwise_mode(self, sample_h5_file_complex):
        """Test coilwise mode iterates through all coil combinations."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="coilwise",
            kind=DatasetKind.COMPLEX_T1W_COILWISE,
        )
        # Subject 1: 3 z-slices * 4 coils = 12, Subject 2: 2 z-slices * 4 coils = 8 = 20 total
        assert len(dataset) == 20
        assert dataset.slice_shape == (64, 64)

    def test_getitem_returns_complex_tensor(self, sample_h5_file_complex):
        """Test __getitem__ returns complex tensor."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="combined",
            kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
        )
        sample = dataset[0]
        assert sample["image"].shape == (64, 64)
        assert sample["image"].is_complex()
        assert sample["kind"].item() == DatasetKind.COMPLEX_T1W_COIL_COMBINED.value

    def test_coilwise_returns_complex_tensor(self, sample_h5_file_complex):
        """Test coilwise __getitem__ returns complex tensor."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="coilwise",
            kind=DatasetKind.COMPLEX_T1W_COILWISE,
        )
        sample = dataset[0]
        assert sample["image"].shape == (64, 64)
        assert sample["image"].is_complex()

    def test_coilwise_access_all_coils(self, sample_h5_file_complex):
        """Test that coilwise mode iterates through all coil combinations."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="coilwise",
            kind=DatasetKind.COMPLEX_T1W_COILWISE,
        )
        # Should access all (z, coil) combinations
        assert len(dataset) == 20  # 3*4 + 2*4
        # Each access should return deterministic data
        sample1 = dataset[0]
        sample2 = dataset[0]
        assert torch.allclose(sample1["image"], sample2["image"])


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_group_raises(self, tmp_path):
        """Test error when group missing from H5."""
        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("other", data=[1, 2, 3])

        with pytest.raises(ValueError, match="not found"):
            ComplexImageDataset(str(h5_path), mode="combined", kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED)

    def test_dataset_instead_of_group_raises(self, tmp_path):
        """Test error when mode points to dataset instead of group."""
        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            # Create 'combined' as a dataset, not a group
            f.create_dataset("combined", data=[[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="must be a group"):
            ComplexImageDataset(str(h5_path), mode="combined", kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED)
