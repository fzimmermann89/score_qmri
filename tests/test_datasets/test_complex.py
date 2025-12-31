"""Tests for ComplexImageDataset."""

import tempfile

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

    def test_coilwise_random_mode(self, sample_h5_file_complex):
        """Test coilwise with random coil selection."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="coilwise",
            kind=DatasetKind.COMPLEX_T1W_COILWISE,
            random_coil=True,
        )
        # Subject 1: 3 z-slices, Subject 2: 2 z-slices = 5 total
        assert len(dataset) == 5

    def test_coilwise_sequential_mode(self, sample_h5_file_complex):
        """Test coilwise with all coils sequentially."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="coilwise",
            kind=DatasetKind.COMPLEX_T1W_COILWISE,
            random_coil=False,
        )
        # Subject 1: 3 z * 4 coils = 12, Subject 2: 2 z * 4 coils = 8 = 20 total
        assert len(dataset) == 20

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

    def test_random_coil_varies(self, sample_h5_file_complex):
        """Test that random_coil=True samples different coils."""
        dataset = ComplexImageDataset(
            sample_h5_file_complex,
            mode="coilwise",
            kind=DatasetKind.COMPLEX_T1W_COILWISE,
            random_coil=True,
        )
        # Sample the same index multiple times - coils should vary
        samples = [dataset[0]["image"] for _ in range(10)]
        # At least some should differ (probability of all same is very low)
        different_found = any(not torch.allclose(samples[0], s) for s in samples[1:])
        assert different_found, "Random coil selection should produce varying results"


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_group_raises(self):
        """Test error when group missing from H5."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, "w") as f:
                f.create_dataset("other", data=[1, 2, 3])

            with pytest.raises(ValueError, match="not found"):
                ComplexImageDataset(temp_path, mode="combined", kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED)
        finally:
            import os

            os.unlink(temp_path)

    def test_dataset_instead_of_group_raises(self):
        """Test error when mode points to dataset instead of group."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, "w") as f:
                # Create 'combined' as a dataset, not a group
                f.create_dataset("combined", data=[[1, 2], [3, 4]])

            with pytest.raises(ValueError, match="must be a group"):
                ComplexImageDataset(temp_path, mode="combined", kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED)
        finally:
            import os

            os.unlink(temp_path)
