"""Tests for BaseMRIDataset."""

import h5py
import numpy as np
import pytest
import torch

from score_qmri.datasets.base import BaseMRIDataset
from score_qmri.datasets.kind import DatasetKind


class DummyDataset(BaseMRIDataset):
    """Dummy dataset for testing base class. Expects flat root datasets."""

    def _count_slices_in_file(self, h5_file):
        keys = [k for k in h5_file if isinstance(h5_file[k], h5py.Dataset)]
        return sum(h5_file[k].shape[0] for k in keys)

    def _get_slice_shape(self, h5_file):
        first_key = next(k for k in h5_file if isinstance(h5_file[k], h5py.Dataset))
        return h5_file[first_key].shape[1], h5_file[first_key].shape[2]

    def _load_slice(self, h5_file, idx):
        keys = [k for k in h5_file if isinstance(h5_file[k], h5py.Dataset)]
        offset = 0
        for k in keys:
            n_slices = h5_file[k].shape[0]
            if idx < offset + n_slices:
                data = h5_file[k][idx - offset, :, :]
                if np.iscomplexobj(data):
                    return torch.from_numpy(data)
                return torch.from_numpy(data.astype(np.float32))
            offset += n_slices
        raise IndexError(f"Index {idx} out of bounds")


class DummyComplexDataset(BaseMRIDataset):
    """Dummy dataset for testing complex grouped structure (combined/ group)."""

    def _count_slices_in_file(self, h5_file):
        combined = h5_file["combined"]
        return sum(combined[k].shape[0] for k in combined)

    def _get_slice_shape(self, h5_file):
        combined = h5_file["combined"]
        first_key = next(iter(combined))
        return combined[first_key].shape[1], combined[first_key].shape[2]

    def _load_slice(self, h5_file, idx):
        combined = h5_file["combined"]
        offset = 0
        for k in combined:
            n_slices = combined[k].shape[0]
            if idx < offset + n_slices:
                data = combined[k][idx - offset, :, :]
                return torch.from_numpy(data)
            offset += n_slices
        raise IndexError(f"Index {idx} out of bounds")


def test_base_dataset_magnitude_init(sample_h5_file):
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

    assert set(sample.keys()) == {"image", "penumbra_mask", "background_mask", "kind", "h5_path"}
    assert sample["image"].shape == (64, 64)
    assert sample["background_mask"].shape == (64, 64)
    assert sample["background_mask"].dtype == torch.bool
    assert sample["kind"].item() == DatasetKind.T1W_MAGNITUDE.value


def test_base_dataset_mask_is_nonzero_pixels(sample_h5_file):
    """Test background_mask correctly identifies zero pixels (inverse of signal)."""
    dataset = DummyDataset(sample_h5_file, kind=DatasetKind.T1W_MAGNITUDE)
    sample = dataset[0]

    image, background_mask = sample["image"], sample["background_mask"]
    signal_mask = ~background_mask
    assert torch.all((image.abs() > 1e-6) == signal_mask)


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


def test_dtype_kind_mismatch_complex_to_magnitude(sample_h5_file_complex):
    """Complex H5 data with magnitude kind raises ValueError."""
    dataset = DummyComplexDataset(sample_h5_file_complex, kind=DatasetKind.T1W_MAGNITUDE)
    with pytest.raises(ValueError, match="Kind/dtype mismatch"):
        _ = dataset[0]


def test_dtype_kind_mismatch_magnitude_to_complex(sample_h5_file_magnitude):
    """Magnitude H5 data with complex kind raises ValueError."""
    dataset = DummyDataset(sample_h5_file_magnitude, kind=DatasetKind.COMPLEX_T1W_COILWISE)
    with pytest.raises(ValueError, match="Kind/dtype mismatch"):
        _ = dataset[0]
