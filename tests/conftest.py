"""Pytest fixtures and shared test utilities."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from datasets.config import AugmentationConfig


@pytest.fixture
def sample_h5_file():
    """Create temporary H5 file with test data."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name

    # Create H5 file with test data
    with h5py.File(h5_path, "w") as f:
        # Combined dataset: (z=5, y=64, x=64)
        combined_data = np.random.randn(5, 64, 64).astype(np.float32)
        f.create_dataset("combined", data=combined_data)

        # Coilwise dataset: (coil=4, z=5, y=64, x=64)
        coilwise_data = np.random.randn(4, 5, 64, 64).astype(np.float32)
        f.create_dataset("coilwise", data=coilwise_data)

        # Magnitude dataset: (z=5, y=64, x=64)
        magnitude_data = np.random.randn(5, 64, 64).astype(np.float32)
        f.create_dataset("magnitude", data=magnitude_data)

        # Mapping dataset: (z=5, y=64, x=64)
        mapping_data = np.random.randn(5, 64, 64).astype(np.float32)
        f.create_dataset("mapping", data=mapping_data)

    yield h5_path

    # Cleanup
    Path(h5_path).unlink(missing_ok=True)


@pytest.fixture
def sample_image_data():
    """Generate sample image array."""
    return np.random.randn(64, 64).astype(np.float32)


@pytest.fixture
def sample_complex_data():
    """Generate sample complex data."""
    real = np.random.randn(64, 64).astype(np.float32)
    imag = np.random.randn(64, 64).astype(np.float32)
    return real + 1j * imag


@pytest.fixture
def sample_magnitude_data():
    """Generate sample magnitude data."""
    return np.abs(np.random.randn(64, 64).astype(np.float32))


@pytest.fixture
def sample_mapping_data():
    """Generate sample mapping data."""
    return np.random.randn(64, 64).astype(np.float32)


@pytest.fixture
def augmentation_config():
    """Create augmentation config for testing."""
    return AugmentationConfig(
        flip_prob=0.5,
        geometric_prob=0.3,
        contrast_prob=0.3,
        scale_range=(0.9, 1.2),
        rotation_range=(-10.0, 10.0),
        shear_range=(-0.1, 0.1),
        contrast_strength=0.2,
    )


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

