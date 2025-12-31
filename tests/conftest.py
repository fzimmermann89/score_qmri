"""Pytest fixtures and shared test utilities."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from datasets.config import AugmentationConfig


@pytest.fixture
def sample_h5_file_complex():
    """Create temporary H5 file with complex data (grouped format)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name

    with h5py.File(h5_path, "w") as f:
        # Create groups for coilwise and combined
        coilwise_group = f.create_group("coilwise")
        combined_group = f.create_group("combined")

        # Subject 1: (N_z=3, N_coils=4, H=64, W=64)
        real1 = np.random.randn(3, 4, 64, 64).astype(np.float32)
        imag1 = np.random.randn(3, 4, 64, 64).astype(np.float32)
        coilwise_group.create_dataset("subject_001", data=real1 + 1j * imag1)

        # Subject 1 combined: (N_z=3, H=64, W=64)
        comb_real1 = np.random.randn(3, 64, 64).astype(np.float32)
        comb_imag1 = np.random.randn(3, 64, 64).astype(np.float32)
        combined_group.create_dataset("subject_001", data=comb_real1 + 1j * comb_imag1)

        # Subject 2: (N_z=2, N_coils=4, H=64, W=64)
        real2 = np.random.randn(2, 4, 64, 64).astype(np.float32)
        imag2 = np.random.randn(2, 4, 64, 64).astype(np.float32)
        coilwise_group.create_dataset("subject_002", data=real2 + 1j * imag2)

        # Subject 2 combined: (N_z=2, H=64, W=64)
        comb_real2 = np.random.randn(2, 64, 64).astype(np.float32)
        comb_imag2 = np.random.randn(2, 64, 64).astype(np.float32)
        combined_group.create_dataset("subject_002", data=comb_real2 + 1j * comb_imag2)

    yield h5_path

    Path(h5_path).unlink(missing_ok=True)


@pytest.fixture
def sample_h5_file_magnitude():
    """Create temporary H5 file with magnitude data (flat subjects)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name

    with h5py.File(h5_path, "w") as f:
        # Subject 1: (N_z=5, H=64, W=64)
        f.create_dataset(
            "subject_001",
            data=np.abs(np.random.randn(5, 64, 64).astype(np.float32)),
        )
        # Subject 2: (N_z=3, H=64, W=64)
        f.create_dataset(
            "subject_002",
            data=np.abs(np.random.randn(3, 64, 64).astype(np.float32)),
        )

    yield h5_path

    Path(h5_path).unlink(missing_ok=True)


@pytest.fixture
def sample_h5_file_mapping():
    """Create temporary H5 file with mapping data (flat subjects)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name

    with h5py.File(h5_path, "w") as f:
        # Subject 1: (N_z=4, H=64, W=64) - simulating T1 values in ms
        f.create_dataset(
            "subject_001",
            data=np.random.uniform(500, 2000, (4, 64, 64)).astype(np.float32),
        )
        # Subject 2: (N_z=3, H=64, W=64)
        f.create_dataset(
            "subject_002",
            data=np.random.uniform(500, 2000, (3, 64, 64)).astype(np.float32),
        )

    yield h5_path

    Path(h5_path).unlink(missing_ok=True)


# Keep old fixture name for backward compatibility with other tests
@pytest.fixture
def sample_h5_file(sample_h5_file_magnitude):
    """Alias for magnitude H5 file (backward compatibility)."""
    return sample_h5_file_magnitude


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
        geometric_prob=0.3,
        contrast_prob=0.3,
        phase_prob=0.5,
        scale_range=(0.9, 1.2),
        rotation_range=(-10.0, 10.0),
        shear_range=(-0.1, 0.1),
        contrast_strength=0.2,
    )


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
