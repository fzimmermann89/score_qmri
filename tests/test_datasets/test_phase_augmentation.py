"""Tests for phase augmentation."""

import torch

from score_qmri.datasets.augmentation import PhaseAugmentation


def test_phase_augmentation_preserves_shape():
    """Test phase augmentation preserves input shapes."""
    aug = PhaseAugmentation(prob=1.0)
    image = torch.randn(1, 64, 64, dtype=torch.complex64)

    aug_image = aug(image)
    assert aug_image.shape == image.shape


def test_phase_augmentation_batch():
    """Test phase augmentation works with batched input."""
    aug = PhaseAugmentation(prob=1.0)
    image = torch.randn(4, 64, 64, dtype=torch.complex64)

    aug_image = aug(image)
    assert aug_image.shape == (4, 64, 64)


def test_phase_augmentation_preserves_magnitude():
    """Test phase rotation preserves magnitude."""
    aug = PhaseAugmentation(prob=1.0, poly_coeff_range=(0, 0))  # Only global phase
    image = torch.randn(1, 32, 32, dtype=torch.complex64)

    aug_image = aug(image)

    torch.testing.assert_close(aug_image.abs(), image.abs(), rtol=1e-5, atol=1e-5)


def test_phase_augmentation_zero_prob_no_change():
    """Test zero probability returns unchanged input."""
    aug = PhaseAugmentation(prob=0.0)
    image = torch.randn(1, 64, 64, dtype=torch.complex64)

    aug_image = aug(image)
    torch.testing.assert_close(aug_image, image)


def test_phase_field_generation():
    """Test polynomial phase field has expected properties."""
    aug = PhaseAugmentation(prob=1.0)
    phase = aug._generate_phase_field(2, 64, 64, torch.device("cpu"), torch.float32)

    assert phase.shape == (2, 64, 64)
    # Phase should be smooth (low variation between neighbors)
    diff_x = torch.abs(phase[:, :, 1:] - phase[:, :, :-1]).mean()
    diff_y = torch.abs(phase[:, 1:, :] - phase[:, :-1, :]).mean()
    assert diff_x < 0.5  # Reasonable smoothness
    assert diff_y < 0.5
