"""Tests for phase augmentation."""

import torch

from score_qmri.datasets.augmentation import PhaseAugmentation


def test_phase_augmentation_preserves_shape():
    """Test phase augmentation preserves input shapes."""
    aug = PhaseAugmentation(prob=1.0)
    real = torch.randn(64, 64)
    imag = torch.randn(64, 64)

    aug_real, aug_imag = aug(real, imag)
    assert aug_real.shape == real.shape
    assert aug_imag.shape == imag.shape


def test_phase_augmentation_batch():
    """Test phase augmentation works with batched input."""
    aug = PhaseAugmentation(prob=1.0)
    real = torch.randn(4, 64, 64)
    imag = torch.randn(4, 64, 64)

    aug_real, aug_imag = aug(real, imag)
    assert aug_real.shape == (4, 64, 64)
    assert aug_imag.shape == (4, 64, 64)


def test_phase_augmentation_preserves_magnitude():
    """Test phase rotation preserves magnitude (approximately)."""
    aug = PhaseAugmentation(prob=1.0, poly_coeff_range=(0, 0))  # Only global phase
    real = torch.randn(32, 32)
    imag = torch.randn(32, 32)

    orig_mag = torch.sqrt(real**2 + imag**2)
    aug_real, aug_imag = aug(real, imag)
    aug_mag = torch.sqrt(aug_real**2 + aug_imag**2)

    assert torch.allclose(orig_mag, aug_mag, rtol=1e-5)


def test_phase_augmentation_zero_prob_no_change():
    """Test zero probability returns unchanged input."""
    aug = PhaseAugmentation(prob=0.0)
    real = torch.randn(64, 64)
    imag = torch.randn(64, 64)

    aug_real, aug_imag = aug(real, imag)
    assert torch.allclose(aug_real, real)
    assert torch.allclose(aug_imag, imag)


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
