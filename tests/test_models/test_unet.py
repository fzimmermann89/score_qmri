"""Tests for UNet model."""

import pytest
import torch

from models.unet import UNet


@pytest.fixture
def unet():
    """Create small UNet for testing."""
    return UNet(in_channels=2, base_channels=16, num_blocks=2, use_film=True)


def test_unet_forward_preserves_spatial_dims(unet):
    """Test that UNet preserves spatial dimensions."""
    x = torch.randn(2, 2, 64, 64)
    x0_pred, noise_pred = unet(x)
    assert x0_pred.shape == (2, 2, 64, 64)
    assert noise_pred.shape == (2, 2, 64, 64)


def test_unet_with_film_accepts_params(unet):
    """Test UNet accepts FiLM parameters."""
    x = torch.randn(2, 2, 64, 64)
    film_params = {
        "scales": torch.ones(2, 2),
        "shifts": torch.zeros(2, 2),
    }
    x0_pred, noise_pred = unet(x, film_params)
    assert x0_pred.shape == (2, 2, 64, 64)
    assert noise_pred.shape == (2, 2, 64, 64)


def test_unet_handles_various_sizes():
    """Test UNet handles different input sizes."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2)
    for size in [32, 64, 128]:
        x = torch.randn(1, 2, size, size)
        x0_pred, noise_pred = unet(x)
        assert x0_pred.shape == (1, 2, size, size)
        assert noise_pred.shape == (1, 2, size, size)


def test_unet_without_film():
    """Test UNet works without FiLM conditioning."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2, use_film=False)
    x = torch.randn(2, 2, 64, 64)
    x0_pred, noise_pred = unet(x)
    assert x0_pred.shape == (2, 2, 64, 64)
    assert noise_pred.shape == (2, 2, 64, 64)


def test_unet_dual_heads_different():
    """Test that x0 and noise prediction heads produce different outputs."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2)
    x = torch.randn(1, 2, 64, 64)
    x0_pred, noise_pred = unet(x)
    assert not torch.allclose(x0_pred, noise_pred)
