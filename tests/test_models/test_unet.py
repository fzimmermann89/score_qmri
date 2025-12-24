"""Tests for UNet model."""

import torch

from models.unet import UNet


def test_unet_forward_preserves_spatial_dims():
    """Test UNet output has same spatial dimensions as input."""
    unet = UNet(in_channels=2, out_channels=2, base_channels=32, num_blocks=2)
    x = torch.randn(2, 2, 64, 64)
    out = unet(x)
    assert out.shape == (2, 2, 64, 64)


def test_unet_with_film_accepts_params():
    """Test UNet accepts FiLM parameters."""
    unet = UNet(in_channels=2, out_channels=2, base_channels=32, num_blocks=2, use_film=True)
    x = torch.randn(2, 2, 64, 64)
    film = {"scales": torch.ones(2, 2), "shifts": torch.zeros(2, 2)}

    out = unet(x, film)
    assert out.shape == (2, 2, 64, 64)


def test_unet_handles_various_sizes():
    """Test UNet works with different spatial sizes."""
    unet = UNet(in_channels=2, out_channels=2, base_channels=32, num_blocks=2)

    for size in [32, 64, 128]:
        x = torch.randn(1, 2, size, size)
        out = unet(x)
        assert out.shape == (1, 2, size, size)


def test_unet_without_film():
    """Test UNet works without FiLM conditioning."""
    unet = UNet(in_channels=2, out_channels=2, base_channels=32, num_blocks=2, use_film=False)
    x = torch.randn(2, 2, 64, 64)
    out = unet(x)
    assert out.shape == (2, 2, 64, 64)
