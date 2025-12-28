"""Tests for UNet model."""

import torch

from models.unet import ResBlock, UNet


def test_resblock_forward():
    """Test ResBlock forward pass."""
    block = ResBlock(32, 64, cond_dim=128)
    x = torch.randn(2, 32, 16, 16)
    cond = torch.randn(2, 128)
    out = block(x, cond)
    assert out.shape == (2, 64, 16, 16)


def test_unet_forward():
    """Test UNet forward pass preserves spatial dims."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2, cond_dim=64)
    x = torch.randn(2, 2, 64, 64)
    cond = torch.randn(2, 64)
    x0_pred, noise_pred = unet(x, cond)
    assert x0_pred.shape == (2, 2, 64, 64)
    assert noise_pred.shape == (2, 2, 64, 64)


def test_unet_various_sizes():
    """Test UNet handles different input sizes."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2, cond_dim=64)
    for size in [32, 64, 128]:
        x = torch.randn(1, 2, size, size)
        cond = torch.randn(1, 64)
        x0, noise = unet(x, cond)
        assert x0.shape == (1, 2, size, size)
        assert noise.shape == (1, 2, size, size)


def test_unet_dual_heads_different():
    """Test x0 and noise heads produce different outputs."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2, cond_dim=64)
    x = torch.randn(1, 2, 32, 32)
    cond = torch.randn(1, 64)
    x0, noise = unet(x, cond)
    assert not torch.allclose(x0, noise)
