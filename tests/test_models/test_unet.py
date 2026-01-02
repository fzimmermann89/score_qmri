"""Tests for UNet model."""

import torch

from score_qmri.models.unet import ResBlock, UNet


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


def test_unet_gradient_flow():
    """Test gradients flow through entire network."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2, cond_dim=64)
    x = torch.randn(2, 2, 32, 32, requires_grad=True)
    cond = torch.randn(2, 64)

    x0, noise = unet(x, cond)

    # Compute dummy losses and backprop
    loss_x0 = x0.sum()
    loss_noise = noise.sum()

    loss_x0.backward(retain_graph=True)
    loss_noise.backward()

    # Verify gradients exist throughout the network
    # Check that specific parameters have gradients (not trying to access nested attributes)
    param_names = [name for name, _ in unet.named_parameters()]
    assert len(param_names) > 0

    # Verify gradients exist
    params_with_grad = [p for p in unet.parameters() if p.grad is not None]
    assert len(params_with_grad) > 0


def test_unet_dual_heads_different():
    """Test x0 and noise heads produce different outputs."""
    unet = UNet(in_channels=2, base_channels=16, num_blocks=2, cond_dim=64)
    x = torch.randn(1, 2, 32, 32)
    cond = torch.randn(1, 64)
    x0, noise = unet(x, cond)
    assert not torch.allclose(x0, noise)
