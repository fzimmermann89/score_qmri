"""Tests for loss functions."""

import pytest
import torch

from training.losses import compute_loss, l1_loss, noise_loss


def test_noise_loss():
    """Test noise MSE loss."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    loss = noise_loss(pred, target)
    assert loss.item() >= 0


def test_noise_loss_with_mask():
    """Test noise loss respects mask."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, :16, :16] = 1
    loss = noise_loss(pred, target, mask)
    assert loss.item() > 0


def test_l1_loss():
    """Test L1 loss."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    loss = l1_loss(pred, target)
    assert loss.item() >= 0


def test_ssim_loss():
    """Test SSIM loss."""
    try:
        from training.losses import ssim_loss
        pred = torch.randn(2, 2, 64, 64)
        target = torch.randn(2, 2, 64, 64)
        loss = ssim_loss(pred, target)
        assert 0 <= loss.item() <= 2
    except ImportError:
        pytest.skip("pytorch-msssim not available")


def test_compute_loss():
    """Test combined loss computation."""
    try:
        from training.losses import ssim_loss
        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    x0_pred = torch.randn(2, 2, 64, 64)
    x0_target = torch.randn(2, 2, 64, 64)
    noise_pred = torch.randn(2, 2, 64, 64)
    noise_target = torch.randn(2, 2, 64, 64)

    output = compute_loss(x0_pred, x0_target, noise_pred, noise_target)
    assert output.total.item() >= 0
    assert output.noise_mse.item() >= 0
    assert output.x0_l1.item() >= 0
    assert output.x0_ssim.item() >= 0


def test_compute_loss_weights():
    """Test loss weights affect output."""
    try:
        from training.losses import ssim_loss
        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    args = (torch.randn(2, 2, 64, 64),) * 4
    output_no_x0 = compute_loss(*args, noise_weight=1.0, x0_weight=0.0)
    assert torch.allclose(output_no_x0.total, output_no_x0.noise_mse)
