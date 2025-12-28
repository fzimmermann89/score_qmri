"""Tests for loss functions."""

import pytest
import torch

from training.losses import compute_diffusion_loss, l1_loss, l_simple


def test_l_simple_computes_mse():
    """Test L_simple computes MSE between noise tensors."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    loss = l_simple(pred, target)
    assert loss.item() >= 0


def test_l_simple_with_mask():
    """Test L_simple respects mask."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    mask = torch.zeros(2, 32, 32)
    mask[:, :16, :16] = 1

    loss = l_simple(pred, target, mask)
    assert loss.item() > 0


def test_l1_loss_computes():
    """Test L1 loss computes correctly."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    loss = l1_loss(pred, target)
    assert loss.item() >= 0


def test_l1_loss_with_mask():
    """Test L1 loss respects mask."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    mask = torch.zeros(2, 32, 32)
    mask[:, :16, :16] = 1

    loss = l1_loss(pred, target, mask)
    assert loss.item() > 0


def test_ssim_loss_requires_library():
    """Test SSIM loss raises if pytorch-msssim not available."""
    try:
        from training.losses import ssim_loss

        pred = torch.randn(2, 2, 64, 64)
        target = torch.randn(2, 2, 64, 64)
        loss = ssim_loss(pred, target)
        assert 0 <= loss.item() <= 2  # SSIM loss is 1 - SSIM, where SSIM in [-1, 1]
    except ImportError:
        pytest.skip("pytorch-msssim not available")


def test_compute_diffusion_loss():
    """Test combined diffusion loss computation."""
    try:
        from training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    x0_pred = torch.randn(2, 2, 64, 64)
    x0_target = torch.randn(2, 2, 64, 64)
    noise_pred = torch.randn(2, 2, 64, 64)
    noise_target = torch.randn(2, 2, 64, 64)

    output = compute_diffusion_loss(
        x0_pred=x0_pred,
        x0_target=x0_target,
        noise_pred=noise_pred,
        noise_target=noise_target,
    )

    assert output.total.item() >= 0
    assert output.l_simple.item() >= 0
    assert output.l1.item() >= 0
    assert output.ssim.item() >= 0


def test_compute_diffusion_loss_with_weights():
    """Test diffusion loss respects weights."""
    try:
        from training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    x0_pred = torch.randn(2, 2, 64, 64)
    x0_target = torch.randn(2, 2, 64, 64)
    noise_pred = torch.randn(2, 2, 64, 64)
    noise_target = torch.randn(2, 2, 64, 64)

    # Test with zero x0 weight
    output_no_x0 = compute_diffusion_loss(
        x0_pred=x0_pred,
        x0_target=x0_target,
        noise_pred=noise_pred,
        noise_target=noise_target,
        noise_weight=1.0,
        x0_weight=0.0,
    )
    assert torch.allclose(output_no_x0.total, output_no_x0.l_simple)
