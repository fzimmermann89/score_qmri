"""Tests for loss functions."""

import pytest
import torch

from training.losses import LossScheduler, MSELoss, SSIML1Loss


def test_mse_loss_computes_masked_error():
    """Test MSE loss computes error only on masked pixels."""
    loss_fn = MSELoss()
    pred = torch.ones(2, 2, 64, 64)
    target = torch.zeros(2, 2, 64, 64)
    mask = torch.ones(2, 64, 64, dtype=torch.bool)

    loss = loss_fn(pred, target, mask)
    assert loss.item() == pytest.approx(1.0, rel=1e-5)


def test_mse_loss_ignores_masked_regions():
    """Test MSE loss ignores regions where mask is False."""
    loss_fn = MSELoss()
    pred = torch.ones(1, 1, 4, 4) * 10
    target = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, :2, :2] = True  # Only top-left 2x2 is valid

    pred[:, :, :2, :2] = 1.0  # Error of 1 in valid region
    loss = loss_fn(pred, target, mask)
    # Only 4 valid pixels with error 1
    assert loss.item() == pytest.approx(1.0, rel=1e-5)


def test_ssim_l1_loss_requires_library():
    """Test SSIM+L1 loss requires pytorch-msssim."""
    try:
        loss_fn = SSIML1Loss()
        pred = torch.rand(2, 2, 64, 64)
        target = torch.rand(2, 2, 64, 64)
        mask = torch.ones(2, 64, 64, dtype=torch.bool)

        loss = loss_fn(pred, target, mask)
        assert loss.item() >= 0
    except ImportError:
        pytest.skip("pytorch-msssim not available")


def test_loss_scheduler_interpolates():
    """Test LossScheduler interpolates between losses."""
    mse = MSELoss()
    try:
        ssim_l1 = SSIML1Loss()
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    # Schedule: always use MSE (weight 0)
    scheduler = LossScheduler(mse, ssim_l1, lambda s: 0.0)

    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    mask = torch.ones(2, 64, 64, dtype=torch.bool)

    loss = scheduler(pred, target, mask)
    mse_only = mse(pred, target, mask)
    assert torch.allclose(loss, mse_only, rtol=1e-4)


def test_loss_scheduler_tracks_step():
    """Test LossScheduler increments step counter."""
    mse = MSELoss()
    try:
        ssim_l1 = SSIML1Loss()
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    scheduler = LossScheduler(mse, ssim_l1, lambda s: 0.0)
    assert scheduler.step == 0

    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    mask = torch.ones(2, 64, 64, dtype=torch.bool)
    scheduler(pred, target, mask)

    assert scheduler.step == 1
