"""Tests for loss functions."""

import pytest
import torch

from score_qmri.training.losses import l1_loss, noise_loss, ssim_loss


def test_noise_loss():
    """Test noise MSE loss matches expected MSE."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    loss = noise_loss(pred, target)

    # Verify matches expected MSE
    expected = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(loss, expected)


def test_noise_loss_identical_inputs():
    """Test noise loss is zero for identical inputs."""
    x = torch.randn(2, 2, 32, 32)
    assert noise_loss(x, x).item() == 0.0


def test_noise_loss_with_spatial_mask():
    """Test noise loss respects spatial mask."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, :16, :16] = 1
    loss = noise_loss(pred, target, spatial_mask=mask)
    assert loss.item() > 0


def test_noise_loss_reduction_none():
    """Test noise_loss with reduction='none' returns per-sample (B,) tensor."""
    pred = torch.randn(4, 2, 32, 32)
    target = torch.randn(4, 2, 32, 32)
    loss = noise_loss(pred, target, reduction="none")
    assert loss.shape == (4,)
    assert (loss >= 0).all()


def test_noise_loss_reduction_none_with_spatial_mask():
    """Test noise_loss reduction='none' with per-sample spatial mask weighting."""
    pred = torch.ones(4, 2, 32, 32)
    target = torch.zeros(4, 2, 32, 32)
    mask = torch.ones(4, 1, 32, 32)
    mask[0] = 0
    loss = noise_loss(pred, target, spatial_mask=mask, reduction="none")
    assert loss.shape == (4,)
    assert loss[0].item() == 0.0
    assert (loss[1:] > 0).all()


def test_noise_loss_zero_mask_returns_zero():
    """Test noise_loss returns 0 when spatial mask is all zeros."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    loss = noise_loss(pred, target, spatial_mask=mask)
    assert loss.item() == 0.0


def test_noise_loss_with_channel_mask():
    """Test noise_loss with channel mask ignores masked channels."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    # Only count channel 0
    channel_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]]).reshape(2, 2, 1, 1)
    loss = noise_loss(pred, target, channel_mask=channel_mask)
    # Loss should be 1.0 (only real channel contributes, all ones vs zeros)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_l1_loss():
    """Test L1 loss matches expected L1."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    loss = l1_loss(pred, target)

    # Verify matches expected L1
    expected = torch.nn.functional.l1_loss(pred, target)
    assert torch.allclose(loss, expected)


def test_l1_loss_identical_inputs():
    """Test L1 loss is zero for identical inputs."""
    x = torch.randn(2, 2, 32, 32)
    assert l1_loss(x, x).item() == 0.0


def test_l1_loss_reduction_none():
    """Test l1_loss with reduction='none' returns per-sample (B,) tensor."""
    pred = torch.randn(4, 2, 32, 32)
    target = torch.randn(4, 2, 32, 32)
    loss = l1_loss(pred, target, reduction="none")
    assert loss.shape == (4,)
    assert (loss >= 0).all()


def test_l1_loss_reduction_none_with_spatial_mask():
    """Test l1_loss reduction='none' with spatial mask."""
    pred = torch.ones(4, 2, 32, 32)
    target = torch.zeros(4, 2, 32, 32)
    mask = torch.ones(4, 32, 32)
    mask[0] = 0
    loss = l1_loss(pred, target, spatial_mask=mask, reduction="none")
    assert loss.shape == (4,)
    assert loss[0].item() == 0.0
    assert (loss[1:] > 0).all()


def test_l1_loss_zero_mask_returns_zero():
    """Test l1_loss returns 0 when spatial mask is all zeros."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    mask = torch.zeros(2, 32, 32)
    loss = l1_loss(pred, target, spatial_mask=mask)
    assert loss.item() == 0.0


def test_l1_loss_with_channel_mask():
    """Test l1_loss with channel mask ignores masked channels."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    # Only count channel 0
    channel_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]]).reshape(2, 2, 1, 1)
    loss = l1_loss(pred, target, channel_mask=channel_mask)
    # Loss should be 1.0 (only real channel contributes, all ones vs zeros)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_ssim_loss():
    """Test SSIM loss."""
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    loss = ssim_loss(pred, target)
    assert 0 <= loss.item() <= 2


def test_ssim_loss_reduction_none():
    """Test ssim_loss with reduction='none' returns per-sample tensor."""
    pred = torch.randn(4, 2, 32, 32)
    target = torch.randn(4, 2, 32, 32)
    loss = ssim_loss(pred, target, reduction="none")
    # Returns (B, 1) or (B,) - squeeze to (B,) for consistency check
    loss_squeezed = loss.squeeze()
    assert loss_squeezed.shape == (4,)
    assert (loss_squeezed >= 0).all()


def test_ssim_loss_with_spatial_mask():
    """Test SSIM loss with spatial mask only considers non-masked windows."""
    pred = torch.zeros(1, 2, 32, 32)
    target = torch.zeros(1, 2, 32, 32)
    pred[:, :, 8:24, 8:24] = 1.0
    target[:, :, 8:24, 8:24] = 1.0

    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0

    loss_with_mask = ssim_loss(pred, target, spatial_mask=mask, window_size=7)
    loss_without_mask = ssim_loss(pred, target, spatial_mask=None, window_size=7)

    assert loss_with_mask < 0.1
    assert loss_without_mask < 0.1


def test_ssim_loss_mask_does_not_inflate():
    """Test that spatial masking doesn't artificially inflate SSIM."""
    pred = torch.zeros(1, 2, 32, 32)
    target = torch.zeros(1, 2, 32, 32)

    pred[:, :, 12:20, 12:20] = 1.0
    target[:, :, 12:20, 12:20] = 0.5

    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0

    loss_with_mask = ssim_loss(pred, target, spatial_mask=mask, window_size=7)
    assert loss_with_mask > 0.0


def test_ssim_loss_mismatched_shapes():
    """Test ssim_loss raises error for mismatched shapes."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        ssim_loss(pred, target)


def test_ssim_loss_invalid_window_size():
    """Test ssim_loss handles invalid window sizes."""
    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)

    with pytest.raises(ValueError, match="too small"):
        ssim_loss(pred, target, window_size=100)


def test_ssim_loss_all_zero_spatial_mask():
    """Test ssim_loss returns NaN-free value for all-zero spatial mask."""
    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)

    loss = ssim_loss(pred, target, spatial_mask=mask)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_ssim_loss_spatial_mask_shape_mismatch():
    """Test ssim_loss raises error for shape mismatch in spatial mask."""
    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)
    mask = torch.zeros(1, 2, 32)

    with pytest.raises((ValueError, IndexError, RuntimeError)):
        ssim_loss(pred, target, spatial_mask=mask)


def test_ssim_loss_negative_spatial_mask_values():
    """Test ssim_loss raises ValueError for negative spatial mask values."""
    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)
    mask = -torch.ones(1, 1, 32, 32)

    with pytest.raises(ValueError, match="negative"):
        ssim_loss(pred, target, spatial_mask=mask)


def test_ssim_loss_with_channel_mask():
    """Test ssim_loss with channel mask ignores masked channels."""
    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(2, 2, 32, 32)
    # Only count channel 0 (real kind)
    channel_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]]).reshape(2, 2, 1, 1)
    loss = ssim_loss(pred, target, channel_mask=channel_mask)
    assert torch.isfinite(loss)
    assert 0 <= loss.item() <= 2
