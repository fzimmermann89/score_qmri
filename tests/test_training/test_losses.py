"""Tests for loss functions."""

import pytest
import torch

from score_qmri.training.losses import l1_loss, noise_loss


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


def test_noise_loss_with_mask():
    """Test noise loss respects mask."""
    pred = torch.ones(2, 2, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, :16, :16] = 1
    loss = noise_loss(pred, target, mask)
    assert loss.item() > 0


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


def test_ssim_loss():
    """Test SSIM loss."""
    from score_qmri.training.losses import ssim_loss

    pred = torch.randn(2, 2, 64, 64)
    target = torch.randn(2, 2, 64, 64)
    loss = ssim_loss(pred, target)
    assert 0 <= loss.item() <= 2


def test_ssim_loss_with_mask():
    """Test SSIM loss with mask only considers non-masked windows."""
    from score_qmri.training.losses import ssim_loss

    # Create identical images in the unmasked region
    pred = torch.zeros(1, 2, 32, 32)
    target = torch.zeros(1, 2, 32, 32)
    pred[:, :, 8:24, 8:24] = 1.0
    target[:, :, 8:24, 8:24] = 1.0

    # Mask that covers only the matching region
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0

    # With mask, should be low loss (only considers masked region where images match)
    loss_with_mask = ssim_loss(pred, target, mask=mask, window_size=7)

    # Without mask, should also be low loss (images match everywhere)
    loss_without_mask = ssim_loss(pred, target, mask=None, window_size=7)

    # Both should be low since images match
    assert loss_with_mask < 0.1
    assert loss_without_mask < 0.1


def test_ssim_loss_mask_does_not_inflate():
    """Test that masking doesn't artificially inflate SSIM."""
    from score_qmri.training.losses import ssim_loss

    # Create case where unmasked region has mismatch but masked region is zero
    pred = torch.zeros(1, 2, 32, 32)
    target = torch.zeros(1, 2, 32, 32)

    # Mismatch in center region
    pred[:, :, 12:20, 12:20] = 1.0
    target[:, :, 12:20, 12:20] = 0.5

    # Mask that covers the mismatched region
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:22, 10:22] = 1.0

    # With mask, should capture the mismatch
    loss_with_mask = ssim_loss(pred, target, mask=mask, window_size=7)

    # Loss should be > 0 since there's a mismatch in masked region
    assert loss_with_mask > 0.0


def test_ssim_loss_mismatched_shapes():
    """Test ssim_loss raises error for mismatched shapes."""
    from score_qmri.training.losses import ssim_loss

    pred = torch.randn(2, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        ssim_loss(pred, target)


def test_ssim_loss_invalid_window_size():
    """Test ssim_loss handles invalid window sizes."""
    from score_qmri.training.losses import ssim_loss

    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)

    # Window size larger than image should raise error
    with pytest.raises(ValueError, match="too small"):
        ssim_loss(pred, target, window_size=100)


def test_ssim_loss_all_zero_mask():
    """Test ssim_loss returns NaN-free value for all-zero mask."""
    from score_qmri.training.losses import ssim_loss

    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)

    # All-zero mask should still return a valid (NaN-free) loss
    loss = ssim_loss(pred, target, mask=mask)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


def test_ssim_loss_mask_shape_mismatch():
    """Test ssim_loss raises error for shape mismatch in mask."""
    from score_qmri.training.losses import ssim_loss

    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)
    mask = torch.zeros(1, 2, 32)  # Wrong shape: (B, C, H) instead of (B, 1, H, W)

    with pytest.raises((ValueError, IndexError, RuntimeError)):
        ssim_loss(pred, target, mask=mask)


def test_ssim_loss_negative_mask_values():
    """Test ssim_loss raises ValueError for negative mask values."""
    from score_qmri.training.losses import ssim_loss

    pred = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 2, 32, 32)
    mask = -torch.ones(1, 1, 32, 32)

    with pytest.raises(ValueError, match="negative"):
        ssim_loss(pred, target, mask=mask)
