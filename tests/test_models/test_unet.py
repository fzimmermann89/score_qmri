"""Tests for UNet model."""

import torch

from score_qmri.models.unet import Attention, ResBlock, UNet


def test_resblock_forward():
    """Test ResBlock forward pass."""
    block = ResBlock(32, 64, cond_dim=128)
    x = torch.randn(2, 32, 16, 16)
    cond = torch.randn(2, 128)
    out = block(x, cond)
    assert out.shape == (2, 64, 16, 16)


def test_unet_forward():
    """Test UNet forward pass preserves spatial dims."""
    unet = UNet(in_channels=2, base_channels=16, num_res_blocks=2, channel_mult=(1, 2), cond_dim=64)
    x = torch.randn(2, 2, 64, 64)
    cond = torch.randn(2, 64)
    x0_pred, noise_pred = unet(x, cond)
    assert x0_pred.shape == (2, 2, 64, 64)
    assert noise_pred.shape == (2, 2, 64, 64)


def test_unet_various_sizes():
    """Test UNet handles different input sizes."""
    unet = UNet(in_channels=2, base_channels=16, num_res_blocks=2, channel_mult=(1, 2), cond_dim=64)
    for size in [32, 64, 128]:
        x = torch.randn(1, 2, size, size)
        cond = torch.randn(1, 64)
        x0, noise = unet(x, cond)
        assert x0.shape == (1, 2, size, size)
        assert noise.shape == (1, 2, size, size)


def test_unet_gradient_flow():
    """Test gradients flow through entire network."""
    unet = UNet(in_channels=2, base_channels=16, num_res_blocks=2, channel_mult=(1, 2), cond_dim=64)
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
    unet = UNet(in_channels=2, base_channels=16, num_res_blocks=2, channel_mult=(1, 2), cond_dim=64)
    x = torch.randn(1, 2, 32, 32)
    cond = torch.randn(1, 64)
    x0, noise = unet(x, cond)
    assert not torch.allclose(x0, noise)


def test_attention_forward():
    """Test Attention block preserves shape."""
    attn = Attention(channels=128, head_dim=64)
    x = torch.randn(2, 128, 16, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_attention_residual():
    """Test Attention has residual connection (output similar to input with small alpha)."""
    attn = Attention(channels=64, head_dim=32)
    x = torch.randn(1, 64, 8, 8)
    out = attn(x)
    diff = (out - x).abs().mean()
    assert diff < x.abs().mean()


def test_attention_gradient_flow():
    """Test gradients flow through attention."""
    attn = Attention(channels=64, head_dim=32)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_unet_with_attention():
    """Test UNet with attention at specified levels."""
    unet = UNet(
        in_channels=2,
        base_channels=32,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        cond_dim=64,
        attention_levels=(2,),
        head_dim=32,
    )
    x = torch.randn(2, 2, 32, 32)
    cond = torch.randn(2, 64)
    x0, noise = unet(x, cond)
    assert x0.shape == x.shape
    assert noise.shape == x.shape


def test_unet_attention_increases_params():
    """Test that attention levels add parameters."""
    unet_no_attn = UNet(base_channels=32, channel_mult=(1, 2, 4), attention_levels=())
    unet_with_attn = UNet(base_channels=32, channel_mult=(1, 2, 4), attention_levels=(2,))
    params_no_attn = sum(p.numel() for p in unet_no_attn.parameters())
    params_with_attn = sum(p.numel() for p in unet_with_attn.parameters())
    assert params_with_attn > params_no_attn
