"""Tests for DiffusionModel."""

import torch

from models.diffusion import DiffusionModel


def test_diffusion_init_creates_schedule():
    """Test DiffusionModel creates noise schedule buffers."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)

    assert model.betas.shape == (100,)
    assert model.alphas_cumprod.shape == (100,)
    assert model.sqrt_alphas_cumprod.shape == (100,)


def test_diffusion_prepare_input_creates_two_channels():
    """Test prepare_input creates two-channel tensor."""
    model = DiffusionModel(num_kinds=3)
    image = torch.randn(2, 64, 64)
    kind = torch.tensor([1, 2])

    out = model.prepare_input(image, kind)
    assert out.shape == (2, 2, 64, 64)
    assert torch.allclose(out[:, 1], torch.zeros_like(out[:, 1]))


def test_diffusion_q_sample_adds_noise():
    """Test q_sample adds appropriate noise."""
    model = DiffusionModel(num_kinds=3, num_timesteps=1000)
    x0 = torch.randn(2, 2, 64, 64)
    t = torch.tensor([100, 500])

    x_t = model.q_sample(x0, t)
    assert x_t.shape == x0.shape
    # Noisy samples should differ from clean
    assert not torch.allclose(x_t, x0)


def test_diffusion_forward_predicts_x0():
    """Test forward pass predicts x0."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    x_t = torch.randn(2, 2, 64, 64)
    t = torch.tensor([10, 50])
    kind = torch.tensor([0, 1])

    pred_x0 = model(x_t, t, kind)
    assert pred_x0.shape == x_t.shape
