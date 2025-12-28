"""Tests for DiffusionModel."""

import torch

from models.diffusion import DiffusionModel, DiffusionOutput


def test_diffusion_init_creates_schedule():
    """Test diffusion model initializes noise schedule buffers."""
    model = DiffusionModel(num_timesteps=100)
    assert hasattr(model, "betas")
    assert hasattr(model, "alphas_cumprod")
    assert model.betas.shape == (100,)


def test_diffusion_prepare_input_creates_two_channels():
    """Test prepare_input stacks magnitude data with zeros."""
    model = DiffusionModel()
    image = torch.randn(4, 64, 64)
    kind = torch.zeros(4, dtype=torch.long)
    prepared = model.prepare_input(image, kind)
    assert prepared.shape == (4, 2, 64, 64)
    assert torch.allclose(prepared[:, 1], torch.zeros_like(prepared[:, 1]))


def test_diffusion_prepare_input_passes_complex():
    """Test prepare_input passes through 2-channel complex data."""
    model = DiffusionModel()
    image = torch.randn(4, 2, 64, 64)
    kind = torch.zeros(4, dtype=torch.long)
    prepared = model.prepare_input(image, kind)
    assert prepared.shape == (4, 2, 64, 64)
    assert torch.allclose(prepared, image)


def test_diffusion_q_sample_adds_noise():
    """Test forward diffusion adds noise."""
    model = DiffusionModel(num_timesteps=100)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([50, 50])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    assert not torch.allclose(x_t, x0)


def test_diffusion_forward_returns_output():
    """Test forward returns DiffusionOutput with both predictions."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([10, 20])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert isinstance(output, DiffusionOutput)
    assert output.x0_pred.shape == (2, 2, 32, 32)
    assert output.noise_pred.shape == (2, 2, 32, 32)


def test_diffusion_predict_x0_from_noise():
    """Test x0 prediction from noise prediction."""
    model = DiffusionModel(num_timesteps=100)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([50, 50])
    noise_pred = torch.randn_like(x_t)

    x0_pred = model.predict_x0_from_noise(x_t, t, noise_pred)
    assert x0_pred.shape == x_t.shape
