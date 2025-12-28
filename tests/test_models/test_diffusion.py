"""Tests for DiffusionModel."""

import torch

from models.diffusion import DiffusionModel, DiffusionOutput


def test_diffusion_init():
    """Test diffusion model initialization."""
    model = DiffusionModel(num_timesteps=100, base_channels=16, num_blocks=2)
    assert model.betas.shape == (100,)
    assert model.alphas_cumprod.shape == (100,)


def test_diffusion_forward():
    """Test forward returns DiffusionOutput."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100, base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([10, 20])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert isinstance(output, DiffusionOutput)
    assert output.x0.shape == (2, 2, 32, 32)
    assert output.noise.shape == (2, 2, 32, 32)


def test_diffusion_q_sample():
    """Test forward diffusion adds noise."""
    model = DiffusionModel(num_timesteps=100, base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([50, 50])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    assert not torch.allclose(x_t, x0)


def test_diffusion_predict_x0():
    """Test x0 prediction from noise."""
    model = DiffusionModel(num_timesteps=100, base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([50, 50])
    noise = torch.randn_like(x_t)

    x0_pred = model.predict_x0(x_t, t, noise)
    assert x0_pred.shape == x_t.shape


def test_diffusion_score():
    """Test score function computation."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100, base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([50, 50])
    kind = torch.tensor([0, 1])

    score = model.score(x_t, t, kind)
    assert score.shape == x_t.shape


def test_diffusion_prepare_input_magnitude():
    """Test prepare_input stacks magnitude with zeros."""
    image = torch.randn(4, 64, 64)
    prepared = DiffusionModel.prepare_input(image)
    assert prepared.shape == (4, 2, 64, 64)
    assert torch.allclose(prepared[:, 1], torch.zeros_like(prepared[:, 1]))


def test_diffusion_prepare_input_complex():
    """Test prepare_input passes through complex data."""
    image = torch.randn(4, 2, 64, 64)
    prepared = DiffusionModel.prepare_input(image)
    assert torch.allclose(prepared, image)
