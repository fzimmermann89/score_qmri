"""Tests for DiffusionModel."""

import torch

from models.diffusion import DiffusionModel, DiffusionOutput


def test_diffusion_init():
    """Test diffusion model initialization."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    assert hasattr(model, "beta_min")
    assert hasattr(model, "beta_max")
    assert hasattr(model, "alpha_bar")
    assert hasattr(model, "beta")
    assert hasattr(model, "sigma")
    assert hasattr(model, "snr")


def test_diffusion_forward():
    """Test forward returns DiffusionOutput with dual predictions."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.1, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert isinstance(output, DiffusionOutput)
    assert output.x0.shape == (2, 2, 32, 32)
    assert output.noise.shape == (2, 2, 32, 32)


def test_diffusion_q_sample():
    """Test forward diffusion adds noise."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.5, 0.5])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    assert not torch.allclose(x_t, x0)


def test_diffusion_predict_x0():
    """Test x0 prediction from noise is inverse of q_sample."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.3, 0.7])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    x0_recovered = model.predict_x0(x_t, t, noise)
    assert torch.allclose(x0_recovered, x0, atol=1e-5)


def test_diffusion_alpha_bar():
    """Test alpha_bar is computed correctly for continuous time."""
    model = DiffusionModel(base_channels=16, num_blocks=2)

    # At t=0, alpha_bar should be close to 1
    ab_0 = model.alpha_bar(torch.tensor([0.0]))
    assert ab_0.item() > 0.99

    # At t=1, alpha_bar should be small
    ab_1 = model.alpha_bar(torch.tensor([1.0]))
    assert ab_1.item() < 0.01

    # Should be monotonically decreasing
    t = torch.linspace(0, 1, 10)
    ab = model.alpha_bar(t)
    assert torch.all(ab[:-1] >= ab[1:])


def test_diffusion_beta():
    """Test beta is linear in time."""
    model = DiffusionModel(beta_min=0.1, beta_max=20.0, base_channels=16, num_blocks=2)

    beta_0 = model.beta(torch.tensor([0.0]))
    beta_1 = model.beta(torch.tensor([1.0]))

    assert torch.isclose(beta_0, torch.tensor(0.1))
    assert torch.isclose(beta_1, torch.tensor(20.0))


def test_diffusion_sigma():
    """Test sigma = sqrt(1 - alpha_bar)."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    t = torch.tensor([0.5])

    ab = model.alpha_bar(t)
    sigma = model.sigma(t)
    assert torch.isclose(sigma, (1.0 - ab).sqrt())


def test_diffusion_snr():
    """Test SNR = alpha_bar / (1 - alpha_bar)."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    t = torch.tensor([0.5])

    ab = model.alpha_bar(t)
    snr = model.snr(t)
    assert torch.isclose(snr, ab / (1.0 - ab))


def test_diffusion_score():
    """Test score function output shape."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.5, 0.5])
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
