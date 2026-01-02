"""Tests for DiffusionModel."""

import torch

from score_qmri.models.diffusion import DiffusionModel, DiffusionOutput


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


def test_predict_x0_is_inverse_of_q_sample():
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


def test_diffusion_boundary_t0():
    """Test t=0 boundary: alpha_bar ~1.0, sigma ~0, noise has no effect."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.0, 0.0])

    # At t=0, alpha_bar should be very close to 1
    ab = model.alpha_bar(t)
    assert ab[0].item() > 0.999
    assert ab[1].item() > 0.999

    # Sigma should be very close to 0
    sigma = model.sigma(t)
    assert sigma[0].item() < 0.01
    assert sigma[1].item() < 0.01

    # q_sample should return essentially unchanged x0
    noise = torch.randn_like(x0)
    x_t = model.q_sample(x0, t, noise)
    assert torch.allclose(x_t, x0, atol=1e-3)


def test_diffusion_boundary_t1():
    """Test t=1 boundary: alpha_bar ~0, sigma ~1, x_t is pure noise."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([1.0, 1.0])

    # At t=1, alpha_bar should be very small
    ab = model.alpha_bar(t)
    assert ab[0].item() < 0.01
    assert ab[1].item() < 0.01

    # Sigma should be close to 1
    sigma = model.sigma(t)
    assert abs(sigma[0].item() - 1.0) < 0.01
    assert abs(sigma[1].item() - 1.0) < 0.01

    # q_sample should return essentially pure noise
    noise = torch.randn_like(x0)
    x_t = model.q_sample(x0, t, noise)
    assert not torch.allclose(x_t, x0, atol=1e-2)


def test_diffusion_invalid_t_values():
    """Test t values outside [0, 1] work mathematically."""
    model = DiffusionModel(base_channels=16, num_blocks=2)

    # Negative t should still work (alpha_bar > 1)
    t_neg = torch.tensor([-0.5, -0.1])
    ab_neg = model.alpha_bar(t_neg)
    assert (ab_neg > 0.0).all()

    # t > 1 should work (alpha_bar very small, can be zero for very large t)
    t_large = torch.tensor([2.0, 5.0])
    ab_large = model.alpha_bar(t_large)
    assert (ab_large >= 0.0).all()
    assert (ab_large < 1e-3).all()


def test_diffusion_different_beta_configs():
    """Test diffusion model with different beta_min/beta_max values."""
    model = DiffusionModel(beta_min=1.0, beta_max=10.0, base_channels=16, num_blocks=2)

    # At t=0, alpha_bar should be 1.0 regardless of betas
    ab_0 = model.alpha_bar(torch.tensor([0.0]))
    assert abs(ab_0.item() - 1.0) < 1e-6

    # At t=0.5, should reflect linear beta schedule
    t_mid = torch.tensor([0.5])
    beta_mid = model.beta(t_mid)
    expected_beta_mid = 1.0 + 0.5 * (10.0 - 1.0)
    assert abs(beta_mid.item() - expected_beta_mid) < 1e-6


def test_diffusion_gradient_flow():
    """Test gradients flow through forward pass."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 2, 32, 32, requires_grad=True)
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)

    # Check gradients exist for x0 and noise predictions
    assert output.x0.grad_fn is not None
    assert output.noise.grad_fn is not None

    # Compute a dummy loss and backprop
    loss_x0 = output.x0.sum()
    loss_noise = output.noise.sum()

    loss_x0.backward(retain_graph=True)
    loss_noise.backward()

    # Verify gradients exist
    x0_grad = next(p for p in model.unet.parameters() if p.grad is not None)
    assert x0_grad is not None


def test_diffusion_device_property():
    """Test device property returns correct device."""
    model = DiffusionModel(base_channels=16, num_blocks=2)

    # Model starts on CPU
    assert model.device.type == "cpu"

    # Move to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        assert model.device.type == "cuda"


def test_diffusion_device_property_updates_on_move():
    """Test device property updates when model moved."""
    model = DiffusionModel(base_channels=16, num_blocks=2)

    # Initially on CPU
    device_initial = model.device
    assert device_initial.type == "cpu"

    # Move parameters to device (if CUDA available)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
        assert model.device.type == "cuda"
    else:
        # Even on CPU, we can test with explicit to() call
        model = model.to(torch.device("cpu"))
        assert model.device.type == "cpu"
