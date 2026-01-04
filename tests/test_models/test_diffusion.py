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
    assert hasattr(model, "normalizer")


def test_diffusion_forward():
    """Test forward returns DiffusionOutput with dual predictions."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 32, 32)
    t = torch.tensor([0.1, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert isinstance(output, DiffusionOutput)
    assert output.x0.shape == (2, 32, 32)
    assert output.noise.shape == (2, 32, 32)


def test_diffusion_forward_complex():
    """Test forward with complex input."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x_t = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    t = torch.tensor([0.1, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert output.x0.is_complex()
    assert output.noise.is_complex()
    assert output.x0.shape == (2, 32, 32)


def test_diffusion_q_sample():
    """Test forward diffusion adds noise."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([0.5, 0.5])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    assert not torch.allclose(x_t, x0)


def test_diffusion_q_sample_complex():
    """Test q_sample with complex input."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    t = torch.tensor([0.5, 0.5])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    assert x_t.is_complex()
    assert x_t.shape == x0.shape


def test_predict_x0_is_inverse_of_q_sample():
    """Test x0 prediction from noise is inverse of q_sample."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([0.3, 0.7])
    noise = torch.randn_like(x0)

    x_t = model.q_sample(x0, t, noise)
    x0_recovered = model.predict_x0(x_t, t, noise)
    assert torch.allclose(x0_recovered, x0, atol=1e-5)


def test_diffusion_alpha_bar():
    """Test alpha_bar is computed correctly for continuous time."""
    model = DiffusionModel(base_channels=16, num_blocks=2)

    ab_0 = model.alpha_bar(torch.tensor([0.0]))
    assert ab_0.item() > 0.99

    ab_1 = model.alpha_bar(torch.tensor([1.0]))
    assert ab_1.item() < 0.01

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
    x_t = torch.randn(2, 32, 32)
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
    """Test prepare_input converts complex to 2-channel."""
    image = torch.complex(torch.randn(4, 64, 64), torch.randn(4, 64, 64))
    prepared = DiffusionModel.prepare_input(image)
    assert prepared.shape == (4, 2, 64, 64)
    assert torch.allclose(prepared[:, 0], image.real)
    assert torch.allclose(prepared[:, 1], image.imag)


def test_diffusion_undo_prepare_magnitude():
    """Test undo_prepare recovers magnitude."""
    image_2ch = torch.randn(4, 2, 64, 64)
    recovered = DiffusionModel.undo_prepare(image_2ch, to_complex=False)
    assert recovered.shape == (4, 64, 64)
    assert torch.allclose(recovered, image_2ch[:, 0])


def test_diffusion_undo_prepare_complex():
    """Test undo_prepare recovers complex."""
    image_2ch = torch.randn(4, 2, 64, 64)
    recovered = DiffusionModel.undo_prepare(image_2ch, to_complex=True)
    assert recovered.is_complex()
    assert recovered.shape == (4, 64, 64)
    assert torch.allclose(recovered.real, image_2ch[:, 0])
    assert torch.allclose(recovered.imag, image_2ch[:, 1])


def test_diffusion_normalize():
    """Test normalize method with native format."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    model.train()

    x = torch.randn(4, 64, 64) * 2.0 + 5.0
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    assert x_norm.shape == (4, 64, 64)
    assert x_norm.mean().abs() < 0.5


def test_diffusion_normalize_complex():
    """Test normalize method with complex input."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    model.train()

    x = torch.complex(torch.randn(4, 64, 64), torch.randn(4, 64, 64))
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    assert x_norm.is_complex()
    assert x_norm.shape == (4, 64, 64)


def test_diffusion_denormalize():
    """Test denormalize method roundtrip."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    model.train()

    x = torch.randn(4, 64, 64) * 2.0 + 5.0
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    x_denorm = model.denormalize(x_norm, kind)

    assert torch.allclose(x_denorm, x, atol=1e-4)


def test_diffusion_boundary_t0():
    """Test t=0 boundary: alpha_bar ~1.0, sigma ~0, noise has no effect."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([0.0, 0.0])

    ab = model.alpha_bar(t)
    assert ab[0].item() > 0.999
    assert ab[1].item() > 0.999

    sigma = model.sigma(t)
    assert sigma[0].item() < 0.01
    assert sigma[1].item() < 0.01

    noise = torch.randn_like(x0)
    x_t = model.q_sample(x0, t, noise)
    assert torch.allclose(x_t, x0, atol=1e-3)


def test_diffusion_boundary_t1():
    """Test t=1 boundary: alpha_bar ~0, sigma ~1, x_t is pure noise."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([1.0, 1.0])

    ab = model.alpha_bar(t)
    assert ab[0].item() < 0.01
    assert ab[1].item() < 0.01

    sigma = model.sigma(t)
    assert abs(sigma[0].item() - 1.0) < 0.01
    assert abs(sigma[1].item() - 1.0) < 0.01

    noise = torch.randn_like(x0)
    x_t = model.q_sample(x0, t, noise)
    assert not torch.allclose(x_t, x0, atol=1e-2)


def test_diffusion_invalid_t_values():
    """Test t values outside [0, 1] work mathematically."""
    model = DiffusionModel(base_channels=16, num_blocks=2)

    t_neg = torch.tensor([-0.5, -0.1])
    ab_neg = model.alpha_bar(t_neg)
    assert (ab_neg > 0.0).all()

    t_large = torch.tensor([2.0, 5.0])
    ab_large = model.alpha_bar(t_large)
    assert (ab_large >= 0.0).all()
    assert (ab_large < 1e-3).all()


def test_diffusion_different_beta_configs():
    """Test diffusion model with different beta_min/beta_max values."""
    model = DiffusionModel(beta_min=1.0, beta_max=10.0, base_channels=16, num_blocks=2)

    ab_0 = model.alpha_bar(torch.tensor([0.0]))
    assert abs(ab_0.item() - 1.0) < 1e-6

    t_mid = torch.tensor([0.5])
    beta_mid = model.beta(t_mid)
    expected_beta_mid = 1.0 + 0.5 * (10.0 - 1.0)
    assert abs(beta_mid.item() - expected_beta_mid) < 1e-6


def test_diffusion_gradient_flow():
    """Test gradients flow through forward pass."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    x_t = torch.randn(2, 32, 32, requires_grad=True)
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)

    assert output.x0.grad_fn is not None
    assert output.noise.grad_fn is not None

    loss_x0 = output.x0.sum()
    loss_noise = output.noise.sum()

    loss_x0.backward(retain_graph=True)
    loss_noise.backward()

    x0_grad = next(p for p in model.unet.parameters() if p.grad is not None)
    assert x0_grad is not None


def test_diffusion_device_property():
    """Test device property returns correct device."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    assert model.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.cuda()
        assert model.device.type == "cuda"


def test_diffusion_device_property_updates_on_move():
    """Test device property updates when model moved."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    assert model.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
        assert model.device.type == "cuda"
    else:
        model = model.to(torch.device("cpu"))
        assert model.device.type == "cpu"
