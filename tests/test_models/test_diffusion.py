"""Tests for DiffusionModel."""

import torch

from score_qmri.models.diffusion import DiffusionModel, DiffusionOutput, DiffusionOutputList


def test_diffusion_init():
    """Test diffusion model initialization."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    assert hasattr(model, "beta_min")
    assert hasattr(model, "beta_max")
    assert hasattr(model, "alpha_bar")
    assert hasattr(model, "beta")
    assert hasattr(model, "sigma")
    assert hasattr(model, "snr")
    assert hasattr(model, "normalizer")


def test_diffusion_forward():
    """Test forward returns DiffusionOutput with dual predictions."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x_t = torch.randn(2, 32, 32)
    t = torch.tensor([0.1, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert isinstance(output, DiffusionOutput)
    assert output.x0.shape == (2, 32, 32)
    assert output.noise.shape == (2, 32, 32)


def test_diffusion_forward_complex():
    """Test forward with complex input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x_t = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    t = torch.tensor([0.1, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert output.x0.is_complex()
    assert output.noise.is_complex()
    assert output.x0.shape == (2, 32, 32)


def test_diffusion_forward_2ch():
    """Test forward with 2-channel input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x_t = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.1, 0.5])
    kind = torch.tensor([0, 1])

    output = model(x_t, t, kind)
    assert output.x0.shape == (2, 2, 32, 32)
    assert output.noise.shape == (2, 2, 32, 32)


def test_diffusion_q_sample():
    """Test forward diffusion adds noise."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 0])
    noise = torch.randn(2, 2, 32, 32)

    x_t = model.q_sample(x0, t, kind, noise)
    assert x_t.shape == x0.shape
    assert not torch.allclose(x_t, x0)


def test_diffusion_q_sample_complex():
    """Test q_sample with complex input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 0])
    noise = torch.randn(2, 2, 32, 32)

    x_t = model.q_sample(x0, t, kind, noise)
    assert x_t.is_complex()
    assert x_t.shape == x0.shape


def test_diffusion_q_sample_2ch():
    """Test q_sample with 2-channel input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 0])
    noise = torch.randn(2, 2, 32, 32)

    x_t = model.q_sample(x0, t, kind, noise)
    assert x_t.shape == (2, 2, 32, 32)


def test_predict_x0_is_inverse_of_q_sample():
    """Test x0 prediction from noise is inverse of q_sample."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([0.3, 0.7])
    kind = torch.tensor([0, 0])
    noise = torch.randn(2, 2, 32, 32)

    x_t = model.q_sample(x0, t, kind, noise)
    x0_recovered = model.predict_x0(x_t, t, noise, kind)
    assert torch.allclose(x0_recovered, x0, atol=1e-5)


def test_diffusion_alpha_bar():
    """Test alpha_bar is computed correctly for continuous time."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    ab_0 = model.alpha_bar(torch.tensor([0.0]))
    assert ab_0.item() > 0.99

    ab_1 = model.alpha_bar(torch.tensor([1.0]))
    assert ab_1.item() < 0.01

    t = torch.linspace(0, 1, 10)
    ab = model.alpha_bar(t)
    assert torch.all(ab[:-1] >= ab[1:])


def test_diffusion_beta():
    """Test beta is linear in time."""
    model = DiffusionModel(beta_min=0.1, beta_max=20.0, base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    beta_0 = model.beta(torch.tensor([0.0]))
    beta_1 = model.beta(torch.tensor([1.0]))

    assert torch.isclose(beta_0, torch.tensor(0.1))
    assert torch.isclose(beta_1, torch.tensor(20.0))


def test_diffusion_sigma():
    """Test sigma = sqrt(1 - alpha_bar)."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    t = torch.tensor([0.5])

    ab = model.alpha_bar(t)
    sigma = model.sigma(t)
    assert torch.isclose(sigma, (1.0 - ab).sqrt())


def test_diffusion_snr():
    """Test SNR = alpha_bar / (1 - alpha_bar)."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    t = torch.tensor([0.5])

    ab = model.alpha_bar(t)
    snr = model.snr(t)
    assert torch.isclose(snr, ab / (1.0 - ab))


def test_diffusion_score():
    """Test score function output shape."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x_t = torch.randn(2, 32, 32)
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 1])

    score = model.score(x_t, t, kind)
    assert score.shape == x_t.shape


def test_diffusion_normalize():
    """Test normalize method with native format."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    model.train()

    x = torch.randn(4, 64, 64) * 2.0 + 5.0
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    assert x_norm.shape == (4, 64, 64)
    assert x_norm.mean().abs() < 0.5


def test_diffusion_normalize_complex():
    """Test normalize method with complex input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    model.train()

    x = torch.complex(torch.randn(4, 64, 64), torch.randn(4, 64, 64))
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    assert x_norm.is_complex()
    assert x_norm.shape == (4, 64, 64)


def test_diffusion_normalize_2ch():
    """Test normalize method with 2-channel input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    model.train()

    x = torch.randn(4, 2, 64, 64) * 2.0 + 5.0
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    assert x_norm.shape == (4, 2, 64, 64)


def test_diffusion_denormalize():
    """Test denormalize method roundtrip."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    model.train()

    x = torch.randn(4, 64, 64) * 2.0 + 5.0
    mask = torch.ones(4, 64, 64, dtype=torch.bool)
    kind = torch.zeros(4, dtype=torch.long)

    x_norm = model.normalize(x, kind, mask)
    x_denorm = model.denormalize(x_norm, kind)

    assert torch.allclose(x_denorm, x, atol=1e-4)


def test_diffusion_boundary_t0():
    """Test t=0 boundary: alpha_bar ~1.0, sigma ~0, noise has no effect."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([0.0, 0.0])
    kind = torch.tensor([0, 0])

    ab = model.alpha_bar(t)
    assert ab[0].item() > 0.999
    assert ab[1].item() > 0.999

    sigma = model.sigma(t)
    assert sigma[0].item() < 0.01
    assert sigma[1].item() < 0.01

    noise = torch.randn(2, 2, 32, 32)
    x_t = model.q_sample(x0, t, kind, noise)
    assert torch.allclose(x_t, x0, atol=1e-3)


def test_diffusion_boundary_t1():
    """Test t=1 boundary: alpha_bar ~0, sigma ~1, x_t is pure noise."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 32, 32)
    t = torch.tensor([1.0, 1.0])
    kind = torch.tensor([0, 0])

    ab = model.alpha_bar(t)
    assert ab[0].item() < 0.01
    assert ab[1].item() < 0.01

    sigma = model.sigma(t)
    assert abs(sigma[0].item() - 1.0) < 0.01
    assert abs(sigma[1].item() - 1.0) < 0.01

    noise = torch.randn(2, 2, 32, 32)
    x_t = model.q_sample(x0, t, kind, noise)
    assert not torch.allclose(x_t, x0, atol=1e-2)


def test_diffusion_invalid_t_values():
    """Test t values outside [0, 1] work mathematically."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    t_neg = torch.tensor([-0.5, -0.1])
    ab_neg = model.alpha_bar(t_neg)
    assert (ab_neg > 0.0).all()

    t_large = torch.tensor([2.0, 5.0])
    ab_large = model.alpha_bar(t_large)
    assert (ab_large >= 0.0).all()
    assert (ab_large < 1e-3).all()


def test_diffusion_different_beta_configs():
    """Test diffusion model with different beta_min/beta_max values."""
    model = DiffusionModel(beta_min=1.0, beta_max=10.0, base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    ab_0 = model.alpha_bar(torch.tensor([0.0]))
    assert abs(ab_0.item() - 1.0) < 1e-6

    t_mid = torch.tensor([0.5])
    beta_mid = model.beta(t_mid)
    expected_beta_mid = 1.0 + 0.5 * (10.0 - 1.0)
    assert abs(beta_mid.item() - expected_beta_mid) < 1e-6


def test_diffusion_gradient_flow():
    """Test gradients flow through forward pass."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
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

    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_diffusion_device_property():
    """Test device property returns correct device."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    assert model.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.cuda()
        assert model.device.type == "cuda"


def test_diffusion_device_property_updates_on_move():
    """Test device property updates when model moved."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    assert model.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
        assert model.device.type == "cuda"
    else:
        model = model.to(torch.device("cpu"))
        assert model.device.type == "cpu"


def test_real_kind_imag_stays_zero():
    """Test that for real kinds, imaginary channel stays zero through diffusion."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 2, 32, 32)
    x0[:, 1] = 0
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([9, 9])

    x_t = model.q_sample(x0, t, kind)
    assert torch.allclose(x_t[:, 1], torch.zeros_like(x_t[:, 1]))

    output = model(x_t, t, kind)
    assert torch.allclose(output.x0[:, 1], torch.zeros_like(output.x0[:, 1]))
    assert torch.allclose(output.noise[:, 1], torch.zeros_like(output.noise[:, 1]))


def test_complex_kind_preserves_imag():
    """Test that for complex kinds, imaginary channel is preserved."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    x0 = torch.randn(2, 2, 32, 32)
    t = torch.tensor([0.5, 0.5])
    kind = torch.tensor([0, 0])

    x_t = model.q_sample(x0, t, kind)
    assert not torch.allclose(x_t[:, 1], torch.zeros_like(x_t[:, 1]))


def test_list_input_forward():
    """Test forward with list of tensors concatenated along batch."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    t1 = torch.randn(2, 32, 32)
    t2 = torch.randn(3, 32, 32)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    kind = torch.tensor([0, 0, 1, 1, 1])

    output = model([t1, t2], t, kind)
    assert isinstance(output, DiffusionOutputList)
    assert len(output.x0) == 2
    assert len(output.noise) == 2
    assert output.x0[0].shape == (2, 32, 32)
    assert output.x0[1].shape == (3, 32, 32)


def test_list_input_mixed_formats():
    """Test list with mixed formats (real and complex tensors)."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    t1 = torch.randn(2, 32, 32)
    t2 = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kind = torch.tensor([9, 9, 0, 0])

    output = model([t1, t2], t, kind)
    assert isinstance(output, DiffusionOutputList)
    assert not output.x0[0].is_complex()
    assert output.x0[1].is_complex()


def test_list_input_q_sample():
    """Test q_sample with list input."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    t1 = torch.randn(2, 32, 32)
    t2 = torch.randn(3, 32, 32)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    kind = torch.tensor([0, 0, 1, 1, 1])

    result = model.q_sample([t1, t2], t, kind)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 32, 32)
    assert result[1].shape == (3, 32, 32)


def test_list_input_2ch_elements():
    """Test list with 2ch tensor elements."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    t1 = torch.randn(2, 2, 32, 32)
    t2 = torch.randn(3, 2, 32, 32)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    kind = torch.tensor([0, 0, 1, 1, 1])

    output = model([t1, t2], t, kind)
    assert isinstance(output, DiffusionOutputList)
    assert output.x0[0].shape == (2, 2, 32, 32)
    assert output.x0[1].shape == (3, 2, 32, 32)


def test_list_input_score():
    """Test score() with list input returns list of correct shapes."""
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))

    t1 = torch.randn(2, 32, 32)
    t2 = torch.randn(3, 32, 32)
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    kind = torch.tensor([0, 0, 1, 1, 1])

    result = model.score([t1, t2], t, kind)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 32, 32)
    assert result[1].shape == (3, 32, 32)
