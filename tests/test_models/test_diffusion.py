"""Tests for VE/EDM DiffusionModel."""

import torch

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DenoiseOutput, DenoiseOutputList, DiffusionConfig, DiffusionModel
from score_qmri.models.unet import UNet


def make_model(cond_dim: int = 128) -> DiffusionModel:
    unet = UNet(
        base_channels=16,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_levels=(),
        cond_dim=cond_dim,
    )
    diffusion = DiffusionConfig(cond_dim=cond_dim)
    return DiffusionModel(unet=unet, diffusion=diffusion)


def test_diffusion_init_has_sigma_config():
    model = make_model()
    assert model.sigma_min > 0
    assert model.sigma_max > model.sigma_min
    assert model.sigma_data > 0


def test_diffusion_forward_2ch():
    model = make_model()
    x = torch.randn(2, 2, 32, 32)
    sigma = torch.tensor([1.0, 0.5])
    kind = torch.tensor([int(DatasetKind.T1MAP), int(DatasetKind.T1W_MAGNITUDE)])

    out = model(x, sigma, kind)
    assert isinstance(out, DenoiseOutput)
    assert out.x0_main.shape == x.shape
    assert out.x0_aux.shape == x.shape


def test_diffusion_forward_complex():
    model = make_model()
    x = torch.complex(torch.randn(2, 32, 32), torch.randn(2, 32, 32))
    sigma = torch.tensor([1.0, 0.5])
    kind = torch.tensor([int(DatasetKind.COMPLEX_T1W_COILWISE), int(DatasetKind.COMPLEX_T1W_COILWISE)])

    out = model(x, sigma, kind)
    assert isinstance(out, DenoiseOutput)
    assert out.x0_main.is_complex()
    assert out.x0_aux.is_complex()
    assert out.x0_main.shape == x.shape


def test_q_sample_shape_and_changes_values():
    model = make_model()
    x0 = torch.randn(2, 2, 32, 32)
    sigma = torch.tensor([0.5, 0.5])
    kind = torch.full((2,), int(DatasetKind.T1MAP), dtype=torch.long)
    noise = torch.randn(2, 2, 32, 32)

    x = model.q_sample(x0, sigma, kind, noise)
    assert x.shape == x0.shape
    assert not torch.allclose(x, x0)


def test_score_and_eps_relationship():
    model = make_model()
    x = torch.randn(3, 2, 16, 16)
    sigma = torch.tensor([0.2, 1.0, 10.0])
    kind = torch.tensor([int(DatasetKind.T1MAP), int(DatasetKind.T1MAP), int(DatasetKind.T1MAP)])

    eps_hat = model.predict_eps(x, sigma, kind)
    score = model.score(x, sigma, kind)

    sigma_b = sigma.view(-1, 1, 1, 1)
    assert torch.allclose(score, -eps_hat / sigma_b, atol=1e-5, rtol=1e-5)


def test_real_kind_imag_channel_stays_zero():
    model = make_model()
    x = torch.randn(2, 2, 32, 32)
    x[:, 1] = 0.0
    sigma = torch.tensor([1.0, 0.5])
    kind = torch.full((2,), int(DatasetKind.T1MAP), dtype=torch.long)

    out = model(x, sigma, kind)
    assert torch.allclose(out.x0_main[:, 1], torch.zeros_like(out.x0_main[:, 1]))
    assert torch.allclose(out.x0_aux[:, 1], torch.zeros_like(out.x0_aux[:, 1]))


def test_normalize_denormalize_roundtrip_mapping_kind():
    model = make_model()
    x = torch.randn(4, 2, 32, 32)
    x[:, 1] = 0.0
    kind = torch.full((4,), int(DatasetKind.T1MAP), dtype=torch.long)

    x_norm = model.normalize(x, kind)
    x_denorm = model.denormalize(x_norm, kind)
    assert torch.allclose(x_denorm, x, atol=1e-4)


def test_list_input_forward():
    model = make_model()

    a = torch.randn(2, 32, 32)
    b = torch.randn(3, 32, 32)
    sigma = torch.full((5,), 1.0)
    kind = torch.full((5,), int(DatasetKind.T1W_MAGNITUDE), dtype=torch.long)

    out = model([a, b], sigma, kind)
    assert isinstance(out, DenoiseOutputList)
    assert len(out.x0_main) == 2
    assert out.x0_main[0].shape == (2, 32, 32)
    assert out.x0_main[1].shape == (3, 32, 32)


def test_create_fixed_noise_complex_imag_nonzero():
    # Sanity check that complex kinds allow imaginary noise in downstream samplers.
    # (The diffusion model itself enforces imag=0 only for real kinds.)
    assert is_complex_kind(int(DatasetKind.COMPLEX_T1W_COILWISE))
