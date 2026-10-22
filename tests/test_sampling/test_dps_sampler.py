"""Distributional and numerical checks for diffusion posterior sampling."""

from dataclasses import replace
from pathlib import Path
from runpy import run_path

import pytest
import torch

from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel
from score_qmri.models.unet import UNet
from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler, FullScoreDAPSConfig, FullScoreDAPSSampler
from score_qmri.sampling._utils import conjugate_gradient, to_2ch, to_complex


class GaussianPrior:
    device = torch.device("cpu")
    sigma_min = 0.001
    sigma_max = 20.0
    sigma_data = 0.2

    def __init__(self, covariance=None):
        self.covariance = torch.eye(2, dtype=torch.float64) * 0.04 if covariance is None else covariance

    def denoise(self, x, sigma, kind):
        sigma = sigma.expand(len(x))
        covariance = self.covariance.to(x)
        gain = torch.linalg.solve(
            covariance + sigma[:, None, None] ** 2 * torch.eye(2, device=x.device, dtype=x.dtype), covariance.expand(len(x), -1, -1)
        )
        return torch.einsum("bij,bcj->bci", gain, x.squeeze(2)).unsqueeze(2)

    def score(self, x, sigma, kind):
        sigma = sigma.expand(len(x))
        precision = self.covariance.to(x) + sigma[:, None, None] ** 2 * torch.eye(2, device=x.device, dtype=x.dtype)
        return -torch.linalg.solve(precision, x.squeeze(2).mT).mT.unsqueeze(2)


class MatrixOperator:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self, x):
        return (x @ self.matrix.to(x).mT,)

    @property
    def H(self):
        return MatrixOperator(self.matrix.mH)


def test_dps_style_guidance_matches_official_norm_correction_on_ve_bridge():
    following = 0.3
    model = GaussianPrior()
    operator = MatrixOperator(torch.eye(2, dtype=torch.float64))
    x = torch.tensor([[[[0.2, -0.1]], [[0.1, 0.3]]]], dtype=torch.float64)
    y = to_complex(torch.full_like(x, 0.05))
    sigma = 0.4
    cfg = DPSConfig(tau=0.05, guidance="dps", guidance_scale=0.3)
    result = DPSSampler(model, cfg)._step(x, torch.tensor([1]), operator, y, sigma, following, torch.zeros_like(x))
    gain = 0.04 / (0.04 + sigma**2)
    residual = gain * x - to_2ch(y)
    gradient = gain * residual / residual.norm()
    expected = (following / sigma) ** 2 * x + (1 - (following / sigma) ** 2) * gain * x - cfg.guidance_scale * gradient
    torch.testing.assert_close(result, expected)
    repeated = DPSSampler(model, replace(cfg, tau=0.5))._step(x, torch.tensor([1]), operator, y, sigma, following, torch.zeros_like(x))
    torch.testing.assert_close(result, repeated)


def test_dps_style_zero_residual_and_batch_independence():
    model = GaussianPrior()
    operator = MatrixOperator(torch.eye(2, dtype=torch.float64))
    cfg = DPSConfig(tau=0.05, guidance="dps")
    x = torch.tensor([0.0, 0.2], dtype=torch.float64)[:, None, None, None].expand(-1, 2, 1, 2)
    y = to_complex(torch.zeros_like(x))
    sampler = DPSSampler(model, cfg)
    batch = sampler._step(x, torch.tensor([1, 1]), operator, y, 0.4, 0.3, torch.zeros_like(x))
    assert torch.equal(batch[0], torch.zeros_like(batch[0]))
    single = sampler._step(x[1:], torch.tensor([1]), operator, y[1:], 0.4, 0.3, torch.zeros_like(x[1:]))
    torch.testing.assert_close(batch[1:], single)


def assert_gaussian_posterior(draws, covariance, operator, y, tau):
    matrix = operator.matrix
    target_cov = torch.linalg.inv(torch.linalg.inv(covariance) + matrix.mH @ matrix / tau**2)
    target_mean = target_cov.to(y) @ matrix.mH.to(y) @ y.flatten() / tau**2
    samples = draws[:, 0, 0, 0]
    for part, mean in [(samples.real, target_mean.real), (samples.imag, target_mean.imag)]:
        observed_cov = torch.cov(part.mT)
        scale = target_cov.diag().real.sqrt()
        covariance_se = ((scale[:, None] * scale[None, :]).square() + target_cov.real.square()).div(len(part) - 1).sqrt()
        assert torch.max((observed_cov - target_cov.real).abs() / covariance_se) < 5
        assert torch.max((part.mean(0) - mean).abs() / (scale / len(part) ** 0.5)) < 5


@pytest.mark.parametrize(
    "matrix,prior_variance",
    [
        (torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)), 0.01),
        (torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64), 0.01),
    ],
)
def test_dps_recovers_observed_nullspace_and_cross_covariance(matrix, prior_variance):
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * prior_variance)
    op = MatrixOperator(matrix)
    y = torch.tensor([0.15 + 0.05j, 0.02 - 0.01j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    y = op(op.H(y)[0])[0] if matrix[1, 1] == 0 else y
    cfg = DPSConfig(tau=0.05, num_steps=512, cg_rtol=1e-10)
    draws = DPSSampler(model, cfg).sample(op, y.expand(2048, -1, -1, -1, -1), 1, generator=torch.Generator().manual_seed(71))
    assert_gaussian_posterior(draws, model.covariance, op, y, cfg.tau)


def test_dps_is_invariant_to_measurement_units_and_reproducible():
    model = GaussianPrior()
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    y = torch.full((3, 1, 1, 1, 2), 0.15 + 0.05j, dtype=torch.complex128)
    cfg = DPSConfig(tau=0.05, num_steps=32, cg_rtol=1e-10)
    first = DPSSampler(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(8))
    scaled = DPSSampler(model, replace(cfg, tau=cfg.tau * 10)).sample(
        MatrixOperator(matrix * 10), y * 10, 1, generator=torch.Generator().manual_seed(8)
    )
    repeated = DPSSampler(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(first, scaled, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)


def test_dps_guidance_uses_diffusion_increment_and_conditional_noise():
    prior_variance = 0.01
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * prior_variance)
    op = MatrixOperator(torch.eye(2, dtype=torch.float64))
    x = torch.full((1, 2, 1, 2), 0.2, dtype=torch.float64)
    y = to_complex(torch.full_like(x, 0.1))
    sigma, following = 0.4, 0.3
    cfg = DPSConfig(tau=0.05, cg_rtol=1e-12)
    actual = DPSSampler(model, cfg)._step(x, torch.tensor([1]), op, y, sigma, following, torch.zeros_like(x))
    gain = prior_variance / (prior_variance + sigma**2)
    conditional_var = sigma**2 * gain
    posterior_mean = gain * x - conditional_var * (gain * x - to_2ch(y)) / (conditional_var + cfg.tau**2)
    ratio = (following / sigma) ** 2
    torch.testing.assert_close(actual, ratio * x + (1 - ratio) * posterior_mean, rtol=1e-12, atol=1e-12)


def test_cg_checks_actual_residual_and_handles_zero_batches():
    matrix = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    rhs = torch.tensor([[1.0 + 2j, 3.0 - 1j], [0j, 0j]], dtype=torch.complex128)
    answer = conjugate_gradient(lambda x: x @ matrix.to(x), rhs, 10, 1e-12)
    torch.testing.assert_close(answer @ matrix.to(answer), rhs)
    with pytest.raises(RuntimeError, match="converge"):
        conjugate_gradient(lambda x: x @ matrix.to(x), rhs, 1, 1e-12)
    with pytest.raises(RuntimeError, match="positive-definite"):
        conjugate_gradient(lambda x: -x, rhs, 10, 1e-12)


def test_dps_requires_physical_measurement_noise():
    tau = 0.0
    with pytest.raises(ValueError):
        DPSConfig(tau=tau)


@pytest.mark.parametrize("sampler_name", ["dps", "daps", "full-score-daps"])
def test_sampler_with_masking_operator(sampler_name):
    operator_type = run_path(str(Path(__file__).parents[2] / "scripts/sample_posterior_ensemble.py"))["MaskedFourier"]
    mask = torch.zeros(1, 1, 1, 32, 32)
    mask[..., ::2, :] = 1
    operator = operator_type(mask)
    model = DiffusionModel(
        unet=UNet(base_channels=8, num_res_blocks=1, channel_mult=(1,), attention_levels=(), cond_dim=16),
        diffusion=DiffusionConfig(cond_dim=16, sigma_min=0.01, sigma_max=1.0),
        compile_model=False,
    ).eval()
    sampler: DPSSampler | DAPSSampler | FullScoreDAPSSampler
    if sampler_name == "dps":
        sampler = DPSSampler(model, DPSConfig(tau=0.1, num_steps=3))
    elif sampler_name == "daps":
        sampler = DAPSSampler(
            model, DAPSConfig(tau=0.1, num_steps=3, sigma_min=0.02, reverse_sigma_min=0.01, reverse_steps=2, langevin_steps=2)
        )
    else:
        sampler = FullScoreDAPSSampler(model, FullScoreDAPSConfig(tau=0.1, num_steps=3, sigma_min=0.02, reverse_steps=2, langevin_steps=2))
    measurement = operator(torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64))[0]
    noise = None
    if sampler_name == "full-score-daps":
        sampler = FullScoreDAPSSampler(model, FullScoreDAPSConfig(tau=0.1, num_steps=3, sigma_min=0.02, reverse_steps=2, langevin_steps=2))
        noise = torch.randn_like(measurement, dtype=torch.complex128)
    image = sampler.sample(operator, measurement, 1, noise=noise, generator=torch.Generator().manual_seed(3))
    assert image.shape == measurement.shape
    assert image.dtype == measurement.dtype and image.isfinite().all()
