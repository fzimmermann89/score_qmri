"""Verify DAPS samples its stated conditional target with full preconditioned noise."""

from dataclasses import replace

import pytest
import torch

from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler

from .test_dps_sampler import GaussianPrior, MatrixOperator, assert_gaussian_posterior


class BimodalPrior:
    device = torch.device("cpu")
    sigma_min = 0.001
    sigma_max = 20.0
    sigma_data = 0.2

    def denoise(self, x, sigma, kind):
        sigma = sigma.expand(len(x))
        variance = 0.01 + sigma[:, None].square()
        centers = x.new_tensor([[-0.15, -0.3], [0.15, 0.3]])
        real = x[:, 0, 0]
        weights = (-0.5 * (real[:, None] - centers).square().sum(-1) / variance).softmax(-1)
        gain = 0.01 / variance
        return torch.stack([gain * real + (1 - gain) * (weights @ centers), gain * x[:, 1, 0]], dim=1).unsqueeze(2)

    def score(self, x, sigma, kind):
        sigma = sigma.expand(len(x))
        return (self.denoise(x, sigma, kind) - x) / sigma[:, None, None, None].square()


class HeterogeneousGaussianPrior(GaussianPrior):
    def __init__(self):
        super().__init__(torch.diag(torch.tensor([0.04] * 8 + [1e-6] * 8, dtype=torch.float64)))
        self.variance = self.covariance.diag().reshape(1, 1, 1, 16)

    def denoise(self, x, sigma, kind):
        variance = self.variance + sigma.expand(len(x))[:, None, None, None].square()
        return x * self.variance / variance

    def score(self, x, sigma, kind):
        return -x / (self.variance + sigma.expand(len(x))[:, None, None, None].square())


class UninformativeOperator:
    def __call__(self, x):
        return (torch.zeros_like(x),)

    @property
    def H(self):
        return self


def test_stiff_background_does_not_freeze_foreground_mixing():
    model = HeterogeneousGaussianPrior()
    initial = torch.zeros(4096, 1, 1, 1, 16, dtype=torch.complex128)
    cfg = DAPSConfig(tau=0.01, num_steps=3, reverse_steps=2)
    # An uninformative likelihood leaves the exact smoothed prior as posterior.
    # Starting at its mean requires generating the missing foreground variance.
    draws = DAPSSampler(model, cfg).sample(
        UninformativeOperator(), initial, 1, initial=initial, generator=torch.Generator().manual_seed(71)
    )[:, 0, 0, 0]
    target_variance = model.variance.flatten() + model.sigma_min**2
    for part in [draws.real, draws.imag]:
        assert torch.max((part.var(0) / target_variance - 1).abs()) < 0.1
        assert torch.max(part.mean(0).abs() / (target_variance / len(part)).sqrt()) < 5


def test_daps_corrects_bimodal_dps_initialization():
    # The measurement changes mode probabilities; checking Gaussian moments
    # alone cannot detect a sampler that keeps the prior's equal mode weights.
    model = BimodalPrior()
    operator = MatrixOperator(torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)))
    y = torch.tensor([0.1 + 0j, 0j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    cfg = DAPSConfig(tau=0.1)
    measurement = y.expand(4096, -1, -1, -1, -1)
    initial = DPSSampler(model, DPSConfig(tau=cfg.tau)).sample(operator, measurement, 1, generator=torch.Generator().manual_seed(61))
    draws = (
        DAPSSampler(model, cfg)
        .sample(operator, measurement, 1, initial=initial, generator=torch.Generator().manual_seed(62))[:, 0, 0, 0]
        .real
    )
    variance = 0.01 + model.sigma_min**2
    means = torch.tensor([[-0.15, -0.3], [0.15, 0.3]], dtype=torch.float64)
    weights = (-0.5 * (0.1 - means[:, 0]).square() / (variance + cfg.tau**2)).softmax(0)
    means[:, 0] = (means[:, 0] * cfg.tau**2 + 0.1 * variance) / (variance + cfg.tau**2)
    component_variance = torch.tensor([variance * cfg.tau**2 / (variance + cfg.tau**2), variance], dtype=torch.float64)
    mean = (weights[:, None] * means).sum(0)
    centered = means - mean
    covariance = component_variance.diag() + (weights[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(0)
    normal = torch.distributions.Normal(means[:, 1], variance**0.5)
    positive_probability = (weights * (1 - normal.cdf(torch.tensor(0.0)))).sum()
    standard_error = (positive_probability * (1 - positive_probability) / len(draws)).sqrt()
    assert abs((initial[:, 0, 0, 0, 1].real > 0).double().mean() - positive_probability) > 5 * standard_error
    assert abs((draws[:, 1] > 0).double().mean() - positive_probability) < 5 * standard_error
    assert torch.max((draws.mean(0) - mean).abs() / (covariance.diag() / len(draws)).sqrt()) < 5
    scale = covariance.diag().sqrt()
    assert torch.max((torch.cov(draws.mT) - covariance).abs() / (scale[:, None] * scale[None, :])) < 0.1


@pytest.mark.parametrize("correlated", [False, True])
def test_daps_recovers_gaussian_posterior_and_cross_covariance(correlated):
    covariance = torch.tensor([[0.04, 0.012], [0.012, 0.07]], dtype=torch.float64) if correlated else None
    model = GaussianPrior(covariance)
    op = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.0]], dtype=torch.float64))
    y = torch.tensor([0.15 + 0.05j, 0j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    cfg = DAPSConfig(tau=0.05, cg_rtol=1e-10)
    draws = DAPSSampler(model, cfg).sample(op, y.expand(8192, -1, -1, -1, -1), 1, generator=torch.Generator().manual_seed(73))
    prior_sigma = model.sigma_min if cfg.prior_sigma is None else cfg.prior_sigma
    prior_covariance = model.covariance + prior_sigma**2 * torch.eye(2, dtype=torch.float64)
    assert_gaussian_posterior(draws, prior_covariance, op, y, cfg.tau)


def test_warm_started_finite_gibbs_chain_preserves_posterior_draws():
    model = GaussianPrior(torch.tensor([[0.04, 0.012], [0.012, 0.07]], dtype=torch.float64))
    op = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.0]], dtype=torch.float64))
    y = torch.tensor([0.15 + 0.05j, 0j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    cfg = DAPSConfig(tau=0.05, num_steps=3, langevin_steps=2, langevin_step=0.05, cg_rtol=1e-10)
    prior_sigma = model.sigma_min if cfg.prior_sigma is None else cfg.prior_sigma
    prior_covariance = model.covariance + prior_sigma**2 * torch.eye(2, dtype=torch.float64)
    covariance = torch.linalg.inv(torch.linalg.inv(prior_covariance) + op.matrix.mT @ op.matrix / cfg.tau**2)
    mean = covariance.to(y) @ op.matrix.mT.to(y) @ y.flatten() / cfg.tau**2
    generator = torch.Generator().manual_seed(91)
    noise = torch.complex(
        torch.randn(8192, 2, generator=generator, dtype=torch.float64), torch.randn(8192, 2, generator=generator, dtype=torch.float64)
    )
    initial = (noise @ torch.linalg.cholesky(covariance).to(y).mT + mean).reshape(8192, 1, 1, 1, 2)
    draws = DAPSSampler(model, cfg).sample(
        op,
        y.expand(8192, -1, -1, -1, -1),
        1,
        generator=generator,
        initial=initial,
    )
    assert_gaussian_posterior(draws, prior_covariance, op, y, cfg.tau)


@pytest.mark.parametrize("initial", [torch.zeros(2), torch.full((1, 1, 1, 1, 2), float("nan"))])
def test_daps_rejects_invalid_initial_image(initial):
    y = torch.zeros(1, 1, 1, 1, 2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="Initial image"):
        DAPSSampler(GaussianPrior(), DAPSConfig(tau=0.05)).sample(MatrixOperator(torch.eye(2)), y, 1, initial=initial)


def test_daps_is_invariant_to_measurement_units_and_retains_randomness():
    model = GaussianPrior()
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    y = torch.full((2, 1, 1, 1, 2), 0.15 + 0.05j, dtype=torch.complex128)
    cfg = DAPSConfig(tau=0.05, num_steps=3, langevin_steps=3, cg_rtol=1e-10)
    first = DAPSSampler(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(8))
    scaled = DAPSSampler(model, replace(cfg, tau=cfg.tau * 10)).sample(
        MatrixOperator(matrix * 10), y * 10, 1, generator=torch.Generator().manual_seed(8)
    )
    repeated = DAPSSampler(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(8))
    other = DAPSSampler(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(first, scaled, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, other)


@pytest.mark.parametrize("tau", [0.0, -1.0, float("nan"), float("inf")])
def test_daps_requires_physical_measurement_noise(tau):
    with pytest.raises(ValueError):
        DAPSConfig(tau=tau)


def test_daps_rejects_unsupported_score_extrapolation():
    cfg = DAPSConfig(tau=0.01, prior_sigma=0.00001)
    y = torch.zeros(1, 1, 1, 1, 2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="Prior score"):
        DAPSSampler(GaussianPrior(), cfg).sample(MatrixOperator(torch.eye(2)), y, 1)
