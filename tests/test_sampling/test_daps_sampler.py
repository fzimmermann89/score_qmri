"""Verify DAPS samples its stated conditional target with full preconditioned noise."""

from dataclasses import replace

import pytest
import torch

from score_qmri.sampling import DAPSConfig, DAPSSampler

from .test_dps_sampler import GaussianPrior, MatrixOperator, assert_gaussian_posterior


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


def test_finite_conditional_kernel_preserves_posterior_draws():
    model = GaussianPrior(torch.tensor([[0.04, 0.012], [0.012, 0.07]], dtype=torch.float64))
    op = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.0]], dtype=torch.float64))
    y = torch.tensor([0.15 + 0.05j, 0j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    cfg = DAPSConfig(tau=0.05, langevin_steps=2, langevin_step=0.05, cg_rtol=1e-10)
    prior_sigma = model.sigma_min if cfg.prior_sigma is None else cfg.prior_sigma
    prior_covariance = model.covariance + prior_sigma**2 * torch.eye(2, dtype=torch.float64)
    covariance = torch.linalg.inv(torch.linalg.inv(prior_covariance) + op.matrix.mT @ op.matrix / cfg.tau**2)
    mean = covariance.to(y) @ op.matrix.mT.to(y) @ y.flatten() / cfg.tau**2
    generator = torch.Generator().manual_seed(91)
    noise = torch.complex(
        torch.randn(8192, 2, generator=generator, dtype=torch.float64), torch.randn(8192, 2, generator=generator, dtype=torch.float64)
    )
    initial = (noise @ torch.linalg.cholesky(covariance).to(y).mT + mean).reshape(8192, 1, 1, 1, 2)
    epsilon = torch.complex(
        torch.randn(initial.shape, generator=generator, dtype=torch.float64),
        torch.randn(initial.shape, generator=generator, dtype=torch.float64),
    )
    sigma = 0.5
    draws = DAPSSampler(model, cfg)._conditional(
        initial + sigma * epsilon,
        torch.ones(8192, dtype=torch.long),
        op,
        y.expand(8192, -1, -1, -1, -1),
        sigma,
        generator,
        initial=initial,
    )
    assert_gaussian_posterior(draws, prior_covariance, op, y, cfg.tau)


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
