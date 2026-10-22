"""Analytic posterior controls for experimental full-score DAPS."""

import torch

from score_qmri.sampling import FullScoreDAPSConfig, FullScoreDAPSSampler

from .test_dps_sampler import GaussianPrior, MatrixOperator, assert_gaussian_posterior


def test_full_score_daps_recovers_gaussian_posterior_and_cross_covariance():
    covariance = torch.tensor([[0.04, 0.012], [0.012, 0.07]], dtype=torch.float64)
    model = GaussianPrior(covariance)
    op = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.0]], dtype=torch.float64))
    y = torch.tensor([0.15 + 0.05j, 0j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    cfg = FullScoreDAPSConfig(tau=0.05, num_steps=8, reverse_steps=4, cg_rtol=1e-10)
    draws = FullScoreDAPSSampler(model, cfg).sample(op, y.expand(2048, -1, -1, -1, -1), 1, generator=torch.Generator().manual_seed(73))
    prior_sigma = model.sigma_min
    prior_covariance = model.covariance + prior_sigma**2 * torch.eye(2, dtype=torch.float64)
    assert_gaussian_posterior(draws, prior_covariance, op, y, cfg.tau)


def test_warm_started_finite_gibbs_chain_preserves_posterior_draws():
    model = GaussianPrior(torch.tensor([[0.04, 0.012], [0.012, 0.07]], dtype=torch.float64))
    op = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.0]], dtype=torch.float64))
    y = torch.tensor([0.15 + 0.05j, 0j], dtype=torch.complex128).reshape(1, 1, 1, 1, 2)
    cfg = FullScoreDAPSConfig(tau=0.05, num_steps=3, langevin_steps=2, langevin_step=0.05, cg_rtol=1e-10)
    prior_sigma = model.sigma_min
    prior_covariance = model.covariance + prior_sigma**2 * torch.eye(2, dtype=torch.float64)
    covariance = torch.linalg.inv(torch.linalg.inv(prior_covariance) + op.matrix.mT @ op.matrix / cfg.tau**2)
    mean = covariance.to(y) @ op.matrix.mT.to(y) @ y.flatten() / cfg.tau**2
    generator = torch.Generator().manual_seed(91)
    noise = torch.complex(
        torch.randn(2048, 2, generator=generator, dtype=torch.float64), torch.randn(2048, 2, generator=generator, dtype=torch.float64)
    )
    initial = (noise @ torch.linalg.cholesky(covariance).to(y).mT + mean).reshape(2048, 1, 1, 1, 2)
    draws = FullScoreDAPSSampler(model, cfg).sample(
        op,
        y.expand(2048, -1, -1, -1, -1),
        1,
        generator=generator,
        initial=initial,
    )
    assert_gaussian_posterior(draws, prior_covariance, op, y, cfg.tau)
