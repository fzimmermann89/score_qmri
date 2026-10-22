"""Check the published Gaussian DAPS algorithm, including its finite chains."""

from dataclasses import replace

import pytest
import torch

from score_qmri.sampling import DAPSConfig, DAPSSampler, FullScoreDAPSConfig, FullScoreDAPSSampler
from score_qmri.sampling.edm_sampler import karras_sigmas

from .test_dps_sampler import GaussianPrior, MatrixOperator


def test_daps_matches_finite_algorithm_moments():
    # Propagate the actual Gaussian approximation, not the true posterior.
    model = GaussianPrior(torch.tensor([[0.04, 0.012], [0.012, 0.07]], dtype=torch.float64))
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    op = MatrixOperator(matrix)
    y = torch.tensor([0.15 + 0.05j, 0.02 - 0.01j], dtype=torch.complex128)
    cfg = DAPSConfig(tau=0.05, num_steps=4, reverse_steps=3, langevin_steps=8, langevin_step=0.0005)
    grid = karras_sigmas(cfg.num_steps, sigma_min=cfg.sigma_min, sigma_max=model.sigma_max, device=model.device)
    identity = torch.eye(2, dtype=torch.float64)
    mean = torch.zeros(2, dtype=torch.complex128)
    covariance = model.sigma_max**2 * identity
    for i, sigma in enumerate(grid.double()):
        pf = karras_sigmas(cfg.reverse_steps, sigma_min=cfg.reverse_sigma_min, sigma_max=float(sigma), device=model.device).double()
        transport = identity.clone()
        for current, following in zip(pf, torch.cat([pf[1:], pf.new_zeros(1)]), strict=True):
            gain = torch.linalg.solve(model.covariance + current**2 * identity, model.covariance)
            transport = (identity + (following - current) / current * (identity - gain)) @ transport
        center_mean = transport.to(mean) @ mean
        center_cov = transport @ covariance @ transport.mT
        step = cfg.langevin_step * (1 + i / cfg.num_steps * (cfg.langevin_step_min_ratio - 1))
        precision = identity / sigma**2 + matrix.mT @ matrix / cfg.tau**2
        transition = identity - step * precision
        gain = identity.clone()
        innovation_cov = torch.zeros_like(identity)
        for _ in range(cfg.langevin_steps):
            innovation_cov += 2 * step * gain @ gain.mT
            gain = transition @ gain
        response = torch.linalg.solve(precision, identity - gain)
        center_gain = gain + response / sigma**2
        mean = center_gain.to(mean) @ center_mean + (response @ matrix.mT).to(mean) @ y / cfg.tau**2
        covariance = center_gain @ center_cov @ center_gain.mT + innovation_cov
        if i < cfg.num_steps - 1:
            covariance += grid[i + 1].double() ** 2 * identity
    draws = DAPSSampler(model, cfg).sample(
        op, y.reshape(1, 1, 1, 1, 2).expand(4096, -1, -1, -1, -1), 1, generator=torch.Generator().manual_seed(73)
    )[:, 0, 0, 0]
    for part, target in [(draws.real, mean.real), (draws.imag, mean.imag)]:
        mean_se = (covariance.diag() / len(draws)).sqrt()
        covariance_se = ((covariance.diag()[:, None] * covariance.diag()[None, :] + covariance.square()) / (len(draws) - 1)).sqrt()
        assert ((part.mean(0) - target).abs() / mean_se).max() < 5
        assert ((torch.cov(part.mT) - covariance).abs() / covariance_se).max() < 5
    assert not draws.requires_grad


@pytest.mark.parametrize(
    "sampler_type,cfg",
    [
        (DAPSSampler, DAPSConfig(tau=0.05, num_steps=3, reverse_steps=2, langevin_steps=3)),
        (FullScoreDAPSSampler, FullScoreDAPSConfig(tau=0.05, num_steps=3, reverse_steps=2, langevin_steps=3, cg_rtol=1e-10)),
    ],
)
def test_daps_measurement_units_and_randomness(sampler_type, cfg):
    model = GaussianPrior()
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    y = torch.full((2, 1, 1, 1, 2), 0.15 + 0.05j, dtype=torch.complex128)
    first = sampler_type(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(8))
    scaled = sampler_type(model, replace(cfg, tau=cfg.tau * 10)).sample(
        MatrixOperator(matrix * 10), y * 10, 1, generator=torch.Generator().manual_seed(8)
    )
    repeated = sampler_type(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(8))
    other = sampler_type(model, cfg).sample(MatrixOperator(matrix), y, 1, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(first, scaled, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not torch.equal(first, other)


def test_daps_requires_physical_measurement_noise():
    with pytest.raises(ValueError):
        DAPSConfig(tau=0.0)
