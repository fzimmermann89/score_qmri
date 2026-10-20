"""Tests for FID utilities."""

import torch

from score_qmri.metrics.fid import compute_statistics_torch, frechet_distance


def test_compute_statistics_torch_shape() -> None:
    features = torch.randn(100, 64, dtype=torch.float64)
    mu, sigma = compute_statistics_torch(features)
    assert mu.shape == (64,)
    assert sigma.shape == (64, 64)


def test_compute_statistics_torch_values() -> None:
    # Known distribution: standard normal features
    torch.manual_seed(42)
    features = torch.randn(10000, 32, dtype=torch.float64)
    mu, sigma = compute_statistics_torch(features)
    # Mean should be close to zero
    assert torch.allclose(mu, torch.zeros(32, dtype=torch.float64), atol=0.1)
    # Covariance should be close to identity
    assert torch.allclose(sigma, torch.eye(32, dtype=torch.float64), atol=0.1)


def test_frechet_distance_identical() -> None:
    mu = torch.zeros(3, dtype=torch.float64)
    sigma = torch.eye(3, dtype=torch.float64)
    fid = frechet_distance(mu, sigma, mu, sigma)
    assert torch.isclose(fid, torch.tensor(0.0, dtype=fid.dtype))


def test_frechet_distance_mean_shift() -> None:
    mu1 = torch.zeros(3, dtype=torch.float64)
    mu2 = torch.ones(3, dtype=torch.float64)
    sigma1 = torch.eye(3, dtype=torch.float64)
    sigma2 = torch.eye(3, dtype=torch.float64)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    assert torch.isclose(fid, torch.tensor(3.0, dtype=fid.dtype))
