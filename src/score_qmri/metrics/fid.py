"""FID utilities."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor


def compute_statistics_torch(features: Tensor) -> tuple[Tensor, Tensor]:
    """Compute mean and covariance from feature vectors.

    Parameters
    ----------
    features
        Feature tensor of shape (N, D).

    Returns
    -------
    tuple[Tensor, Tensor]
        Mean (D,) and covariance (D, D).
    """
    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / (features.shape[0] - 1)
    return mean, cov


def prepare_inception_input(images_2ch: Tensor, is_complex: Tensor) -> Tensor:
    """Prepare 2-channel MRI images for Inception.

    Parameters
    ----------
    images_2ch
        2-channel tensor of shape (B, 2, H, W) scaled to [0, 1].
    is_complex
        Boolean tensor of shape (B,) indicating complex-valued samples. Complex
        samples use sqrt((real^2 + imag^2) / 2) as magnitude; real samples use
        channel 0 directly.

    Returns
    -------
    Tensor
        3-channel tensor of shape (B, 3, H, W) scaled to [0, 1].
    """
    if images_2ch.ndim != 4 or images_2ch.shape[1] != 2:
        raise ValueError(f"Expected input shape (B, 2, H, W), got {tuple(images_2ch.shape)}")

    is_complex = is_complex.to(dtype=torch.bool, device=images_2ch.device)
    if is_complex.ndim != 1 or is_complex.shape[0] != images_2ch.shape[0]:
        raise ValueError(f"Expected is_complex shape (B,), got {tuple(is_complex.shape)} for B={images_2ch.shape[0]}")

    real = images_2ch[:, 0]
    imag = images_2ch[:, 1]
    mag_complex = torch.sqrt((real**2 + imag**2) / 2.0)
    magnitude = torch.where(is_complex.view(-1, 1, 1), mag_complex, real)
    magnitude = torch.clamp(magnitude, 0.0, 1.0)

    return magnitude.unsqueeze(1).expand(-1, 3, -1, -1)


def frechet_distance(
    mu1: Tensor,
    sigma1: Tensor,
    mu2: Tensor,
    sigma2: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Compute Frechet distance between two Gaussians.

    Parameters
    ----------
    mu1, mu2
        Mean vectors of shape (D,).
    sigma1, sigma2
        Covariance matrices of shape (D, D).
    eps
        Numerical stability epsilon for near-singular products.

    Returns
    -------
    Tensor
        Frechet distance scalar.
    """
    mu1 = mu1.reshape(-1)
    mu2 = mu2.reshape(-1)
    sigma1 = sigma1.reshape(mu1.numel(), mu1.numel())
    sigma2 = sigma2.reshape(mu1.numel(), mu1.numel())

    diff = mu1 - mu2
    covmean = _sqrtm_psd(_symmetrize(sigma1) @ _symmetrize(sigma2), eps=eps)
    tr_covmean = torch.trace(covmean)
    distance = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return torch.clamp(distance, min=0.0)


def _symmetrize(matrix: Tensor) -> Tensor:
    return (matrix + matrix.transpose(-1, -2)) * 0.5


def _sqrtm_psd(matrix: Tensor, eps: float) -> Tensor:
    sym = _symmetrize(matrix)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eigvals = cast(Tensor, eigvals)
    eigvecs = cast(Tensor, eigvecs)
    # Retry with regularization if eigenvalues are non-finite or significantly negative
    if not torch.isfinite(eigvals).all() or eigvals.min() < -eps:
        sym = sym + torch.eye(sym.shape[-1], device=sym.device, dtype=sym.dtype) * eps
        eigvals, eigvecs = torch.linalg.eigh(sym)
        eigvals = cast(Tensor, eigvals)
        eigvecs = cast(Tensor, eigvecs)
    # Clamp small negative values (floating point artifacts) to zero
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_eigs = torch.sqrt(eigvals)
    return (eigvecs * sqrt_eigs) @ eigvecs.transpose(-1, -2)
