"""Gaussian conditionals for linear inverse problems in complex MRI coordinates."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from .protocols import LinearOperatorProtocol


@dataclass
class CGInfo:
    """Convergence diagnostics for independent solves along the batch axis."""

    iterations: int
    relative_residual: Tensor


def _inner(x: Tensor, y: Tensor) -> Tensor:
    return (x.conj() * y).real.flatten(1).sum(1)


def complex_normal_like(x: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Draw complex noise with independent unit-variance real and imaginary parts."""
    real = torch.randn(x.shape, dtype=x.real.dtype, device=x.device, generator=generator)
    imag = torch.randn(x.shape, dtype=x.real.dtype, device=x.device, generator=generator)
    return torch.complex(real, imag)


@torch.no_grad()
def sample_linear_gaussian(
    operator: LinearOperatorProtocol,
    measurement: Tensor,
    center: Tensor,
    prior_std: float,
    noise_std: float,
    *,
    generator: torch.Generator | None = None,
    max_iter: int = 200,
    rtol: float = 1e-5,
    atol: float = 0.0,
    mean_only: bool = False,
) -> tuple[Tensor, CGInfo]:
    """Sample the Gaussian DAPS inner conditional using perturb-and-solve.

    Parameters
    ----------
    operator
        Batch-independent complex linear operator with an accurate adjoint.
    measurement
        Observed, optionally prewhitened data with batch on axis zero.
    center
        Conditional prior mean, in the same image units as the operator input.
    prior_std, noise_std
        Standard deviations **per real coordinate** of the prior and measurement
        likelihood. Thus E[|complex noise|²] = 2 * noise_std².
    generator
        Generator on the same device as the input tensors.
    max_iter, rtol, atol
        CG iteration limit and per-batch residual tolerances. Nonconvergence or
        nonpositive curvature raises an error instead of returning a bad sample.
    mean_only
        Return the conditional mean instead of a random draw, for ridge/CG
        baselines. This is not posterior sampling.

    Returns
    -------
    sample, diagnostics
        Gaussian draw with mean Q⁻¹b and covariance Q⁻¹ in real coordinates,
        up to solve tolerance, where Q = I/prior_std² + A* A/noise_std².
        The outer DAPS prior approximation is not made exact by this solve.
    """
    if not all(isfinite(v) and v > 0 for v in (prior_std, noise_std)):
        raise ValueError("prior_std and noise_std must be finite and positive")
    if max_iter < 1 or not isfinite(rtol) or not isfinite(atol) or rtol < 0 or atol < 0 or rtol + atol == 0:
        raise ValueError("CG needs positive max_iter and a positive finite tolerance")
    if not center.is_complex() or not measurement.is_complex():
        raise ValueError("center and measurement must be complex tensors")
    if center.ndim < 2 or center.shape[0] != measurement.shape[0]:
        raise ValueError("center and measurement need matching batch axes")
    if center.device != measurement.device or center.dtype != measurement.dtype:
        raise ValueError("center and measurement must share dtype and device")
    if not center.isfinite().all() or not measurement.isfinite().all():
        raise ValueError("center and measurement must be finite")

    prior_precision, noise_precision = prior_std**-2, noise_std**-2

    def precision(x: Tensor) -> Tensor:
        return prior_precision * x + noise_precision * operator.H(operator(x)[0])[0]

    rhs = prior_precision * center + noise_precision * operator.H(measurement)[0]
    if not mean_only:
        rhs = rhs + complex_normal_like(center, generator) / prior_std
        rhs = rhs + operator.H(complex_normal_like(measurement, generator))[0] / noise_std
    if rhs.shape != center.shape or not rhs.isfinite().all():
        raise ValueError("operator adjoint must produce finite tensors matching center")

    expand = (center.shape[0],) + (1,) * (center.ndim - 1)
    x = torch.zeros_like(center)
    residual = rhs.clone()
    direction = residual.clone()
    squared_norm = _inner(residual, residual)
    rhs_norm = squared_norm.sqrt()
    threshold = atol + rtol * rhs_norm
    active = rhs_norm > threshold
    iterations = 0
    for iterations in range(1, max_iter + 1):
        if not active.any():
            iterations -= 1
            break
        product = precision(direction)
        curvature = _inner(direction, product)
        if not product.isfinite().all() or (active & (curvature <= 0)).any():
            raise RuntimeError("CG encountered nonfinite or nonpositive curvature; check operator/adjoint")
        alpha = torch.where(active, squared_norm / torch.where(active, curvature, 1), 0)
        x = x + alpha.reshape(expand) * direction
        residual = residual - alpha.reshape(expand) * product
        next_squared_norm = _inner(residual, residual)
        next_active = next_squared_norm.sqrt() > threshold
        beta = torch.where(next_active, next_squared_norm / torch.where(active, squared_norm, 1), 0)
        direction = residual + beta.reshape(expand) * direction
        direction = torch.where(next_active.reshape(expand), direction, 0)
        squared_norm, active = next_squared_norm, next_active

    # Check the actual residual: a small recursive CG residual can hide roundoff.
    true_residual = rhs - precision(x)
    true_norm = _inner(true_residual, true_residual).sqrt()
    relative = true_norm / rhs_norm.clamp_min(torch.finfo(rhs.real.dtype).tiny)
    if not x.isfinite().all() or (true_norm > threshold).any():
        raise RuntimeError(f"Gaussian CG failed after {iterations} iterations (max relative residual {relative.max().item():.3g})")
    return x, CGInfo(iterations=iterations, relative_residual=relative)
