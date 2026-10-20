"""Tensor layout and batched linear solves shared by posterior samplers."""

from collections.abc import Callable

import torch
from torch import Tensor


def to_complex(x: Tensor) -> Tensor:
    """Convert real (B,2,H,W) coordinates to complex (B,1,1,H,W)."""
    return torch.complex(x[:, 0], x[:, 1]).unsqueeze(1).unsqueeze(1)


def to_2ch(x: Tensor) -> Tensor:
    """Convert complex (B,1,1,H,W) coordinates to real (B,2,H,W)."""
    return torch.view_as_real(x.squeeze(1).squeeze(1)).moveaxis(-1, 1).contiguous()


def normal_like(x: Tensor, generator: torch.Generator | None) -> Tensor:
    """Independent unit Gaussian noise in each real coordinate."""
    noise = torch.randn((*x.shape, 2), device=x.device, dtype=x.real.dtype, generator=generator)
    return torch.view_as_complex(noise)


@torch.no_grad()
def conjugate_gradient(matvec: Callable[[Tensor], Tensor], rhs: Tensor, max_iter: int, rtol: float) -> Tensor:
    """Solve independent positive-definite real-linear systems per batch."""

    def inner(x: Tensor, y: Tensor) -> Tensor:
        return (x.conj() * y).real.flatten(1).sum(1)

    expand = (rhs.shape[0],) + (1,) * (rhs.ndim - 1)
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    squared = inner(residual, residual)
    threshold = rtol**2 * squared
    active = squared > threshold
    for _ in range(max_iter):
        if not active.any():
            residual = rhs - matvec(x)
            squared = inner(residual, residual)
            active = squared > threshold
            if not active.any():
                break
            direction = residual.clone()
        product = matvec(direction)
        curvature = inner(direction, product)
        if not product.isfinite().all() or (active & (curvature <= 0)).any():
            raise RuntimeError("CG requires a finite positive-definite system; check the operator and adjoint")
        alpha = torch.where(active, squared / torch.where(active, curvature, 1), 0)
        x += alpha.reshape(expand) * direction
        residual -= alpha.reshape(expand) * product
        following = inner(residual, residual)
        next_active = following > threshold
        beta = torch.where(next_active, following / torch.where(active, squared, 1), 0)
        direction = torch.where(next_active.reshape(expand), residual + beta.reshape(expand) * direction, 0)
        squared, active = following, next_active
    residual = rhs - matvec(x)
    error = inner(residual, residual)
    if not x.isfinite().all() or (error > threshold).any():
        raise RuntimeError(
            f"CG did not converge (relative residual {(error / inner(rhs, rhs).clamp_min(1e-30)).max().sqrt().item():.3g}); check the operator/adjoint"
        )
    return x
