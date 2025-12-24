"""Phase augmentation for complex MRI data using polynomial-based slowly varying phase."""

import math

import torch
from einops import einsum, rearrange
from torch import Tensor


class PhaseAugmentation:
    """Applies random 2D phase rotation to complex images.

    Uses a low-order polynomial to generate smoothly varying spatial phase,
    simulating realistic B0 inhomogeneity and receive coil phase.

    Parameters
    ----------
    mean_phase_range : tuple[float, float]
        Range for random mean phase offset in radians (default: [0, 2π]).
    poly_coeff_range : tuple[float, float]
        Range for polynomial coefficients controlling spatial variation.
    prob : float
        Probability of applying augmentation.
    """

    def __init__(
        self,
        mean_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi),
        poly_coeff_range: tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.5,
    ) -> None:
        self.mean_phase_range = mean_phase_range
        self.poly_coeff_range = poly_coeff_range
        self.prob = prob

    def __call__(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        """Apply random phase rotation to complex image.

        Parameters
        ----------
        real : Tensor
            Real part of shape (H, W) or (B, H, W).
        imag : Tensor
            Imaginary part of shape (H, W) or (B, H, W).

        Returns
        -------
        tuple[Tensor, Tensor]
            Augmented (real, imag) with same shapes.
        """
        if torch.rand(1).item() > self.prob:
            return real, imag

        squeeze_batch = real.ndim == 2
        if squeeze_batch:
            real = rearrange(real, "h w -> 1 h w")
            imag = rearrange(imag, "h w -> 1 h w")

        B, H, W = real.shape
        phase = self._generate_phase_field(B, H, W, real.device, real.dtype)

        cos_p, sin_p = phase.cos(), phase.sin()
        aug_real = real * cos_p - imag * sin_p
        aug_imag = real * sin_p + imag * cos_p

        if squeeze_batch:
            return aug_real.squeeze(0), aug_imag.squeeze(0)
        return aug_real, aug_imag

    def _generate_phase_field(
        self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Generate polynomial phase fields: a + bx + cy + dxy + ex² + fy².

        Coordinates are normalized to [-1, 1] for numerical stability.
        """
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        # Random coefficients: first is mean phase, rest are spatial variation
        coeffs = torch.empty(B, 6, device=device, dtype=dtype)
        coeffs[:, 0].uniform_(*self.mean_phase_range)
        coeffs[:, 1:].uniform_(*self.poly_coeff_range)

        # Polynomial basis: 1, X, Y, XY, X², Y²
        basis = torch.stack([torch.ones_like(X), X, Y, X * Y, X.pow(2), Y.pow(2)])
        return einsum(coeffs, basis, "b c, c h w -> b h w")
