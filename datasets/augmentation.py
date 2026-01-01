"""Augmentation pipeline for MRI data."""

import math

import kornia.augmentation as K
import torch
from einops import einsum
from torch import Tensor

from .config import AugmentationConfig


class AugmentationPipeline:
    """Spatial and intensity augmentations for MRI images.

    Supports both real-valued (magnitude) and complex-valued images.
    For complex data, applies phase augmentation simulating B0 inhomogeneity.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self._affine_img = K.RandomAffine(
            degrees=config.rotation_range,
            translate=(0.02, 0.02),
            scale=config.scale_range,
            shear=config.shear_range,
            resample="bilinear",
            p=1.0,
        )
        self._affine_mask = K.RandomAffine(
            degrees=config.rotation_range,
            translate=(0.02, 0.02),
            scale=config.scale_range,
            shear=config.shear_range,
            resample="nearest",
            p=1.0,
        )
        self._hflip = K.RandomHorizontalFlip(p=config.flip_prob)

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply augmentations to image and mask.

        Parameters
        ----------
        image
            Image tensor of shape (H, W). Can be real (float) or complex.
        mask
            Boolean mask of shape (H, W).

        Returns
        -------
        Augmented (image, mask) with same shapes.
        """
        is_complex = image.is_complex()

        if is_complex:
            real, imag = image.real, image.imag
            if torch.rand(1).item() < self.config.phase_prob:
                real, imag = self._apply_phase(real, imag)
            # Apply spatial augmentations to both real and imag
            real, mask = self._apply_spatial(real, mask)
            imag, _ = self._apply_spatial(imag, mask.clone())
            image = torch.complex(real, imag)
        else:
            image, mask = self._apply_spatial(image, mask)
            # Apply contrast only to real-valued images
            if torch.rand(1).item() < self.config.contrast_prob:
                image = self._apply_gamma(image)

        return image, mask

    def _apply_spatial(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply spatial augmentations (shift, flip, affine)."""
        # Random shift
        if torch.rand(1).item() < self.config.roll_prob:
            shift_min, shift_max = self.config.shift_range
            shift_x: int = int(torch.randint(shift_min, shift_max + 1, (1,)).item())
            shift_y: int = int(torch.randint(shift_min, shift_max + 1, (1,)).item())
            shifts: list[int] = [shift_y, shift_x]
            image = torch.roll(image, shifts=shifts, dims=(0, 1))
            mask = torch.roll(mask, shifts=shifts, dims=(0, 1))

        # Random horizontal flip (probability handled by K.RandomHorizontalFlip)
        img_4d = image[None, None]
        mask_4d = mask[None, None].float()
        image = self._hflip(img_4d)[0, 0]
        mask = self._hflip(mask_4d, params=self._hflip._params)[0, 0].bool()

        # Random affine
        if torch.rand(1).item() < self.config.geometric_prob:
            image, mask = self._apply_affine(image, mask)

        return image, mask

    def _apply_affine(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Random affine (scale, rotate, shear, translate)."""
        img_4d = image[None, None]
        mask_4d = mask[None, None].float()
        img_out = self._affine_img(img_4d)
        mask_out = self._affine_mask(mask_4d, params=self._affine_img._params)
        return img_out[0, 0], mask_out[0, 0].bool()

    def _apply_gamma(self, image: Tensor) -> Tensor:
        """Random gamma correction for contrast augmentation."""
        img_min, img_max = image.min(), image.max()
        if img_max <= img_min:
            return image
        img_norm = (image - img_min) / (img_max - img_min)
        gamma = (
            torch.empty(1)
            .uniform_(
                1.0 - self.config.contrast_strength,
                1.0 + self.config.contrast_strength,
            )
            .item()
        )
        return img_norm.pow(gamma) * (img_max - img_min) + img_min

    def _apply_phase(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        """Apply random phase rotation simulating B0 inhomogeneity.

        Uses polynomial phase field: φ(x,y) = a + bx + cy + dxy + ex² + fy²
        """
        H, W = real.shape
        device, dtype = real.device, real.dtype

        # Generate normalized coordinates
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        # Random coefficients: mean phase + spatial variation
        coeffs = torch.empty(6, device=device, dtype=dtype)
        coeffs[0].uniform_(0.0, 2.0 * math.pi)  # Mean phase
        coeffs[1:].uniform_(-0.3, 0.3)  # Spatial variation

        # Polynomial basis and phase field
        basis = torch.stack([torch.ones_like(X), X, Y, X * Y, X.pow(2), Y.pow(2)])
        phase = einsum(coeffs, basis, "c, c h w -> h w")

        # Apply rotation: (real + i*imag) * exp(i*phase)
        cos_p, sin_p = phase.cos(), phase.sin()
        aug_real = real * cos_p - imag * sin_p
        aug_imag = real * sin_p + imag * cos_p

        return aug_real, aug_imag
