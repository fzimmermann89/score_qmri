"""Augmentation pipeline for MRI data."""

import math

import einops
import kornia.augmentation as K
import torch
from einops import rearrange
from torch import Tensor

from .config import AugmentationConfig


class PhaseAugmentation:
    """Applies random 2D phase rotation to complex images.

    Uses a low-order polynomial to generate smoothly varying spatial phase,
    simulating realistic B0 inhomogeneity and receive coil phase.
    """

    def __init__(
        self,
        mean_phase_range: tuple[float, float] = (0.0, 2.0 * math.pi),
        poly_coeff_range: tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        mean_phase_range
            Range for random mean phase offset in radians (default: [0, 2π]).
        poly_coeff_range
            Range for polynomial coefficients controlling spatial variation.
        prob
            Probability of applying augmentation.
        """
        self.mean_phase_range = mean_phase_range
        self.poly_coeff_range = poly_coeff_range
        self.prob = prob

    def __call__(self, real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        """Apply random phase rotation to complex image.

        Parameters
        ----------
        real
            Real part of shape (H, W) or (B, H, W).
        imag
            Imaginary part of shape (H, W) or (B, H, W).

        Returns
        -------
        tuple
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

    def _generate_phase_field(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Generate polynomial phase fields: a + bx + cy + dxy + ex² + fy².

        Coordinates are normalized to [-1, 1] for numerical stability.
        """
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")

        coeffs = torch.empty(B, 6, device=device, dtype=dtype)
        coeffs[:, 0].uniform_(*self.mean_phase_range)
        coeffs[:, 1:].uniform_(*self.poly_coeff_range)

        basis = torch.stack([torch.ones_like(X), X, Y, X * Y, X.pow(2), Y.pow(2)])
        return einops.einsum(coeffs, basis, "b c, c h w -> b h w")


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
        self.phase_aug = (
            PhaseAugmentation(
                mean_phase_range=(0.0, 2.0 * math.pi),
                poly_coeff_range=config.poly_coeff_range,
                prob=config.phase_prob,
            )
            if config.phase_prob > 0
            else None
        )

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply augmentations to image and mask.

        Parameters
        ----------
        image
            Image tensor of shape (H, W) or (B, H, W). Can be real (float) or complex.
        mask
            Boolean mask of shape (H, W) or (B, H, W).

        Returns
        -------
        image : Tensor
            Augmented image with same shape as input.
        mask : Tensor
            Augmented boolean mask with same shape as input.
        """
        is_complex = image.is_complex()

        if is_complex:
            real, imag = image.real.clone(), image.imag.clone()
            orig_ndim = real.ndim  # Save before phase_aug potentially modifies
            if self.phase_aug is not None:
                real, imag = self.phase_aug(real, imag)
            # Stack as channels: (H, W) -> (2, H, W) or (B, H, W) -> (B, 2, H, W)
            stacked = torch.stack([real, imag], dim=0) if orig_ndim == 2 else torch.stack([real, imag], dim=1)
            stacked, mask = self._apply_spatial_complex(stacked, mask)
            # Unstack back
            if orig_ndim == 2:
                real, imag = stacked[0], stacked[1]
            else:
                real, imag = stacked[:, 0], stacked[:, 1]
            image = torch.complex(real, imag)
        else:
            image, mask = self._apply_spatial(image, mask)
            if torch.rand(1).item() < self.config.contrast_prob:
                image = self._apply_gamma(image)

        return image, mask

    def _apply_spatial_complex(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply spatial augmentations to 2-channel image and mask.

        Parameters
        ----------
        image
            Image tensor of shape (2, H, W) or (B, 2, H, W).
        mask
            Mask of shape (H, W) or (B, H, W).
        """
        # Random shift
        if torch.rand(1).item() < self.config.roll_prob:
            shift_min, shift_max = self.config.shift_range
            shift_x: int = int(torch.randint(shift_min, shift_max + 1, (1,)).item())
            shift_y: int = int(torch.randint(shift_min, shift_max + 1, (1,)).item())
            shifts: list[int] = [shift_y, shift_x]
            image = torch.roll(image, shifts=shifts, dims=(-2, -1))
            mask = torch.roll(mask, shifts=shifts, dims=(-2, -1))

        is_2d_input = image.ndim == 3  # (2, H, W) means original was 2D

        # Reshape for kornia: (2, H, W) -> (1, 2, H, W) or keep (B, 2, H, W)
        if is_2d_input:
            img_4d = image.unsqueeze(0)  # (1, 2, H, W)
            mask_4d = mask[None, None].float()  # (1, 1, H, W)
        else:
            img_4d = image  # (B, 2, H, W)
            mask_4d = mask.unsqueeze(1).float()  # (B, 1, H, W)

        # Apply hflip - same transform applied to all channels automatically
        img_out = self._hflip(img_4d)
        mask_out = self._hflip(mask_4d, params=self._hflip._params)

        # Random affine
        if torch.rand(1).item() < self.config.geometric_prob:
            img_out = self._affine_img(img_out)
            mask_out = self._affine_mask(mask_out, params=self._affine_img._params)

        # Reshape back
        if is_2d_input:
            return img_out.squeeze(0), mask_out[0, 0].bool()
        else:
            return img_out, mask_out.squeeze(1).bool()

    def _apply_spatial(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply spatial augmentations (shift, flip, affine)."""
        # Random shift
        if torch.rand(1).item() < self.config.roll_prob:
            shift_min, shift_max = self.config.shift_range
            shift_x: int = int(torch.randint(shift_min, shift_max + 1, (1,)).item())
            shift_y: int = int(torch.randint(shift_min, shift_max + 1, (1,)).item())
            shifts: list[int] = [shift_y, shift_x]
            image = torch.roll(image, shifts=shifts, dims=(-2, -1))
            mask = torch.roll(mask, shifts=shifts, dims=(-2, -1))

        # Reshape for kornia's (B, C, H, W) format
        if image.ndim == 2:
            img_4d = image[None, None]
            mask_4d = mask[None, None].float()
        else:  # image.ndim == 3
            img_4d = image.unsqueeze(1)
            mask_4d = mask.unsqueeze(1).float()

        image_out = self._hflip(img_4d)
        mask_out = self._hflip(mask_4d, params=self._hflip._params)

        # Remove extra dimensions to get back to original shape
        if image.ndim == 2:
            image = image_out[0, 0]
            mask = mask_out[0, 0].bool()
        else:
            image = image_out.squeeze(1)
            mask = mask_out.squeeze(1).bool()

        # Random affine
        if torch.rand(1).item() < self.config.geometric_prob:
            image, mask = self._apply_affine(image, mask)

        return image, mask

    def _apply_affine(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Random affine (scale, rotate, shear, translate)."""
        if image.ndim == 2:
            img_4d = image[None, None]
            mask_4d = mask[None, None].float()
        else:  # image.ndim == 3
            img_4d = image.unsqueeze(1)
            mask_4d = mask.unsqueeze(1).float()

        img_out = self._affine_img(img_4d)
        mask_out = self._affine_mask(mask_4d, params=self._affine_img._params)

        # Remove extra dimensions to get back to original shape
        if image.ndim == 2:
            return img_out[0, 0], mask_out[0, 0].bool()
        else:
            return img_out.squeeze(1), mask_out.squeeze(1).bool()

    def _apply_gamma(self, image: Tensor) -> Tensor:
        """Random gamma correction for contrast augmentation."""
        img_min, img_max = image.min(), image.max()
        if (img_max - img_min) < 1e-6:
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
