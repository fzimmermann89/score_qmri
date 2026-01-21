"""Augmentation pipeline for MRI data."""

import math

import kornia.augmentation as K
import torch
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

    def __call__(self, x: Tensor) -> Tensor:
        """Apply random phase rotation to complex image.

        Parameters
        ----------
        x
            Complex tensor of shape (B, H, W).

        Returns
        -------
        Tensor
            Phase-augmented complex tensor with same shape.
        """
        if torch.rand(()) > self.prob:
            return x
        b, h, w = x.shape
        phase = self._generate_phase_field(b, h, w, x.device, x.real.dtype)
        return x * torch.polar(torch.ones_like(phase), phase)

    def _generate_phase_field(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Generate polynomial phase fields: a + bx + cy + dxy + ex² + fy².

        Coordinates are normalized to [-1, 1] for numerical stability.
        """
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        coeffs = torch.empty(b, 6, device=device, dtype=dtype)
        coeffs[:, 0].uniform_(*self.mean_phase_range)
        coeffs[:, 1:].uniform_(*self.poly_coeff_range)

        basis = torch.stack([torch.ones_like(xx), xx, yy, xx * yy, xx.square(), yy.square()])  # (6, H, W)
        return (coeffs[:, :, None, None] * basis[None, ...]).sum(dim=1)  # (B, H, W)


class AugmentationPipeline:
    """Spatial and intensity augmentations for MRI images.

    Supports both real-valued (magnitude) and complex-valued images.
    For complex data, applies phase augmentation simulating B0 inhomogeneity.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        """
        Parameters
        ----------
        config
            Augmentation configuration specifying probabilities and ranges.
        """
        self.config = config
        self._affine = K.RandomAffine(
            degrees=config.rotation_range,
            translate=(0.02, 0.02),
            scale=config.scale_range,
            shear=config.shear_range,
            resample="bilinear",
            p=1.0,
        )
        self._scale = K.RandomAffine(
            degrees=0.0,
            scale=config.scale_range,
            p=1.0,
        )
        self._hflip = K.RandomHorizontalFlip(p=config.flip_prob)
        self.phase_aug = (
            PhaseAugmentation(
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
            Image tensor of shape (H, W) or (B, H, W). Can be real or complex.
        mask
            Boolean mask of shape (H, W) or (B, H, W).

        Returns
        -------
        tuple[Tensor, Tensor]
            Augmented (image, mask) with same shapes as input.
        """
        squeeze = image.ndim == 2
        is_complex = image.is_complex()

        # atleast_3d adds trailing dim, we need leading batch dim
        img_bhw = image.unsqueeze(0) if squeeze else image
        mask_bhw = mask.unsqueeze(0) if squeeze else mask

        if is_complex:
            if self.phase_aug is not None:
                img_bhw = self.phase_aug(img_bhw)
            img_bchw = torch.moveaxis(torch.view_as_real(img_bhw), -1, 1).contiguous()
        else:
            img_bchw = img_bhw.unsqueeze(1)

        mask_bchw = mask_bhw.unsqueeze(1).float()

        img_bchw, mask_bchw = self._apply_spatial(img_bchw, mask_bchw)

        if is_complex:
            img_bhw = torch.view_as_complex(torch.moveaxis(img_bchw, 1, -1).contiguous())
        else:
            img_bhw = img_bchw[:, 0]
            if torch.rand(()) < self.config.contrast_prob:
                img_bhw = self._apply_gamma(img_bhw)

        mask_bhw = mask_bchw[:, 0] > 0.9

        if squeeze:
            return img_bhw[0], mask_bhw[0]
        return img_bhw, mask_bhw

    def _apply_spatial(self, img: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply spatial augmentations (shift, flip, affine).

        Parameters
        ----------
        img
            Image tensor of shape (B, C, H, W).
        mask
            Mask tensor of shape (B, 1, H, W).

        Returns
        -------
        tuple[Tensor, Tensor]
            Augmented (image, mask) tensors.
        """
        if torch.rand(()) < self.config.roll_prob:
            lo, hi = self.config.shift_range
            shifts = torch.randint(lo, hi + 1, (2,)).tolist()
            img = torch.roll(img, shifts=shifts, dims=(-2, -1))
            mask = torch.roll(mask, shifts=shifts, dims=(-2, -1))

        img = self._hflip(img)
        mask = self._hflip(mask, params=self._hflip._params)

        stacked = torch.cat([img, mask], dim=1)
        stacked = self._affine(stacked) if torch.rand(()) < self.config.geometric_prob else self._scale(stacked)

        img, mask = stacked[:, :-1], stacked[:, -1:]
        return img, mask

    def _apply_gamma(self, image: Tensor) -> Tensor:
        """Random gamma correction for contrast augmentation.

        Parameters
        ----------
        image
            Real-valued image tensor of shape (B, H, W).

        Returns
        -------
        Tensor
            Gamma-corrected image with same shape.
        """
        img_min, img_max = image.min(), image.max()
        if (img_max - img_min) < 1e-6:
            return image
        img_norm = (image - img_min) / (img_max - img_min)
        gamma = torch.empty(()).uniform_(
            1.0 - self.config.contrast_strength,
            1.0 + self.config.contrast_strength,
        )
        return img_norm.pow(gamma) * (img_max - img_min) + img_min
