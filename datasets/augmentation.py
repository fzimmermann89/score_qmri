"""Augmentation pipeline for MRI data."""

import kornia.augmentation as K
import torch
from torch import Tensor

from .config import AugmentationConfig


class AugmentationPipeline:
    """Spatial and intensity augmentations for MRI images."""

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
        self._hflip = K.RandomHorizontalFlip(p=0.5)

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply augmentations to image (H, W) and mask (H, W)."""
        shift_x = torch.randint(-10, 11, (1,)).item()
        shift_y = torch.randint(-10, 11, (1,)).item()
        image = torch.roll(image, shifts=(shift_y, shift_x), dims=(0, 1))
        mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(0, 1))

        img_4d = image[None, None]
        mask_4d = mask[None, None].float()
        image = self._hflip(img_4d)[0, 0]
        mask = self._hflip(mask_4d, params=self._hflip._params)[0, 0].bool()

        if torch.rand(1).item() < self.config.geometric_prob:
            image, mask = self._apply_affine(image, mask)

        if torch.rand(1).item() < self.config.contrast_prob:
            image = self._apply_gamma(image)

        return image, mask

    def _apply_affine(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Random affine (scale, rotate, shear, translate)."""
        img_4d = image[None, None]
        mask_4d = mask[None, None].float()
        img_out = self._affine_img(img_4d)
        mask_out = self._affine_mask(mask_4d, params=self._affine_img._params)
        return img_out[0, 0], mask_out[0, 0].bool()

    def _apply_gamma(self, image: Tensor) -> Tensor:
        """Random gamma correction."""
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
