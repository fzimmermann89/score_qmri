"""Augmentation pipeline for MRI data."""

import math

import torch
import torchvision.transforms.functional as TF
from torch import Tensor

from .config import AugmentationConfig


class AugmentationPipeline:
    """Pipeline for spatial and intensity augmentations.

    Applies flip, geometric (scale/rotate/shear), and contrast augmentations
    with configurable probabilities.

    Parameters
    ----------
    config : AugmentationConfig
        Augmentation configuration.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply augmentations to image and mask.

        Parameters
        ----------
        image : Tensor
            Image tensor of shape (H, W).
        mask : Tensor
            Mask tensor of shape (H, W).

        Returns
        -------
        tuple[Tensor, Tensor]
            Augmented (image, mask).
        """
        if torch.rand(1).item() < self.config.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if torch.rand(1).item() < self.config.geometric_prob:
            image, mask = self._geometric_transform(image, mask)

        if torch.rand(1).item() < self.config.contrast_prob:
            image = self._contrast_transform(image)

        return image, mask

    def _geometric_transform(
        self, image: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply random affine transformation (scale, rotate, shear)."""
        cfg = self.config

        angle = torch.empty(1).uniform_(*cfg.rotation_range).item()
        scale = torch.empty(1).uniform_(*cfg.scale_range).item()
        shear_x = torch.empty(1).uniform_(*cfg.shear_range).item() * (180 / math.pi)
        shear_y = torch.empty(1).uniform_(*cfg.shear_range).item() * (180 / math.pi)

        # torchvision expects (C, H, W) input
        img_3d = image.unsqueeze(0)
        mask_3d = mask.float().unsqueeze(0)

        img_out = TF.affine(
            img_3d, angle=angle, translate=[0, 0], scale=scale,
            shear=[shear_x, shear_y], interpolation=TF.InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask_out = TF.affine(
            mask_3d, angle=angle, translate=[0, 0], scale=scale,
            shear=[shear_x, shear_y], interpolation=TF.InterpolationMode.NEAREST,
            fill=0.0,
        )

        return img_out.squeeze(0), mask_out.squeeze(0).bool()

    def _contrast_transform(self, image: Tensor) -> Tensor:
        """Apply random gamma correction."""
        img_min, img_max = image.min(), image.max()
        if img_max <= img_min:
            return image

        img_norm = (image - img_min) / (img_max - img_min)
        gamma = torch.empty(1).uniform_(
            1.0 - self.config.contrast_strength,
            1.0 + self.config.contrast_strength,
        ).item()

        return img_norm.pow(gamma) * (img_max - img_min) + img_min
