"""Augmentation pipeline for MRI data."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import AugmentationConfig


class AugmentationPipeline:
    """Pipeline for applying augmentations to MRI images and masks."""

    def __init__(self, config: AugmentationConfig):
        """Initialize augmentation pipeline with configuration.

        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.rng = torch.Generator()

    def __call__(
        self, image: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply augmentations to image and mask.

        Args:
            image: Image tensor of shape (H, W)
            mask: Mask tensor of shape (H, W)

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        aug_image = image.clone()
        aug_mask = mask.clone()

        # Flip augmentation (50% probability)
        if torch.rand(1, generator=self.rng).item() < self.config.flip_prob:
            aug_image = torch.flip(aug_image, dims=[-1])
            aug_mask = torch.flip(aug_mask, dims=[-1])

        # Geometric augmentation (30% probability)
        if torch.rand(1, generator=self.rng).item() < self.config.geometric_prob:
            aug_image, aug_mask = self._apply_geometric_transform(
                aug_image, aug_mask
            )

        # Contrast/Gamma augmentation (30% probability)
        if torch.rand(1, generator=self.rng).item() < self.config.contrast_prob:
            aug_image = self._apply_contrast_transform(aug_image)

        return aug_image, aug_mask

    def _apply_geometric_transform(
        self, image: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply random geometric transformations.

        Args:
            image: Image tensor of shape (H, W)
            mask: Mask tensor of shape (H, W)

        Returns:
            Tuple of (transformed_image, transformed_mask)
        """
        # Add batch and channel dimensions for affine_grid
        img_batch = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_batch = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        H, W = image.shape

        # Generate random transformation parameters
        scale = torch.empty(1, generator=self.rng).uniform_(
            *self.config.scale_range
        ).item()
        angle = torch.empty(1, generator=self.rng).uniform_(
            *self.config.rotation_range
        ).item() * (torch.pi / 180.0)  # Convert to radians
        shear_x = torch.empty(1, generator=self.rng).uniform_(
            *self.config.shear_range
        ).item()
        shear_y = torch.empty(1, generator=self.rng).uniform_(
            *self.config.shear_range
        ).item()

        # Create affine transformation matrix
        # Center of image
        cx, cy = W / 2.0, H / 2.0

        # Translation to center
        T1 = torch.tensor(
            [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]],
            dtype=image.dtype,
            device=image.device,
        )

        # Scaling
        S = torch.tensor(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=image.dtype,
            device=image.device,
        )

        # Rotation
        cos_a = torch.cos(torch.tensor(angle, device=image.device))
        sin_a = torch.sin(torch.tensor(angle, device=image.device))
        R = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=image.dtype,
            device=image.device,
        )

        # Shearing
        Sh = torch.tensor(
            [[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=image.dtype,
            device=image.device,
        )

        # Translation back
        T2 = torch.tensor(
            [[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]],
            dtype=image.dtype,
            device=image.device,
        )

        # Combine transformations: T2 * Sh * R * S * T1
        M = T2 @ Sh @ R @ S @ T1

        # Extract 2x3 matrix for affine_grid
        theta = M[:2, :].unsqueeze(0)  # (1, 2, 3)

        # Create grid
        grid = F.affine_grid(
            theta, img_batch.size(), align_corners=False
        )

        # Apply transformation
        img_transformed = F.grid_sample(
            img_batch.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        mask_transformed = F.grid_sample(
            mask_batch.float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )

        # Remove batch and channel dimensions
        return img_transformed.squeeze(), mask_transformed.squeeze()

    def _apply_contrast_transform(self, image: Tensor) -> Tensor:
        """Apply random contrast/gamma transformation.

        Uses a monotonic power-law mapping with randomized exponent.

        Args:
            image: Image tensor of shape (H, W)

        Returns:
            Transformed image tensor
        """
        # Normalize to [0, 1] range
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            img_norm = (image - img_min) / (img_max - img_min)
        else:
            return image

        # Generate random gamma value
        # Strength controls the deviation from 1.0
        gamma_min = 1.0 - self.config.contrast_strength
        gamma_max = 1.0 + self.config.contrast_strength
        gamma = torch.empty(1, generator=self.rng).uniform_(
            gamma_min, gamma_max
        ).item()

        # Apply power-law transformation
        img_gamma = torch.pow(img_norm, gamma)

        # Denormalize back to original range
        img_transformed = img_gamma * (img_max - img_min) + img_min

        return img_transformed

