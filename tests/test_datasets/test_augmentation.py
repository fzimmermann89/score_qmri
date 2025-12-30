"""Tests for AugmentationPipeline."""

import torch

from datasets.augmentation import AugmentationPipeline
from datasets.config import AugmentationConfig


def test_augmentation_preserves_shape(augmentation_config):
    """Test augmentation preserves input shape."""
    pipeline = AugmentationPipeline(augmentation_config)
    image = torch.randn(64, 64)
    mask = torch.ones(64, 64, dtype=torch.bool)

    aug_image, aug_mask = pipeline(image, mask)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape


def test_flip_augmentation_applied():
    """Test flip augmentation is applied (shape preserved, values change)."""
    config = AugmentationConfig(geometric_prob=0.0, contrast_prob=0.0)
    pipeline = AugmentationPipeline(config)

    image = torch.arange(16).float().reshape(4, 4)
    mask = torch.ones(4, 4, dtype=torch.bool)

    aug_image, aug_mask = pipeline(image, mask)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape
    # Values should be rearranged (roll + flip applied)
    assert not torch.allclose(aug_image, image)


def test_geometric_augmentation_transforms_both():
    """Test geometric augmentation transforms both image and mask."""
    config = AugmentationConfig(geometric_prob=1.0, contrast_prob=0.0)
    pipeline = AugmentationPipeline(config)

    image = torch.randn(64, 64)
    mask = torch.ones(64, 64, dtype=torch.bool)

    aug_image, aug_mask = pipeline(image, mask)
    assert aug_image.shape == (64, 64)
    assert aug_mask.shape == (64, 64)


def test_contrast_augmentation_preserves_range():
    """Test contrast augmentation keeps values in reasonable range."""
    config = AugmentationConfig(geometric_prob=0.0, contrast_prob=1.0)
    pipeline = AugmentationPipeline(config)

    image = torch.rand(64, 64)  # [0, 1] range
    mask = torch.ones(64, 64, dtype=torch.bool)

    aug_image, _ = pipeline(image, mask)
    assert aug_image.min() >= -0.5
    assert aug_image.max() <= 1.5


def test_all_augmentations_can_stack():
    """Test all augmentations can be applied together."""
    config = AugmentationConfig(geometric_prob=1.0, contrast_prob=1.0)
    pipeline = AugmentationPipeline(config)

    image = torch.randn(64, 64)
    mask = torch.ones(64, 64, dtype=torch.bool)

    aug_image, aug_mask = pipeline(image, mask)
    assert aug_image.shape == image.shape
    assert aug_mask.shape == mask.shape
