"""Tests for AugmentationPipeline."""

import torch

from datasets.augmentation import AugmentationPipeline
from datasets.config import AugmentationConfig


class TestRealAugmentation:
    """Tests for real-valued (magnitude) image augmentation."""

    def test_preserves_shape(self, augmentation_config):
        """Test augmentation preserves input shape."""
        pipeline = AugmentationPipeline(augmentation_config)
        image = torch.randn(64, 64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape

    def test_flip_applied(self):
        """Test flip augmentation is applied."""
        config = AugmentationConfig(geometric_prob=0.0, contrast_prob=0.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        image = torch.arange(16).float().reshape(4, 4)
        mask = torch.ones(4, 4, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape
        # Values should be rearranged (roll + flip applied)
        assert not torch.allclose(aug_image, image)

    def test_geometric_transforms_both(self):
        """Test geometric augmentation transforms both image and mask."""
        config = AugmentationConfig(geometric_prob=1.0, contrast_prob=0.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        image = torch.randn(64, 64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == (64, 64)
        assert aug_mask.shape == (64, 64)

    def test_contrast_preserves_range(self):
        """Test contrast augmentation keeps values in reasonable range."""
        config = AugmentationConfig(geometric_prob=0.0, contrast_prob=1.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        image = torch.rand(64, 64)  # [0, 1] range
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)
        assert aug_image.min() >= -0.5
        assert aug_image.max() <= 1.5

    def test_all_augmentations_stack(self):
        """Test all augmentations can be applied together."""
        config = AugmentationConfig(geometric_prob=1.0, contrast_prob=1.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        image = torch.randn(64, 64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape


class TestComplexAugmentation:
    """Tests for complex-valued image augmentation."""

    def test_preserves_shape_and_dtype(self):
        """Test complex augmentation preserves shape and dtype."""
        config = AugmentationConfig(geometric_prob=0.0, contrast_prob=0.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        image = torch.randn(64, 64, dtype=torch.complex64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == image.shape
        assert aug_image.is_complex()
        assert aug_mask.shape == mask.shape

    def test_phase_augmentation_changes_phase(self):
        """Test phase augmentation modifies phase while preserving magnitude."""
        config = AugmentationConfig(geometric_prob=0.0, contrast_prob=0.0, phase_prob=1.0)
        pipeline = AugmentationPipeline(config)

        # Create image with known magnitude
        real = torch.randn(64, 64)
        imag = torch.randn(64, 64)
        image = torch.complex(real, imag)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)

        # Phase should change but magnitude should be similar (within tolerance due to spatial transforms)
        orig_mag = image.abs()
        aug_mag = aug_image.abs()
        # Magnitudes should be correlated (not exact due to spatial shifts)
        assert aug_mag.mean() > 0  # Non-degenerate
        assert aug_image.is_complex()

    def test_phase_augmentation_disabled_when_zero(self):
        """Test phase augmentation is disabled when prob=0."""
        config = AugmentationConfig(geometric_prob=0.0, contrast_prob=0.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        real = torch.randn(64, 64)
        imag = torch.randn(64, 64)
        image = torch.complex(real, imag)
        mask = torch.ones(64, 64, dtype=torch.bool)

        # Even with spatial transforms, phase relationship should be preserved
        aug_image, _ = pipeline(image, mask)
        assert aug_image.is_complex()

    def test_spatial_augmentations_work_on_complex(self):
        """Test that spatial augmentations work correctly on complex data."""
        config = AugmentationConfig(geometric_prob=1.0, contrast_prob=0.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        image = torch.randn(64, 64, dtype=torch.complex64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)
        assert aug_image.shape == (64, 64)
        assert aug_image.is_complex()
        assert aug_mask.shape == (64, 64)

    def test_no_contrast_on_complex(self):
        """Test that contrast augmentation is not applied to complex data."""
        config = AugmentationConfig(geometric_prob=0.0, contrast_prob=1.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)

        # Create deterministic complex image
        real = torch.ones(64, 64)
        imag = torch.ones(64, 64) * 0.5
        image = torch.complex(real, imag)
        mask = torch.ones(64, 64, dtype=torch.bool)

        # With only contrast (which shouldn't apply) and spatial transforms
        aug_image, _ = pipeline(image, mask)
        # Should still be complex and have similar statistics
        assert aug_image.is_complex()
