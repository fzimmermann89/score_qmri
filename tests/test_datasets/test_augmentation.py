"""Tests for AugmentationPipeline."""

import torch

from datasets.augmentation import AugmentationPipeline
from datasets.config import AugmentationConfig


def _disabled_config() -> AugmentationConfig:
    """Config with all augmentations disabled."""
    return AugmentationConfig(
        roll_prob=0.0,
        flip_prob=0.0,
        geometric_prob=0.0,
        contrast_prob=0.0,
        phase_prob=0.0,
        shift_range=(-10, 10),
    )


def _enabled_config() -> AugmentationConfig:
    """Config with all augmentations enabled."""
    return AugmentationConfig(
        roll_prob=1.0,
        flip_prob=1.0,
        geometric_prob=1.0,
        contrast_prob=1.0,
        phase_prob=1.0,
        shift_range=(-10, 10),
    )


class TestNoOp:
    """Tests for disabled augmentation behavior."""

    def test_real_unchanged(self):
        """Disabled augmentation returns identical real input."""
        pipeline = AugmentationPipeline(_disabled_config())
        image = torch.arange(64).float().reshape(8, 8)
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[:4, :] = True

        aug_image, aug_mask = pipeline(image, mask)

        torch.testing.assert_close(aug_image, image)
        torch.testing.assert_close(aug_mask, mask)

    def test_complex_unchanged(self):
        """Disabled augmentation returns identical complex input."""
        pipeline = AugmentationPipeline(_disabled_config())
        image = torch.randn(32, 32, dtype=torch.complex64)
        mask = torch.ones(32, 32, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)

        torch.testing.assert_close(aug_image, image)
        torch.testing.assert_close(aug_mask, mask)


class TestRealAugmentation:
    """Tests for real-valued image augmentation."""

    def test_preserves_shape(self):
        """All augmentations preserve input shape."""
        pipeline = AugmentationPipeline(_enabled_config())
        image = torch.randn(64, 64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)

        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape

    def test_roll_modifies_image(self):
        """Roll transform shifts image content."""
        config = AugmentationConfig(roll_prob=1.0, flip_prob=0.0, geometric_prob=0.0, contrast_prob=0.0)
        pipeline = AugmentationPipeline(config)
        image = torch.arange(64).float().reshape(8, 8)
        mask = torch.ones(8, 8, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)

        assert not torch.equal(aug_image, image)
        assert torch.equal(aug_image.sum(), image.sum())

    def test_flip_modifies_image(self):
        """Horizontal flip mirrors image."""
        config = AugmentationConfig(roll_prob=0.0, flip_prob=1.0, geometric_prob=0.0, contrast_prob=0.0)
        pipeline = AugmentationPipeline(config)
        image = torch.arange(64).float().reshape(8, 8)
        mask = torch.ones(8, 8, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)

        expected = image.flip(dims=(1,))
        torch.testing.assert_close(aug_image, expected)

    def test_geometric_modifies_both(self):
        """Affine transform modifies both image and asymmetric mask."""
        config = AugmentationConfig(roll_prob=0.0, flip_prob=0.0, geometric_prob=1.0, contrast_prob=0.0)
        pipeline = AugmentationPipeline(config)
        image = torch.arange(64 * 64).float().reshape(64, 64)
        mask = torch.zeros(64, 64, dtype=torch.bool)
        mask[:32, :] = True

        aug_image, aug_mask = pipeline(image, mask)

        assert not torch.equal(aug_image, image)
        assert not torch.equal(aug_mask, mask)

    def test_gamma_preserves_range(self):
        """Gamma correction keeps [0,1] input in [0,1] range."""
        config = AugmentationConfig(roll_prob=0.0, flip_prob=0.0, geometric_prob=0.0, contrast_prob=1.0)
        pipeline = AugmentationPipeline(config)
        image = torch.rand(64, 64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)

        assert aug_image.min() >= -1e-6
        assert aug_image.max() <= 1.0 + 1e-6


class TestComplexAugmentation:
    """Tests for complex-valued image augmentation."""

    def test_preserves_dtype(self):
        """Augmentations preserve complex dtype."""
        pipeline = AugmentationPipeline(_enabled_config())
        image = torch.randn(64, 64, dtype=torch.complex64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, aug_mask = pipeline(image, mask)

        assert aug_image.is_complex()
        assert aug_image.shape == image.shape
        assert aug_mask.dtype == torch.bool

    def test_phase_preserves_magnitude(self):
        """Phase augmentation changes phase but preserves magnitude."""
        config = AugmentationConfig(roll_prob=0.0, flip_prob=0.0, geometric_prob=0.0, phase_prob=1.0)
        pipeline = AugmentationPipeline(config)
        image = torch.randn(64, 64, dtype=torch.complex64)
        mask = torch.ones(64, 64, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)

        torch.testing.assert_close(aug_image.abs(), image.abs(), atol=1e-5, rtol=1e-5)
        assert not torch.allclose(aug_image.angle(), image.angle(), atol=0.01)

    def test_spatial_on_complex(self):
        """Spatial transforms work on complex data."""
        config = AugmentationConfig(roll_prob=1.0, flip_prob=1.0, geometric_prob=1.0, phase_prob=0.0)
        pipeline = AugmentationPipeline(config)
        image = torch.arange(64, dtype=torch.float).reshape(8, 8)
        image = torch.complex(image, image)
        mask = torch.ones(8, 8, dtype=torch.bool)

        aug_image, _ = pipeline(image, mask)

        assert aug_image.is_complex()
        assert not torch.equal(aug_image, image)
