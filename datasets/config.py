"""Augmentation configuration dataclasses with preset instances."""

from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline."""

    flip_prob: float = 0.5
    geometric_prob: float = 0.3
    contrast_prob: float = 0.3
    scale_range: tuple[float, float] = (0.9, 1.2)
    rotation_range: tuple[float, float] = (-10.0, 10.0)
    shear_range: tuple[float, float] = (-0.1, 0.1)
    contrast_strength: float = 0.2


# Preset configurations for different dataset types
MAGNITUDE_CONFIG = AugmentationConfig(
    flip_prob=0.5,
    geometric_prob=0.3,
    contrast_prob=0.3,
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.3,  # Higher strength for magnitude images
)

MAPPING_CONFIG = AugmentationConfig(
    flip_prob=0.5,
    geometric_prob=0.3,
    contrast_prob=0.3,
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.1,  # Lower strength for mapping datasets
)

COMPLEX_CONFIG = AugmentationConfig(
    flip_prob=0.5,
    geometric_prob=0.3,
    contrast_prob=0.3,
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.3,  # Higher strength for complex images
)

