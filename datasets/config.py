"""Augmentation configuration dataclasses with preset instances."""

from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline."""

    geometric_prob: float = 0.3
    contrast_prob: float = 0.3
    phase_prob: float = 0.3
    scale_range: tuple[float, float] = (0.9, 1.2)
    rotation_range: tuple[float, float] = (-10.0, 10.0)
    shear_range: tuple[float, float] = (-0.1, 0.1)
    contrast_strength: float = 0.2


# Preset configurations for different dataset types
MAGNITUDE_CONFIG = AugmentationConfig(
    geometric_prob=0.3,
    contrast_prob=0.3,
    phase_prob=0.0,  # No phase augmentation for magnitude
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.3,
)

MAPPING_CONFIG = AugmentationConfig(
    geometric_prob=0.3,
    contrast_prob=0.3,
    phase_prob=0.0,  # No phase augmentation for mapping
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.1,
)

COMPLEX_CONFIG = AugmentationConfig(
    geometric_prob=0.3,
    contrast_prob=0.3,
    phase_prob=0.5,  # Phase augmentation for complex data
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.3,
)
