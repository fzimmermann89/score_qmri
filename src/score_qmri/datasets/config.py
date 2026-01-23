"""Augmentation configuration dataclasses with preset instances."""

from dataclasses import dataclass


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline."""

    geometric_prob: float = 0.8
    contrast_prob: float = 0.3
    phase_prob: float = 0.3
    flip_prob: float = 0.5
    roll_prob: float = 1.0
    shift_range: tuple[int, int] = (-10, 10)
    scale_range: tuple[float, float] = (0.9, 1.1)
    rotation_range: tuple[float, float] = (-10.0, 10.0)
    shear_range: tuple[float, float] = (-0.1, 0.1)
    contrast_strength: float = 0.2
    poly_coeff_range: tuple[float, float] = (-0.3, 0.3)
    intensity_scale_prob: float = 0.0  # Disabled by default
    intensity_scale_range: tuple[float, float] = (0.25, 4.0)


MAGNITUDE_CONFIG = AugmentationConfig(
    geometric_prob=0.8,
    contrast_prob=0.3,
    phase_prob=0.0,
    shift_range=(-10, 10),
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.3,
    intensity_scale_prob=0.0,  # Disabled by default
)

MAPPING_CONFIG = AugmentationConfig(
    geometric_prob=0.8,
    contrast_prob=0.3,
    phase_prob=0.0,
    shift_range=(-10, 10),
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.1,
    intensity_scale_prob=0.0,  # Disabled by default
)

COMPLEX_CONFIG = AugmentationConfig(
    geometric_prob=0.8,
    contrast_prob=0.3,
    phase_prob=0.5,
    shift_range=(-10, 10),
    scale_range=(0.9, 1.2),
    rotation_range=(-10.0, 10.0),
    shear_range=(-0.1, 0.1),
    contrast_strength=0.3,
    intensity_scale_prob=0.0,  # Disabled by default
)
