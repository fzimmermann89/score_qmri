"""Augmentation configuration dataclasses with preset instances."""

from dataclasses import dataclass


@dataclass(frozen=True)
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
    poly_coeff_range: tuple[float, float] = (-0.2, 0.2)
    intensity_scale_prob: float = 0.0  # Disabled by default
    intensity_scale_range: tuple[float, float] = (0.9, 1.05)


@dataclass(frozen=True)
class ComplexAugmentationConfig(AugmentationConfig):
    """Preset defaults for complex-valued datasets."""

    phase_prob: float = 0.5
    scale_range: tuple[float, float] = (0.9, 1.2)
    contrast_strength: float = 0.3
    intensity_scale_prob: float = 0.1
    intensity_scale_range: tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class MagnitudeAugmentationConfig(AugmentationConfig):
    """Preset defaults for magnitude datasets."""

    phase_prob: float = 0.0
    scale_range: tuple[float, float] = (0.9, 1.2)
    contrast_strength: float = 0.3
    intensity_scale_prob: float = 0.1
    intensity_scale_range: tuple[float, float] = (0.8, 1.2)


@dataclass(frozen=True)
class MappingAugmentationConfig(AugmentationConfig):
    """Preset defaults for quantitative mapping datasets."""

    phase_prob: float = 0.0
    scale_range: tuple[float, float] = (0.9, 1.2)
    contrast_strength: float = 0.2
    intensity_scale_prob: float = 0.3
    intensity_scale_range: tuple[float, float] = (0.9, 1.1)


COMPLEX_CONFIG = ComplexAugmentationConfig()
MAGNITUDE_CONFIG = MagnitudeAugmentationConfig()
MAPPING_CONFIG = MappingAugmentationConfig()
