import jsonargparse

from score_qmri.datasets.config import COMPLEX_CONFIG, MAGNITUDE_CONFIG
from score_qmri.datasets.lightning import ComplexDatasetConfig, MagnitudeDatasetConfig


def _parse_config(klass: type, data: dict[str, object]):
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(klass, "dataset")
    cfg = parser.parse_object({"dataset": data})
    return parser.instantiate_classes(cfg).dataset


def test_complex_partial_augmentation_inherits_preset_defaults() -> None:
    cfg = _parse_config(
        ComplexDatasetConfig,
        {
            "h5_path": "dummy.h5",
            "augmentation_config": {"scale_range": [0.7, 0.8]},
        },
    )

    assert cfg.augmentation_config is not None
    assert cfg.augmentation_config.scale_range == (0.7, 0.8)
    assert cfg.augmentation_config.phase_prob == COMPLEX_CONFIG.phase_prob
    assert cfg.augmentation_config.contrast_strength == COMPLEX_CONFIG.contrast_strength


def test_magnitude_partial_augmentation_inherits_preset_defaults() -> None:
    cfg = _parse_config(
        MagnitudeDatasetConfig,
        {
            "h5_path": "dummy.h5",
            "augmentation_config": {"scale_range": [0.7, 0.8]},
        },
    )

    assert cfg.augmentation_config is not None
    assert cfg.augmentation_config.scale_range == (0.7, 0.8)
    assert cfg.augmentation_config.phase_prob == MAGNITUDE_CONFIG.phase_prob
    assert cfg.augmentation_config.contrast_strength == MAGNITUDE_CONFIG.contrast_strength
