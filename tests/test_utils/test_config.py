"""Tests for configuration utilities."""

from score_qmri.utils.config import normalize_config


def test_normalize_config_stability():
    """Test that normalize_config produces stable output across different representations."""
    config_a = {
        "model": {"beta_max": 10, "lr": 1e-4},
        "trainer": {
            "max_steps": 100,
            "logger": "<wandb.logger object at 0x123>",
            "callbacks": ["<checkpoint at 0x456>"],
        },
        "data": [{"batch_size": 32}],
    }

    config_b = {
        "model": {"beta_max": 10.0, "lr": 1e-4},  # float instead of int
        "trainer": {
            "max_steps": 100,
            "logger": "<wandb.logger object at 0x999>",  # different address
            "callbacks": ["<checkpoint at 0x888>"],  # different address
        },
        "data": [{"batch_size": 32}],
    }

    norm_a = normalize_config(config_a)
    norm_b = normalize_config(config_b)

    assert norm_a == norm_b
    assert "logger" not in norm_a["trainer"]
    assert "callbacks" not in norm_a["trainer"]
    assert isinstance(norm_a["model"]["beta_max"], float)
    assert norm_a["model"]["beta_max"] == 10.0


def test_normalize_nested_structures():
    """Test normalization of deeply nested structures."""
    config = {
        "a": [1, 2, {"b": 3}],
        "c": {"d": [4, 5]},
    }
    expected = {
        "a": [1.0, 2.0, {"b": 3.0}],
        "c": {"d": [4.0, 5.0]},
    }
    assert normalize_config(config) == expected
