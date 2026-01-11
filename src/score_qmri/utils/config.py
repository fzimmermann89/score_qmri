"""Configuration utilities."""

from typing import Any


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a configuration dictionary to ensure stability across runs.

    Removes unstable keys (containing object addresses) and normalizes types.

    Parameters
    ----------
    config
        Input configuration dictionary.

    Returns
    -------
    dict
        Normalized configuration dictionary.
    """
    normalized: dict[str, Any] = {}

    for key, value in config.items():
        # Drop unstable trainer keys containing object representations
        if key == "trainer":
            if isinstance(value, dict):
                normalized[key] = _normalize_trainer_config(value)
            else:
                # Should be a dict, but keep as-is if not (e.g. None)
                normalized[key] = value
        elif isinstance(value, dict):
            normalized[key] = normalize_config(value)
        elif isinstance(value, list):
            normalized[key] = [_normalize_value(v) for v in value]
        else:
            normalized[key] = _normalize_value(value)

    return normalized


def _normalize_trainer_config(trainer_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize trainer config by removing unstable keys."""
    normalized: dict[str, Any] = {}
    unstable_keys = {"logger", "callbacks"}

    for key, value in trainer_config.items():
        if key in unstable_keys:
            continue
        if isinstance(value, dict):
            normalized[key] = normalize_config(value)
        elif isinstance(value, list):
            normalized[key] = [_normalize_value(v) for v in value]
        else:
            normalized[key] = _normalize_value(value)

    return normalized


def _normalize_value(value: Any) -> Any:
    """Normalize a single value."""
    if isinstance(value, dict):
        return normalize_config(value)
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]

    # Normalize numbers: treat 1 and 1.0 as 1.0
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    return value
