"""Utilities for working with ConcatDataset."""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset


def create_concat_dataset(datasets: list[Dataset]) -> ConcatDataset:
    """Create a ConcatDataset from a list of datasets.

    Args:
        datasets: List of dataset instances

    Returns:
        ConcatDataset wrapping all datasets
    """
    return ConcatDataset(datasets)


def get_size_labels(datasets: list[Dataset]) -> list[tuple[int, int]]:
    """Get size labels (height, width) for each dataset.

    Each dataset must have a fixed size. This function extracts the size
    from the first sample of each dataset.

    Args:
        datasets: List of dataset instances

    Returns:
        List of (height, width) tuples, one per dataset
    """
    size_labels = []
    for dataset in datasets:
        if len(dataset) == 0:
            raise ValueError("Cannot get size from empty dataset")
        # Get first sample to determine size
        sample = dataset[0]
        if isinstance(sample, dict):
            image = sample["image"]
        else:
            image = sample[0] if isinstance(sample, tuple) else sample

        if image.ndim == 2:
            H, W = image.shape
        elif image.ndim == 3:
            _, H, W = image.shape
        else:
            raise ValueError(
                f"Unexpected image shape: {image.shape}. Expected 2D or 3D."
            )

        size_labels.append((H, W))

    return size_labels


def create_schedule_functions(
    schedules: list[dict[str, Any]] | list[Callable[[int], float]],
) -> list[Callable[[int], float]]:
    """Create schedule functions from configuration.

    Args:
        schedules: List of schedule configurations or callable functions.
            If dict, must have 'type' key with values:
            - 'constant': constant weight
            - 'fade_out': fade out from start_step over fade_steps
            - 'step': step function that drops to 0 after threshold

    Returns:
        List of callable functions p_i(step) -> float
    """
    schedule_funcs = []

    for schedule in schedules:
        if callable(schedule):
            schedule_funcs.append(schedule)
        elif isinstance(schedule, dict):
            schedule_type = schedule.get("type", "constant")

            if schedule_type == "constant":
                weight = schedule.get("weight", 1.0)
                schedule_funcs.append(lambda step, w=weight: w)

            elif schedule_type == "fade_out":
                start_step = schedule.get("start_step", 0)
                fade_steps = schedule.get("fade_steps", 1000)
                schedule_funcs.append(
                    lambda step, s=start_step, f=fade_steps: max(
                        0.0, 1.0 - (step - s) / f
                    )
                )

            elif schedule_type == "step":
                threshold = schedule.get("threshold", 0)
                schedule_funcs.append(
                    lambda step, t=threshold: 0.0 if step > t else 1.0
                )

            else:
                raise ValueError(f"Unknown schedule type: {schedule_type}")
        else:
            raise ValueError(
                f"Schedule must be callable or dict, got {type(schedule)}"
            )

    return schedule_funcs

