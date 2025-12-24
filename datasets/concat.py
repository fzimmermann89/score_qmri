"""Utilities for working with ConcatDataset."""

from collections.abc import Callable
from typing import Any

from torch.utils.data import ConcatDataset, Dataset


def create_concat_dataset(datasets: list[Dataset]) -> ConcatDataset:
    """Create a ConcatDataset from a list of datasets."""
    return ConcatDataset(datasets)


def get_size_labels(datasets: list[Dataset]) -> list[tuple[int, int]]:
    """Get size labels (height, width) for each dataset.

    Parameters
    ----------
    datasets : list[Dataset]
        List of dataset instances. Each must have fixed size.

    Returns
    -------
    list[tuple[int, int]]
        List of (height, width) tuples, one per dataset.

    Raises
    ------
    ValueError
        If dataset is empty or has unexpected image shape.
    """
    size_labels = []
    for dataset in datasets:
        if len(dataset) == 0:
            raise ValueError("Cannot get size from empty dataset")

        sample = dataset[0]
        image = sample["image"] if isinstance(sample, dict) else sample[0]

        match image.ndim:
            case 2:
                H, W = image.shape
            case 3:
                _, H, W = image.shape
            case _:
                raise ValueError(f"Unexpected image shape: {image.shape}")

        size_labels.append((H, W))

    return size_labels


def create_schedule_functions(
    schedules: list[dict[str, Any] | Callable[[int], float]],
) -> list[Callable[[int], float]]:
    """Create schedule functions from configuration.

    Parameters
    ----------
    schedules : list[dict | Callable]
        List of schedule configurations or callable functions.
        If dict, must have 'type' key:
        - 'constant': constant weight (default 1.0)
        - 'fade_out': fade from 1.0 to 0.0 over fade_steps after start_step
        - 'step': drop to 0 after threshold

    Returns
    -------
    list[Callable[[int], float]]
        List of callable functions p_i(step) -> float.
    """
    schedule_funcs = []

    for schedule in schedules:
        if callable(schedule):
            schedule_funcs.append(schedule)
            continue

        if not isinstance(schedule, dict):
            raise ValueError(f"Schedule must be callable or dict, got {type(schedule)}")

        schedule_type = schedule.get("type", "constant")

        if schedule_type == "constant":
            w = schedule.get("weight", 1.0)
            schedule_funcs.append(lambda step, w=w: w)

        elif schedule_type == "fade_out":
            start = schedule.get("start_step", 0)
            duration = schedule.get("fade_steps", 1000)
            schedule_funcs.append(
                lambda step, s=start, d=duration: max(0.0, 1.0 - (step - s) / d)
            )

        elif schedule_type == "step":
            threshold = schedule.get("threshold", 0)
            schedule_funcs.append(lambda step, t=threshold: 0.0 if step > t else 1.0)

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    return schedule_funcs
