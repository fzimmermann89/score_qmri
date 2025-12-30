"""Utilities for working with ConcatDataset."""

from collections.abc import Callable, Sequence, Sized
from typing import cast

from torch.utils.data import ConcatDataset, Dataset


def create_concat_dataset(datasets: Sequence[Dataset]) -> ConcatDataset:
    """Create a ConcatDataset from a list of datasets."""
    return ConcatDataset(datasets)


def get_size_labels(datasets: Sequence[Dataset]) -> list[tuple[int, int]]:
    """Get size labels (height, width) for each dataset."""
    size_labels = []
    for dataset in datasets:
        if len(cast(Sized, dataset)) == 0:
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


def _make_constant_schedule(weight: float) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return weight

    return schedule


def _make_fade_out_schedule(start: int, duration: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return max(0.0, 1.0 - (step - start) / duration)

    return schedule


def _make_step_schedule(threshold: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return 0.0 if step > threshold else 1.0

    return schedule


def create_schedule_functions(
    schedules: list[dict | Callable[[int], float]],
) -> list[Callable[[int], float]]:
    """Create schedule functions from configuration.

    Supported schedule types:
    - 'constant': constant weight (default 1.0)
    - 'fade_out': fade from 1.0 to 0.0 over fade_steps after start_step
    - 'step': drop to 0 after threshold
    """
    schedule_funcs: list[Callable[[int], float]] = []

    for schedule in schedules:
        if callable(schedule) and not isinstance(schedule, dict):
            schedule_funcs.append(schedule)
            continue

        if not isinstance(schedule, dict):
            raise ValueError(f"Schedule must be callable or dict, got {type(schedule)}")

        schedule_type: str = schedule.get("type", "constant")

        match schedule_type:
            case "constant":
                weight_value = schedule.get("weight", 1.0)
                schedule_funcs.append(_make_constant_schedule(float(weight_value)))
            case "fade_out":
                start_step_value = schedule.get("start_step", 0)
                fade_steps_value = schedule.get("fade_steps", 1000)
                schedule_funcs.append(
                    _make_fade_out_schedule(
                        int(start_step_value),
                        int(fade_steps_value),
                    )
                )
            case "step":
                threshold_value = schedule.get("threshold", 0)
                schedule_funcs.append(_make_step_schedule(int(threshold_value)))
            case _:
                raise ValueError(f"Unknown schedule type: {schedule_type}")

    return schedule_funcs
