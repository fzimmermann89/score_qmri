"""Utilities for working with ConcatDataset."""

from collections.abc import Sequence, Sized
from typing import cast

from torch.utils.data import Dataset


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
