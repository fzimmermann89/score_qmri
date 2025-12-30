"""Custom BatchSampler with step-based probability scheduling."""

from collections import defaultdict
from collections.abc import Callable, Sized
from typing import cast

import torch
from torch.utils.data import BatchSampler, ConcatDataset


class ScheduledBatchSampler(BatchSampler):
    """Batch sampler with step-based probability scheduling for multiple datasets.

    Groups datasets by size, samples buckets proportionally, then samples
    datasets within buckets. Each batch comes from a single dataset to ensure
    size homogeneity.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        size_labels: list[tuple[int, int]],
        schedule_functions: list[Callable[[int], float]],
        batch_size: int,
        base_seed: int = 42,
        rank: int = 0,
    ):
        if not isinstance(dataset, ConcatDataset):
            raise ValueError("dataset must be a ConcatDataset")
        if len(size_labels) != len(dataset.datasets):
            raise ValueError(f"size_labels length ({len(size_labels)}) != datasets ({len(dataset.datasets)})")
        if len(schedule_functions) != len(dataset.datasets):
            raise ValueError(f"schedule_functions length ({len(schedule_functions)}) != datasets ({len(dataset.datasets)})")

        self.dataset = dataset
        self.size_labels = size_labels
        self.schedule_functions = schedule_functions
        self.batch_size = batch_size
        self.rank = rank
        self.step = 0

        self.rng = torch.Generator()
        self.rng.manual_seed(base_seed + rank)

        # Precompute dataset offsets and lengths
        lengths = [len(cast(Sized, ds)) for ds in dataset.datasets]
        self.dataset_lengths = torch.tensor(lengths)
        self.dataset_offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.dataset_lengths, dim=0)[:-1]])

        # Group datasets by size
        self.size_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, size in enumerate(size_labels):
            self.size_buckets[size].append(i)

    def __iter__(self) -> "ScheduledBatchSampler":
        return self

    def __next__(self) -> list[int]:
        """Get next batch of indices in ConcatDataset index space."""
        weights = torch.tensor([f(self.step) for f in self.schedule_functions])

        # Compute bucket weights
        bucket_data = []
        for size, indices in self.size_buckets.items():
            bucket_weight = weights[indices].sum().item()
            if bucket_weight > 0:
                bucket_data.append((size, indices, bucket_weight))

        if not bucket_data:
            raise StopIteration("All dataset weights are zero")

        bucket_weights = torch.tensor([b[2] for b in bucket_data])
        bucket_idx = int(torch.multinomial(bucket_weights, 1, generator=self.rng).item())
        _, dataset_indices, _ = bucket_data[bucket_idx]

        # Sample dataset within bucket
        ds_weights = weights[dataset_indices]
        ds_idx_local = int(torch.multinomial(ds_weights, 1, generator=self.rng).item())
        ds_idx = dataset_indices[ds_idx_local]

        # Sample batch indices (with replacement)
        ds_len = int(self.dataset_lengths[ds_idx].item())
        if ds_len == 0:
            raise ValueError(f"Dataset {ds_idx} is empty")

        local_indices = torch.randint(0, ds_len, (self.batch_size,), generator=self.rng)
        offset = int(self.dataset_offsets[ds_idx].item())
        batch: list[int] = (local_indices + offset).tolist()

        self.step += 1
        return batch

    def __len__(self) -> int:
        """Estimate number of batches."""
        return max(1, int(self.dataset_lengths.sum().item()) // self.batch_size)

    def reset_step(self, step: int = 0) -> None:
        """Reset step counter."""
        self.step = step
