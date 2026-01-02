"""Custom BatchSampler with step-based probability scheduling."""

from collections import defaultdict
from collections.abc import Callable, Iterator

import torch
from torch.utils.data import BatchSampler, ConcatDataset


class ScheduledBatchSampler(BatchSampler):
    """Batch sampler with size-proportional, schedule-modulated sampling.

    Final weight = schedule_fn(step) * dataset_length
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        size_labels: list[tuple[int, int]],
        dataset_lengths: list[int],
        schedules: list[Callable[[int], float]],
        batch_size: int,
        base_seed: int = 42,
        rank: int = 0,
    ):
        n_ds = len(dataset.datasets)
        if not (len(size_labels) == len(dataset_lengths) == len(schedules) == n_ds):
            raise ValueError("Length mismatch between datasets and config lists")

        self.dataset = dataset
        self.size_labels = size_labels
        self.schedules = schedules
        self.batch_size = batch_size
        self.step = 0

        self.rng = torch.Generator()
        self.rng.manual_seed(base_seed + rank)

        self.dataset_lengths = torch.tensor(dataset_lengths, dtype=torch.float32)
        self.dataset_offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.dataset_lengths.long(), dim=0)[:-1]])

        self.size_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, size in enumerate(size_labels):
            self.size_buckets[size].append(i)

    def _compute_weights(self) -> torch.Tensor:
        modifiers = torch.tensor([s(self.step) for s in self.schedules])
        return modifiers * self.dataset_lengths

    def _next_batch(self) -> list[int]:
        """Generate a single batch of indices."""
        weights = self._compute_weights()

        bucket_data = []
        for indices in self.size_buckets.values():
            bucket_weight = weights[indices].sum().item()
            if bucket_weight > 0:
                bucket_data.append((indices, bucket_weight))

        if not bucket_data:
            raise RuntimeError("All dataset weights are zero")

        bucket_weights = torch.tensor([b[1] for b in bucket_data])
        bucket_idx = int(torch.multinomial(bucket_weights, 1, generator=self.rng).item())
        dataset_indices = bucket_data[bucket_idx][0]

        ds_weights = weights[dataset_indices]
        ds_idx = dataset_indices[int(torch.multinomial(ds_weights, 1, generator=self.rng).item())]

        ds_len = int(self.dataset_lengths[ds_idx].item())
        local_indices = torch.randint(0, ds_len, (self.batch_size,), generator=self.rng)
        offset = int(self.dataset_offsets[ds_idx].item())

        self.step += 1
        return (local_indices + offset).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(len(self)):
            yield self._next_batch()

    def __len__(self) -> int:
        return max(1, int(self.dataset_lengths.sum().item()) // self.batch_size)

    def reset_step(self, step: int = 0) -> None:
        self.step = step

    def set_epoch(self, epoch: int) -> None:
        """Reset step counter for new epoch (for distributed training compatibility)."""
        self.step = 0
