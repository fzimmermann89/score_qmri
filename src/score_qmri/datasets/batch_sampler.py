"""Custom BatchSampler with step-based probability scheduling."""

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.data import BatchSampler, ConcatDataset


class ScheduledBatchSampler(BatchSampler):
    """Batch sampler with size-proportional, schedule-modulated sampling.

    Final weight = schedule_fn(step) * dataset_length

    Note: The `sampler` parameter exists for Lightning DDP compatibility.
    Lightning's dataloader wrapper checks for this parameter; we handle
    distribution ourselves via rank-based seeding.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        dataset_lengths: list[int],
        schedules: list[Callable[[int], float]],
        batch_size: int,
        base_seed: int = 42,
        rank: int = 0,
        sampler: "torch.utils.data.Sampler | None" = None,
    ):
        n_ds = len(dataset.datasets)
        if not (len(dataset_lengths) == len(schedules) == n_ds):
            raise ValueError("Length mismatch between datasets and config lists")

        self.dataset = dataset
        self.schedules = schedules
        self.batch_size = batch_size
        self.step = 0
        self.sampler: Any = sampler

        self.rng = torch.Generator()
        self.rng.manual_seed(base_seed + rank)

        self.dataset_lengths = torch.tensor(dataset_lengths, dtype=torch.float32)
        self.dataset_offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.dataset_lengths.long(), dim=0)[:-1]])

    def _compute_weights(self) -> torch.Tensor:
        modifiers = torch.tensor([s(self.step) for s in self.schedules])
        return modifiers * self.dataset_lengths

    def _next_batch(self) -> list[int]:
        weights = self._compute_weights()

        if weights.sum() <= 0:
            raise RuntimeError("All dataset weights are zero")

        ds_idx = int(torch.multinomial(weights, 1, generator=self.rng).item())

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
        self.step = 0
