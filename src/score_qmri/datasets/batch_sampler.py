"""Custom BatchSampler with step-based probability scheduling."""

import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import BatchSampler, ConcatDataset

from score_qmri.datasets.kind import DatasetKind


class ScheduledBatchSampler(BatchSampler):
    """Batch sampler with per-element dataset sampling and kind normalization.

    Samples dataset index independently for each element in the batch,
    producing mixed-kind batches. Supports interpolation between size-proportional
    and kind-uniform sampling via `kind_normalization`.

    Probability formula for dataset j with kind k:
        p_size(j) = w_j / W_total           (size-proportional)
        p_kind(j) = (1/K) * (w_j / W_k)     (kind-uniform)
        p(j) = (1-α) * p_size(j) + α * p_kind(j)

    where:
        w_j = schedule_j(step) * len_j      (raw weight)
        W_k = sum of w_j for datasets with kind k
        W_total = sum of all w_j
        K = number of active kinds (kinds with W_k > 0)
        α = kind_normalization

    Note: The `sampler` parameter exists for Lightning DDP compatibility.
    Lightning's dataloader wrapper checks for this parameter; we handle
    distribution ourselves via rank-based seeding.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        dataset_lengths: list[int],
        dataset_kinds: Sequence[DatasetKind],
        schedules: Sequence[Callable[[int], float]],
        batch_size: int,
        kind_normalization: float = 0.5,
        base_seed: int = 42,
        rank: int = 0,
        sampler: "torch.utils.data.Sampler | None" = None,
        max_steps: int | None = None,
    ):
        """
        Parameters
        ----------
        dataset
            ConcatDataset containing all sub-datasets.
        dataset_lengths
            Length of each sub-dataset.
        dataset_kinds
            DatasetKind for each sub-dataset.
        schedules
            Schedule functions returning weight modifier for each dataset at given step.
        batch_size
            Number of samples per batch.
        kind_normalization
            Interpolation between size-proportional (0.0) and kind-uniform (1.0) sampling.
            Default 0.5 balances both.
        base_seed
            Base random seed.
        rank
            Distributed rank for seed offset.
        sampler
            Unused, exists for Lightning DDP compatibility.
        max_steps
            Total training steps. Sampler reports this as its length, enabling single-epoch
            training that avoids iterator recreation overhead. If None, falls back to
            total_samples // batch_size (legacy behavior with warning).
        """
        n_ds = len(dataset.datasets)
        if not (len(dataset_lengths) == len(dataset_kinds) == len(schedules) == n_ds):
            raise ValueError("Length mismatch between datasets and config lists")
        if not 0.0 <= kind_normalization <= 1.0:
            raise ValueError(f"kind_normalization must be in [0, 1], got {kind_normalization}")

        if max_steps is None:
            warnings.warn(
                "ScheduledBatchSampler: max_steps not set. Using total_samples // batch_size as epoch length. "
                "This causes iterator recreation at epoch boundaries. Set max_steps=trainer.max_steps for "
                "continuous training.",
                UserWarning,
                stacklevel=2,
            )

        self.dataset = dataset
        self.schedules = schedules
        self.batch_size = batch_size
        self.kind_normalization = kind_normalization
        self.step = 0
        self.sampler: Any = sampler
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.rank = rank

        self.rng = torch.Generator()
        self.rng.manual_seed(base_seed + rank)

        self.dataset_lengths = torch.tensor(dataset_lengths, dtype=torch.float32)
        self.dataset_kinds = torch.tensor([int(k) for k in dataset_kinds], dtype=torch.long)
        self.dataset_offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.dataset_lengths.long(), dim=0)[:-1]])

        self._unique_kinds = torch.unique(self.dataset_kinds)

    def _compute_probs(self) -> torch.Tensor:
        # Raw weights: schedule(step) * length
        modifiers = torch.tensor([s(self.step) for s in self.schedules], dtype=torch.float32)
        weights = modifiers * self.dataset_lengths  # (n_ds,)

        w_total = weights.sum()
        if w_total <= 0:
            raise RuntimeError("All dataset weights are zero")

        # Size-proportional probabilities
        p_size = weights / w_total

        if self.kind_normalization == 0.0:
            return p_size

        # Compute per-kind total weights
        kind_weights = torch.zeros(self._unique_kinds.max() + 1, dtype=torch.float32)
        kind_weights.scatter_add_(0, self.dataset_kinds, weights)

        # Count active kinds (non-zero weight)
        n_active_kinds = (kind_weights > 0).sum().item()

        if n_active_kinds == 0:
            raise RuntimeError("No active kinds")

        # Kind-uniform probabilities: (1/K) * (w_j / W_k)
        # For inactive kinds, W_k = 0, but w_j = 0 too, so p_kind = 0
        w_k_per_dataset = kind_weights[self.dataset_kinds]  # W_k for each dataset's kind
        # Avoid division by zero for inactive kinds
        safe_w_k = w_k_per_dataset.clamp(min=1e-10)
        p_kind = (1.0 / n_active_kinds) * (weights / safe_w_k)

        # Interpolate
        alpha = self.kind_normalization
        probs = (1 - alpha) * p_size + alpha * p_kind

        return probs

    def _next_batch(self) -> list[int]:
        probs = self._compute_probs()
        ds_indices = torch.multinomial(probs, self.batch_size, replacement=True, generator=self.rng)

        ds_lens = self.dataset_lengths[ds_indices]
        local_indices = (torch.rand(self.batch_size, generator=self.rng) * ds_lens).long()
        global_indices = local_indices + self.dataset_offsets[ds_indices].long()

        self.step += 1
        return global_indices.tolist()

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(len(self)):
            yield self._next_batch()

    def __len__(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        return max(1, int(self.dataset_lengths.sum().item()) // self.batch_size)

    def reset_step(self, step: int = 0) -> None:
        self.step = step
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Called by Lightning at epoch boundaries. Reseeds RNG but preserves step for schedule continuity."""
        self.rng.manual_seed(self.base_seed + self.rank + epoch * 10000 + self.step * 1000003)
