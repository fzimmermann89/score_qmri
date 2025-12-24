"""Custom BatchSampler with step-based probability scheduling."""

import random
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.utils.data import BatchSampler, ConcatDataset, Dataset


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
        """Initialize scheduled batch sampler.

        Args:
            dataset: ConcatDataset containing all datasets
            size_labels: List of (height, width) tuples, one per dataset
            schedule_functions: List of functions p_i(step) -> float for each dataset
            batch_size: Batch size
            base_seed: Base seed for RNG
            rank: Rank for DDP (used for seeding)
        """
        if not isinstance(dataset, ConcatDataset):
            raise ValueError("dataset must be a ConcatDataset")

        if len(size_labels) != len(dataset.datasets):
            raise ValueError(
                f"size_labels length ({len(size_labels)}) must match "
                f"number of datasets ({len(dataset.datasets)})"
            )

        if len(schedule_functions) != len(dataset.datasets):
            raise ValueError(
                f"schedule_functions length ({len(schedule_functions)}) must match "
                f"number of datasets ({len(dataset.datasets)})"
            )

        self.dataset = dataset
        self.size_labels = size_labels
        self.schedule_functions = schedule_functions
        self.batch_size = batch_size
        self.rank = rank

        # Initialize RNG with rank-aware seed
        self.rng = random.Random(base_seed + rank)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(base_seed + rank)

        # Step counter (per-rank)
        self.step = 0

        # Compute dataset offsets in ConcatDataset index space
        self.dataset_offsets = [0]
        for ds in dataset.datasets:
            self.dataset_offsets.append(
                self.dataset_offsets[-1] + len(ds)
            )

        # Group datasets by size
        self.size_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, size in enumerate(size_labels):
            self.size_buckets[size].append(i)

    def __iter__(self):
        """Create iterator over batches."""
        return self

    def __next__(self) -> list[int]:
        """Get next batch of indices.

        Returns:
            List of indices in ConcatDataset index space
        """
        # Compute weights for all datasets
        weights = [
            func(self.step) for func in self.schedule_functions
        ]

        # Group weights by size bucket
        bucket_weights: dict[tuple[int, int], float] = {}
        for size, dataset_indices in self.size_buckets.items():
            bucket_weight = sum(weights[i] for i in dataset_indices)
            if bucket_weight > 0:
                bucket_weights[size] = bucket_weight

        if not bucket_weights:
            raise StopIteration("All dataset weights are zero")

        # Sample bucket proportionally
        sizes = list(bucket_weights.keys())
        bucket_probs = [
            bucket_weights[size] for size in sizes
        ]
        total_weight = sum(bucket_probs)
        bucket_probs = [w / total_weight for w in bucket_probs]

        # Use torch for sampling to ensure reproducibility
        bucket_idx = torch.multinomial(
            torch.tensor(bucket_probs), 1, generator=self.torch_rng
        ).item()
        selected_size = sizes[bucket_idx]
        selected_bucket = self.size_buckets[selected_size]

        # Sample dataset within bucket proportionally
        dataset_weights = [
            weights[i] for i in selected_bucket
        ]
        total_dataset_weight = sum(dataset_weights)
        dataset_probs = [w / total_dataset_weight for w in dataset_weights]

        dataset_idx_in_bucket = torch.multinomial(
            torch.tensor(dataset_probs), 1, generator=self.torch_rng
        ).item()
        selected_dataset_idx = selected_bucket[dataset_idx_in_bucket]

        # Get the actual dataset
        selected_dataset = self.dataset.datasets[selected_dataset_idx]
        dataset_len = len(selected_dataset)

        if dataset_len == 0:
            raise ValueError(
                f"Selected dataset {selected_dataset_idx} is empty"
            )

        # Draw batch_size indices uniformly from dataset (with replacement)
        # Convert to ConcatDataset index space
        dataset_offset = self.dataset_offsets[selected_dataset_idx]
        batch_indices = [
            dataset_offset + self.rng.randint(0, dataset_len - 1)
            for _ in range(self.batch_size)
        ]

        # Increment step
        self.step += 1

        return batch_indices

    def __len__(self) -> int:
        """Estimate number of batches.

        Returns:
            Estimated number of batches (approximate)
        """
        # Rough estimate based on total samples
        total_samples = sum(len(ds) for ds in self.dataset.datasets)
        return max(1, total_samples // self.batch_size)

    def reset_step(self, step: int = 0) -> None:
        """Reset step counter (useful for testing or restarting).

        Args:
            step: New step value (default: 0)
        """
        self.step = step

