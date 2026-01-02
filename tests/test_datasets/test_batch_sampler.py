"""Tests for ScheduledBatchSampler."""

import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from score_qmri.datasets.batch_sampler import ScheduledBatchSampler


def test_batch_sampler_produces_correct_batch_size():
    """Test sampler produces batches of correct size."""
    ds = TensorDataset(torch.randn(20, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(concat, [(32, 32)], [20], [lambda s: 1.0], batch_size=4, base_seed=42, rank=0)

    batch = next(iter(sampler))
    assert len(batch) == 4


def test_batch_sampler_indices_in_valid_range():
    """Test sampler produces valid indices."""
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(concat, [(32, 32), (32, 32)], [10, 15], [lambda s: 1.0] * 2, batch_size=4, base_seed=42, rank=0)

    batch = next(iter(sampler))
    assert all(0 <= idx < len(concat) for idx in batch)


def test_batch_sampler_respects_zero_weight():
    """Test sampler skips datasets with zero weight."""
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    # First dataset has weight 0
    sampler = ScheduledBatchSampler(
        concat, [(32, 32), (32, 32)], [10, 15], [lambda s: 0.0, lambda s: 1.0], batch_size=4, base_seed=42, rank=0
    )

    batch = next(iter(sampler))
    # All indices should be from second dataset (offset 10)
    assert all(idx >= 10 for idx in batch)


def test_batch_sampler_different_ranks_differ():
    """Test different ranks produce different batches."""
    ds = TensorDataset(torch.randn(100, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler0 = ScheduledBatchSampler(concat, [(32, 32)], [100], [lambda s: 1.0], batch_size=8, base_seed=42, rank=0)
    sampler1 = ScheduledBatchSampler(concat, [(32, 32)], [100], [lambda s: 1.0], batch_size=8, base_seed=42, rank=1)

    batch0 = next(iter(sampler0))
    batch1 = next(iter(sampler1))
    assert batch0 != batch1


def test_batch_sampler_increments_step():
    """Test sampler increments step counter."""
    ds = TensorDataset(torch.randn(20, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(concat, [(32, 32)], [20], [lambda s: 1.0], batch_size=4, base_seed=42, rank=0)

    assert sampler.step == 0
    next(iter(sampler))
    assert sampler.step == 1


def test_batch_sampler_all_weights_zero():
    """Test sampler raises RuntimeError when all weights are zero."""
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    # Both datasets have weight 0
    sampler = ScheduledBatchSampler(
        concat, [(32, 32), (32, 32)], [10, 15], [lambda s: 0.0, lambda s: 0.0], batch_size=4, base_seed=42, rank=0
    )

    iterator = iter(sampler)
    with pytest.raises(RuntimeError, match="All dataset weights are zero"):
        next(iterator)


def test_batch_sampler_single_dataset():
    """Test sampler works correctly with single dataset."""
    ds = TensorDataset(torch.randn(30, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(concat, [(32, 32)], [30], [lambda s: 1.0], batch_size=4, base_seed=42, rank=0)

    batch = next(iter(sampler))
    assert len(batch) == 4
    assert all(0 <= idx < len(concat) for idx in batch)


def test_batch_sampler_negative_weights():
    """Test sampler handles negative weights by raising RuntimeError when combined weight is zero or negative."""
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    # Both datasets in same bucket: 1.0*10 + (-1.0)*15 = -5 < 0
    sampler = ScheduledBatchSampler(
        concat, [(32, 32), (32, 32)], [10, 15], [lambda s: 1.0, lambda s: -1.0], batch_size=4, base_seed=42, rank=0
    )

    iterator = iter(sampler)
    with pytest.raises(RuntimeError, match="All dataset weights are zero"):
        next(iterator)


def test_batch_sampler_handles_large_steps():
    """Test sampler properly handles large step values that drop weight to zero."""
    ds = TensorDataset(torch.randn(10, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    # Schedule drops weight to zero after step 2
    def step_schedule(step: int) -> float:
        return 1.0 if step < 2 else 0.0

    sampler = ScheduledBatchSampler(concat, [(32, 32)], [10], [step_schedule], batch_size=4, base_seed=42, rank=0)

    # At step 0, should sample normally
    batch1 = next(iter(sampler))
    assert len(batch1) == 4

    # At step 1, should still sample normally
    batch2 = next(iter(sampler))
    assert len(batch2) == 4

    # After step 2, weight becomes zero, should raise RuntimeError
    with pytest.raises(RuntimeError, match="All dataset weights are zero"):
        next(iter(sampler))
