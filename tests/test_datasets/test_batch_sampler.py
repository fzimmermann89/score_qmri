"""Tests for ScheduledBatchSampler."""

import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from score_qmri.datasets.batch_sampler import ScheduledBatchSampler
from score_qmri.datasets.kind import DatasetKind


def test_batch_sampler_produces_correct_batch_size():
    ds = TensorDataset(torch.randn(20, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(
        concat,
        [20],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    batch = next(iter(sampler))
    assert len(batch) == 4


def test_batch_sampler_indices_in_valid_range():
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [10, 15],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 1.0] * 2,
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    batch = next(iter(sampler))
    assert all(0 <= idx < len(concat) for idx in batch)


def test_batch_sampler_respects_zero_weight():
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [10, 15],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 0.0, lambda s: 1.0],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    batch = next(iter(sampler))
    assert all(idx >= 10 for idx in batch)


def test_batch_sampler_different_ranks_differ():
    ds = TensorDataset(torch.randn(100, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler0 = ScheduledBatchSampler(
        concat,
        [100],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=8,
        base_seed=42,
        rank=0,
    )
    sampler1 = ScheduledBatchSampler(
        concat,
        [100],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=8,
        base_seed=42,
        rank=1,
    )

    batch0 = next(iter(sampler0))
    batch1 = next(iter(sampler1))
    assert batch0 != batch1


def test_batch_sampler_increments_step():
    ds = TensorDataset(torch.randn(20, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(
        concat,
        [20],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    assert sampler.step == 0
    next(iter(sampler))
    assert sampler.step == 1


def test_batch_sampler_all_weights_zero():
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [10, 15],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 0.0, lambda s: 0.0],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    iterator = iter(sampler)
    with pytest.raises(RuntimeError, match="All dataset weights are zero"):
        next(iterator)


def test_batch_sampler_single_dataset():
    ds = TensorDataset(torch.randn(30, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(
        concat,
        [30],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    batch = next(iter(sampler))
    assert len(batch) == 4
    assert all(0 <= idx < len(concat) for idx in batch)


def test_batch_sampler_negative_weights():
    ds1 = TensorDataset(torch.randn(10, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(15, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [10, 15],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 1.0, lambda s: -1.0],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    iterator = iter(sampler)
    with pytest.raises(RuntimeError, match="All dataset weights are zero"):
        next(iterator)


def test_batch_sampler_handles_large_steps():
    ds = TensorDataset(torch.randn(10, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    def step_schedule(step: int) -> float:
        return 1.0 if step < 2 else 0.0

    sampler = ScheduledBatchSampler(
        concat,
        [10],
        [DatasetKind.T1MAP],
        [step_schedule],
        batch_size=4,
        base_seed=42,
        rank=0,
    )

    batch1 = next(iter(sampler))
    assert len(batch1) == 4

    batch2 = next(iter(sampler))
    assert len(batch2) == 4

    with pytest.raises(RuntimeError, match="All dataset weights are zero"):
        next(iter(sampler))


def test_batch_produces_mixed_kinds():
    # given: two datasets with different kinds, large batch
    ds1 = TensorDataset(torch.randn(100, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(100, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [100, 100],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 1.0, lambda s: 1.0],
        batch_size=32,
        kind_normalization=0.5,
        base_seed=42,
        rank=0,
    )

    # when: sample a batch
    batch = next(iter(sampler))

    # then: batch should contain indices from both datasets
    from_ds1 = [idx for idx in batch if idx < 100]
    from_ds2 = [idx for idx in batch if idx >= 100]
    assert len(from_ds1) > 0, "Expected some indices from dataset 1"
    assert len(from_ds2) > 0, "Expected some indices from dataset 2"


def test_kind_normalization_zero_is_size_proportional():
    # given: two datasets with different sizes, kind_normalization=0
    ds1 = TensorDataset(torch.randn(100, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(900, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [100, 900],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 1.0, lambda s: 1.0],
        batch_size=1000,
        kind_normalization=0.0,
        base_seed=42,
        rank=0,
    )

    # when: sample many batches
    counts = [0, 0]
    for _ in range(10):
        batch = sampler._next_batch()
        for idx in batch:
            counts[0 if idx < 100 else 1] += 1

    # then: ratio should be approximately 1:9 (10% vs 90%)
    total = sum(counts)
    ratio_ds1 = counts[0] / total
    assert 0.05 < ratio_ds1 < 0.15, f"Expected ~10% from ds1, got {ratio_ds1:.1%}"


def test_kind_normalization_one_is_kind_uniform():
    # given: two datasets with different sizes but different kinds, kind_normalization=1
    ds1 = TensorDataset(torch.randn(100, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(900, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [100, 900],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 1.0, lambda s: 1.0],
        batch_size=1000,
        kind_normalization=1.0,
        base_seed=42,
        rank=0,
    )

    # when: sample many batches
    counts = [0, 0]
    for _ in range(10):
        batch = sampler._next_batch()
        for idx in batch:
            counts[0 if idx < 100 else 1] += 1

    # then: ratio should be approximately 1:1 (50% each) since each kind gets equal probability
    total = sum(counts)
    ratio_ds1 = counts[0] / total
    assert 0.40 < ratio_ds1 < 0.60, f"Expected ~50% from ds1, got {ratio_ds1:.1%}"


def test_schedule_with_kind_normalization_one():
    # given: one dataset phased out at step 0, kind_normalization=1
    ds1 = TensorDataset(torch.randn(100, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(100, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2])

    sampler = ScheduledBatchSampler(
        concat,
        [100, 100],
        [DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 0.0, lambda s: 1.0],
        batch_size=16,
        kind_normalization=1.0,
        base_seed=42,
        rank=0,
    )

    # when: sample a batch
    batch = next(iter(sampler))

    # then: all indices should be from dataset 2 (phased-out dataset has zero weight)
    assert all(idx >= 100 for idx in batch), "Expected all indices from ds2"


def test_multiple_datasets_same_kind_with_kind_normalization_one():
    # given: 3 datasets - 2 of kind T1MAP (sizes 100, 300), 1 of kind T2MAP (size 200)
    ds1 = TensorDataset(torch.randn(100, 3, 32, 32))
    ds2 = TensorDataset(torch.randn(300, 3, 32, 32))
    ds3 = TensorDataset(torch.randn(200, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds1, ds2, ds3])

    sampler = ScheduledBatchSampler(
        concat,
        [100, 300, 200],
        [DatasetKind.T1MAP, DatasetKind.T1MAP, DatasetKind.T2MAP],
        [lambda s: 1.0, lambda s: 1.0, lambda s: 1.0],
        batch_size=1000,
        kind_normalization=1.0,
        base_seed=42,
        rank=0,
    )

    # when: sample many batches
    counts = [0, 0, 0]
    for _ in range(10):
        batch = sampler._next_batch()
        for idx in batch:
            if idx < 100:
                counts[0] += 1
            elif idx < 400:
                counts[1] += 1
            else:
                counts[2] += 1

    # then: T1MAP kind gets 50%, T2MAP kind gets 50%
    # Within T1MAP, ds1:ds2 should be 100:300 = 1:3 (so ds1=12.5%, ds2=37.5%)
    total = sum(counts)
    ratio_t1map = (counts[0] + counts[1]) / total
    ratio_t2map = counts[2] / total

    assert 0.40 < ratio_t1map < 0.60, f"Expected ~50% T1MAP, got {ratio_t1map:.1%}"
    assert 0.40 < ratio_t2map < 0.60, f"Expected ~50% T2MAP, got {ratio_t2map:.1%}"

    # Within T1MAP kind, ds1 should be ~25% of T1MAP (100/(100+300))
    t1map_total = counts[0] + counts[1]
    if t1map_total > 0:
        ds1_within_t1 = counts[0] / t1map_total
        assert 0.15 < ds1_within_t1 < 0.35, f"Expected ~25% ds1 within T1MAP, got {ds1_within_t1:.1%}"


def test_kind_normalization_validation():
    ds = TensorDataset(torch.randn(10, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    with pytest.raises(ValueError, match="kind_normalization must be in"):
        ScheduledBatchSampler(
            concat,
            [10],
            [DatasetKind.T1MAP],
            [lambda s: 1.0],
            batch_size=4,
            kind_normalization=1.5,
        )

    with pytest.raises(ValueError, match="kind_normalization must be in"):
        ScheduledBatchSampler(
            concat,
            [10],
            [DatasetKind.T1MAP],
            [lambda s: 1.0],
            batch_size=4,
            kind_normalization=-0.1,
        )


def test_max_steps_controls_length():
    ds = TensorDataset(torch.randn(100, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(
        concat,
        [100],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=4,
        max_steps=500,
    )

    assert len(sampler) == 500


def test_max_steps_none_uses_default_length():
    ds = TensorDataset(torch.randn(100, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    with pytest.warns(UserWarning, match="max_steps not set"):
        sampler = ScheduledBatchSampler(
            concat,
            [100],
            [DatasetKind.T1MAP],
            [lambda s: 1.0],
            batch_size=4,
            max_steps=None,
        )

    assert len(sampler) == 100 // 4


def test_set_epoch_does_not_reset_step():
    ds = TensorDataset(torch.randn(20, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    sampler = ScheduledBatchSampler(
        concat,
        [20],
        [DatasetKind.T1MAP],
        [lambda s: 1.0],
        batch_size=4,
        max_steps=100,
    )

    sampler._next_batch()
    sampler._next_batch()
    assert sampler.step == 2

    sampler.set_epoch(1)
    assert sampler.step == 2


def test_schedule_continues_across_epochs():
    ds = TensorDataset(torch.randn(20, 3, 32, 32))
    concat: ConcatDataset = ConcatDataset([ds])

    steps_seen: list[int] = []

    def tracking_schedule(step: int) -> float:
        steps_seen.append(step)
        return 1.0

    sampler = ScheduledBatchSampler(
        concat,
        [20],
        [DatasetKind.T1MAP],
        [tracking_schedule],
        batch_size=4,
        max_steps=5,
    )

    # given: first "epoch"
    for _ in sampler:
        pass

    # when: simulate epoch boundary and second "epoch"
    sampler.set_epoch(1)
    for _ in sampler:
        pass

    # then: steps should be continuous 0..9
    assert steps_seen == list(range(10))


def test_resume_reseeds_without_restarting_schedule():
    dataset = ConcatDataset([TensorDataset(torch.arange(1000))])
    seen = []
    sampler = ScheduledBatchSampler(
        dataset, [1000], [DatasetKind.T1MAP], [lambda step: seen.append(step) or 1.0], batch_size=16, max_steps=100
    )
    original = next(iter(sampler))
    sampler.reset_step(50)
    sampler.set_epoch(0)
    resumed = next(iter(sampler))
    assert seen == [0, 50]
    assert resumed != original
    sampler.reset_step(50)
    sampler.set_epoch(0)
    assert next(iter(sampler)) == resumed
