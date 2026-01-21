"""Tests for EDM sampler."""

import torch

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.edm_sampler import EDMSampler, create_fixed_noise


def test_edm_sampler_shape():
    model = DiffusionModel(base_channels=16, num_res_blocks=1, channel_mult=(1, 2))
    sampler = EDMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([int(DatasetKind.T1MAP), int(DatasetKind.T1W_MAGNITUDE)])
    samples = sampler.sample(shape, kind)
    assert samples.shape == shape


def test_edm_sampler_fixed_noise_deterministic():
    model = DiffusionModel(base_channels=16, num_res_blocks=1, channel_mult=(1, 2))
    sampler = EDMSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([int(DatasetKind.T1MAP)])
    noise = torch.randn(shape)

    s1 = sampler.sample(shape, kind, initial_noise=noise.clone())
    s2 = sampler.sample(shape, kind, initial_noise=noise.clone())
    assert torch.allclose(s1, s2)


def test_edm_real_kind_imag_stays_zero():
    model = DiffusionModel(base_channels=16, num_res_blocks=1, channel_mult=(1, 2))
    sampler = EDMSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([int(DatasetKind.T1MAP)])
    samples = sampler.sample(shape, kind)
    assert torch.allclose(samples[:, 1], torch.zeros_like(samples[:, 1]))


def test_create_fixed_noise():
    from score_qmri.datasets.kind import NUM_DATASET_KINDS

    noise = create_fixed_noise((64, 64))
    assert len(noise) == NUM_DATASET_KINDS

    for kind in DatasetKind:
        n = noise[int(kind)]
        assert n.shape == (4, 2, 64, 64)
        if is_complex_kind(int(kind)):
            assert not torch.allclose(n[:, 1], torch.zeros_like(n[:, 1]))
        else:
            assert torch.allclose(n[:, 1], torch.zeros_like(n[:, 1]))


def test_create_fixed_noise_reproducible():
    from score_qmri.datasets.kind import NUM_DATASET_KINDS

    n1 = create_fixed_noise((32, 32), seed=42)
    n2 = create_fixed_noise((32, 32), seed=42)
    for k in range(NUM_DATASET_KINDS):
        assert torch.allclose(n1[k], n2[k])


def test_edm_collect_stats():
    model = DiffusionModel(base_channels=16, num_res_blocks=1, channel_mult=(1, 2))
    sampler = EDMSampler(model, num_steps=25)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([int(DatasetKind.T1MAP)])
    samples = sampler.sample(shape, kind, collect_stats=True)
    assert samples.shape == shape
    assert sampler.last_stats is not None

    stats = sampler.last_stats
    assert len(stats.steps) > 0
    assert len(stats.intermediates) == len(stats.steps)
    assert len(stats.x0_std) == len(stats.steps)
    assert len(stats.sigma) == len(stats.steps)

    for inter in stats.intermediates:
        assert inter.shape == shape
