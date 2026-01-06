"""Tests for DDIM sampler."""

import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.ddim_sampler import DDIMSampler, create_fixed_noise


def test_ddim_sampler_shape():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=10)

    shape = (2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, kind)
    assert samples.shape == shape


def test_ddim_sampler_fixed_noise():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=10)

    shape = (1, 32, 32)
    kind = torch.tensor([0])
    noise = torch.randn(shape)

    samples = sampler.sample(shape, kind, initial_noise=noise)
    assert samples.shape == shape


def test_ddim_deterministic():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=10, eta=0.0)

    shape = (1, 32, 32)
    kind = torch.tensor([0])
    noise = torch.randn(shape)

    s1 = sampler.sample(shape, kind, noise.clone())
    s2 = sampler.sample(shape, kind, noise.clone())
    assert torch.allclose(s1, s2)


def test_ddim_complex_input():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=10, eta=0.0)

    shape = (1, 32, 32)
    kind = torch.tensor([0])
    noise = torch.complex(torch.randn(shape), torch.randn(shape))

    samples = sampler.sample(shape, kind, noise)
    assert samples.shape == shape
    assert samples.is_complex()


def test_create_fixed_noise():
    from score_qmri.datasets.kind import NUM_DATASET_KINDS

    noise = create_fixed_noise((64, 64))
    assert len(noise) == NUM_DATASET_KINDS

    for kind in DatasetKind:
        n = noise[int(kind)]
        assert n.shape == (1, 64, 64)
        if kind.is_complex:
            assert n.is_complex()
        else:
            assert not n.is_complex()


def test_create_fixed_noise_reproducible():
    from score_qmri.datasets.kind import NUM_DATASET_KINDS

    n1 = create_fixed_noise((32, 32), seed=42)
    n2 = create_fixed_noise((32, 32), seed=42)
    for k in range(NUM_DATASET_KINDS):
        assert torch.allclose(n1[k], n2[k])


def test_ddim_sampler_uses_model_device():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=10)

    model = model.to(torch.device("cpu"))
    shape = (1, 32, 32)
    kind = torch.tensor([0])

    samples = sampler.sample(shape, kind)
    assert samples.device.type == "cpu"


def test_ddim_sampler_device_changes_with_model():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=10)

    shape = (1, 32, 32)
    kind = torch.tensor([0])

    samples_cpu = sampler.sample(shape, kind)
    assert samples_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
        samples_cuda = sampler.sample(shape, kind)
        assert samples_cuda.device.type == "cuda"


def test_ddim_collect_stats():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDIMSampler(model, num_steps=50)

    shape = (1, 32, 32)
    kind = torch.tensor([0])

    samples = sampler.sample(shape, kind, collect_stats=True)
    assert samples.shape == shape
    assert sampler.last_stats is not None

    stats = sampler.last_stats
    assert len(stats.steps) > 0
    assert len(stats.intermediates) == len(stats.steps)
    assert len(stats.x0_std) == len(stats.steps)
    assert len(stats.t_values) == len(stats.steps)

    for inter in stats.intermediates:
        assert inter.shape == shape
