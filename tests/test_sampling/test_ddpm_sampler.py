"""Tests for DDPM sampler."""

import torch

from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.ddpm_sampler import DDPMSampler


def test_ddpm_sampler_shape():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDPMSampler(model, num_steps=10)

    shape = (2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, kind)
    assert samples.shape == shape


def test_ddpm_sampler_default_kind():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDPMSampler(model, num_steps=10)

    shape = (2, 32, 32)
    samples = sampler.sample(shape)
    assert samples.shape == shape


def test_ddpm_sampler_num_steps():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDPMSampler(model, num_steps=5)
    assert sampler.num_steps == 5

    sampler_default = DDPMSampler(model, num_steps=100)
    assert sampler_default.num_steps == 100


def test_ddpm_sampler_uses_model_device():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDPMSampler(model, num_steps=10)

    model = model.to(torch.device("cpu"))
    shape = (1, 32, 32)
    kind = torch.tensor([0])

    samples = sampler.sample(shape, kind)
    assert samples.device.type == "cpu"


def test_ddpm_sampler_device_changes_with_model():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDPMSampler(model, num_steps=10)

    shape = (1, 32, 32)
    kind = torch.tensor([0])

    model = model.to(torch.device("cpu"))
    samples_cpu = sampler.sample(shape, kind)
    assert samples_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
        samples_cuda = sampler.sample(shape, kind)
        assert samples_cuda.device.type == "cuda"


def test_ddpm_sampler_complex_input():
    model = DiffusionModel(base_channels=16, num_res_blocks=2, channel_mult=(1, 2))
    sampler = DDPMSampler(model, num_steps=10)

    shape = (1, 32, 32)
    kind = torch.tensor([0])
    noise = torch.complex(torch.randn(shape), torch.randn(shape))

    samples = sampler.sample(shape, kind, initial_noise=noise)
    assert samples.shape == shape
    assert samples.is_complex()
