"""Tests for DDPM sampler."""

import torch

from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.ddpm_sampler import DDPMSampler


def test_ddpm_sampler_shape():
    """Test DDPM sampler output shape."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, kind)
    assert samples.shape == shape


def test_ddpm_sampler_default_kind():
    """Test DDPM sampler with default kind."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    samples = sampler.sample(shape)
    assert samples.shape == shape


def test_ddpm_sampler_num_steps():
    """Test DDPM sampler respects num_steps."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=5)
    assert sampler.num_steps == 5

    sampler_default = DDPMSampler(model, num_steps=100)
    assert sampler_default.num_steps == 100


def test_ddpm_sampler_uses_model_device():
    """Test DDPM sampler uses device from model."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=10)

    # Put model on CPU
    model = model.to(torch.device("cpu"))
    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])

    samples = sampler.sample(shape, kind)
    assert samples.device.type == "cpu"


def test_ddpm_sampler_device_changes_with_model():
    """Test DDPM sampler follows model device changes."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])

    # Sample on CPU
    model = model.to(torch.device("cpu"))
    samples_cpu = sampler.sample(shape, kind)
    assert samples_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        # Move model to CUDA
        model = model.to(torch.device("cuda:0"))
        samples_cuda = sampler.sample(shape, kind)
        assert samples_cuda.device.type == "cuda"
