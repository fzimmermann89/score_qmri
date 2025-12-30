"""Tests for DDPM sampler."""

import torch

from models.diffusion import DiffusionModel
from sampling.ddpm_sampler import DDPMSampler


def test_ddpm_sampler_shape():
    """Test DDPM sampler output shape."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, torch.device("cpu"), kind)
    assert samples.shape == shape


def test_ddpm_sampler_default_kind():
    """Test DDPM sampler with default kind."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    samples = sampler.sample(shape, torch.device("cpu"))
    assert samples.shape == shape


def test_ddpm_sampler_num_steps():
    """Test DDPM sampler respects num_steps."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDPMSampler(model, num_steps=5)
    assert sampler.num_steps == 5

    sampler_default = DDPMSampler(model, num_steps=100)
    assert sampler_default.num_steps == 100
