"""Tests for DDPM sampler."""

import torch

from models.diffusion import DiffusionModel
from sampling.ddpm_sampler import DDPMSampler


def test_ddpm_sampler_produces_correct_shape():
    """Test DDPM sampler produces output of requested shape."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    sampler = DDPMSampler(model, num_steps=5)

    samples = sampler.sample((2, 2, 32, 32), torch.device("cpu"))
    assert samples.shape == (2, 2, 32, 32)


def test_ddpm_sampler_accepts_conditioning():
    """Test DDPM sampler accepts conditioning tensor."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    sampler = DDPMSampler(model, num_steps=5)

    kind = torch.tensor([0, 1])
    samples = sampler.sample((2, 2, 32, 32), torch.device("cpu"), kind)
    assert samples.shape == (2, 2, 32, 32)


def test_ddpm_sampler_conditional_alias():
    """Test sample_conditional is alias for sample."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    sampler = DDPMSampler(model, num_steps=5)

    kind = torch.tensor([0, 1])
    samples = sampler.sample_conditional((2, 2, 32, 32), torch.device("cpu"), kind)
    assert samples.shape == (2, 2, 32, 32)


def test_ddpm_sampler_respects_num_steps():
    """Test sampler uses specified number of steps."""
    model = DiffusionModel(num_kinds=3, num_timesteps=1000)
    sampler = DDPMSampler(model, num_steps=10)

    assert sampler.num_steps == 10
