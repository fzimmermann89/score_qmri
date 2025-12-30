"""Tests for SDE-based samplers."""

import torch

from models.diffusion import DiffusionModel
from sampling.sde_sampler import ProbabilityFlowSampler, SDESampler


def test_sde_sampler_shape():
    """Test SDE sampler output shape."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = SDESampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, torch.device("cpu"), kind)
    assert samples.shape == shape


def test_sde_sampler_default_kind():
    """Test SDE sampler with default kind."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = SDESampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    samples = sampler.sample(shape, torch.device("cpu"))
    assert samples.shape == shape


def test_sde_sampler_fixed_noise():
    """Test SDE sampler accepts initial noise."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = SDESampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    noise = torch.randn(shape)
    kind = torch.tensor([0])

    samples = sampler.sample(shape, torch.device("cpu"), kind, initial_noise=noise)
    assert samples.shape == shape


def test_probability_flow_sampler_shape():
    """Test probability flow sampler output shape."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = ProbabilityFlowSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, torch.device("cpu"), kind)
    assert samples.shape == shape


def test_probability_flow_deterministic():
    """Test probability flow sampler is deterministic."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = ProbabilityFlowSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    noise = torch.randn(shape)
    kind = torch.tensor([0])

    s1 = sampler.sample(shape, torch.device("cpu"), kind, noise.clone())
    s2 = sampler.sample(shape, torch.device("cpu"), kind, noise.clone())
    assert torch.allclose(s1, s2)


def test_sde_sampler_num_steps():
    """Test SDE sampler respects num_steps."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = SDESampler(model, num_steps=50)
    assert sampler.num_steps == 50

