"""Tests for DDIM sampler."""

import torch

from models.diffusion import DiffusionModel
from sampling.ddim_sampler import DDIMSampler, create_fixed_noise


def test_ddim_sampler_produces_correct_shape():
    """Test DDIM sampler produces correct output shape."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    sampler = DDIMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, torch.device("cpu"), kind)

    assert samples.shape == shape


def test_ddim_sampler_accepts_fixed_noise():
    """Test DDIM sampler accepts initial noise."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    sampler = DDIMSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])
    initial_noise = torch.randn(shape)

    samples = sampler.sample(shape, torch.device("cpu"), kind, initial_noise=initial_noise)
    assert samples.shape == shape


def test_ddim_sampler_deterministic_with_same_noise():
    """Test DDIM sampler is deterministic with same initial noise."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    sampler = DDIMSampler(model, num_steps=10, eta=0.0)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])
    initial_noise = torch.randn(shape)

    samples1 = sampler.sample(shape, torch.device("cpu"), kind, initial_noise=initial_noise.clone())
    samples2 = sampler.sample(shape, torch.device("cpu"), kind, initial_noise=initial_noise.clone())

    assert torch.allclose(samples1, samples2)


def test_create_fixed_noise():
    """Test fixed noise creation for each kind."""
    num_kinds = 3
    image_shape = (64, 64)
    fixed_noise = create_fixed_noise(num_kinds, image_shape)

    assert len(fixed_noise) == num_kinds
    for kind_idx in range(num_kinds):
        assert kind_idx in fixed_noise
        assert fixed_noise[kind_idx].shape == (1, 2, 64, 64)


def test_create_fixed_noise_reproducible():
    """Test fixed noise is reproducible with same seed."""
    noise1 = create_fixed_noise(3, (32, 32), seed=42)
    noise2 = create_fixed_noise(3, (32, 32), seed=42)

    for kind_idx in range(3):
        assert torch.allclose(noise1[kind_idx], noise2[kind_idx])


def test_create_fixed_noise_different_seeds():
    """Test fixed noise differs with different seeds."""
    noise1 = create_fixed_noise(3, (32, 32), seed=42)
    noise2 = create_fixed_noise(3, (32, 32), seed=123)

    for kind_idx in range(3):
        assert not torch.allclose(noise1[kind_idx], noise2[kind_idx])
