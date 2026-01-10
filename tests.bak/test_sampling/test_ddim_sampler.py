"""Tests for DDIM sampler."""

import torch

from models.diffusion import DiffusionModel
from sampling.ddim_sampler import DDIMSampler, create_fixed_noise


def test_ddim_sampler_shape():
    """Test DDIM sampler output shape."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDIMSampler(model, num_steps=10)

    shape = (2, 2, 32, 32)
    kind = torch.tensor([0, 1])
    samples = sampler.sample(shape, kind)
    assert samples.shape == shape


def test_ddim_sampler_fixed_noise():
    """Test DDIM sampler accepts fixed noise."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDIMSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])
    noise = torch.randn(shape)

    samples = sampler.sample(shape, kind, initial_noise=noise)
    assert samples.shape == shape


def test_ddim_deterministic():
    """Test DDIM is deterministic with eta=0."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDIMSampler(model, num_steps=10, eta=0.0)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])
    noise = torch.randn(shape)

    s1 = sampler.sample(shape, kind, noise.clone())
    s2 = sampler.sample(shape, kind, noise.clone())
    assert torch.allclose(s1, s2)


def test_create_fixed_noise():
    """Test fixed noise creation."""
    from datasets.kind import NUM_DATASET_KINDS

    noise = create_fixed_noise((64, 64))
    assert len(noise) == NUM_DATASET_KINDS
    for k in range(NUM_DATASET_KINDS):
        assert noise[k].shape == (1, 2, 64, 64)


def test_create_fixed_noise_reproducible():
    """Test fixed noise is reproducible."""
    from datasets.kind import NUM_DATASET_KINDS

    n1 = create_fixed_noise((32, 32), seed=42)
    n2 = create_fixed_noise((32, 32), seed=42)
    for k in range(NUM_DATASET_KINDS):
        assert torch.allclose(n1[k], n2[k])


def test_ddim_sampler_uses_model_device():
    """Test DDIM sampler uses device from model."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDIMSampler(model, num_steps=10)

    # Put model on CPU
    model = model.to(torch.device("cpu"))
    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])

    samples = sampler.sample(shape, kind)
    assert samples.device.type == "cpu"


def test_ddim_sampler_device_changes_with_model():
    """Test DDIM sampler follows model device changes."""
    model = DiffusionModel(base_channels=16, num_blocks=2)
    sampler = DDIMSampler(model, num_steps=10)

    shape = (1, 2, 32, 32)
    kind = torch.tensor([0])

    # Sample on CPU
    samples_cpu = sampler.sample(shape, kind)
    assert samples_cpu.device.type == "cpu"

    if torch.cuda.is_available():
        # Move model to CUDA
        model = model.to(torch.device("cuda:0"))
        samples_cuda = sampler.sample(shape, kind)
        assert samples_cuda.device.type == "cuda"
