"""Tests for conditioning module."""

import torch

from models.conditioning import ConditioningMLP, SinusoidalEmbedding


def test_sinusoidal_embedding():
    """Test sinusoidal embedding shape."""
    embed = SinusoidalEmbedding(dim=64)
    t = torch.tensor([0, 100, 500, 999])
    out = embed(t)
    assert out.shape == (4, 64)


def test_sinusoidal_embedding_different_timesteps():
    """Test different timesteps produce different embeddings."""
    embed = SinusoidalEmbedding(dim=64)
    t1 = torch.tensor([0])
    t2 = torch.tensor([500])
    assert not torch.allclose(embed(t1), embed(t2))


def test_conditioning_mlp():
    """Test conditioning MLP output shape."""
    mlp = ConditioningMLP(cond_dim=128)
    t = torch.tensor([10, 50])
    kind = torch.tensor([0, 1])
    cond = mlp(t, kind)
    assert cond.shape == (2, 128)


def test_conditioning_mlp_different_kinds():
    """Test different kinds produce different conditioning."""
    mlp = ConditioningMLP(cond_dim=128)
    t = torch.tensor([50, 50])
    kind1 = torch.tensor([0, 0])
    kind2 = torch.tensor([1, 1])
    cond1 = mlp(t, kind1)
    cond2 = mlp(t, kind2)
    assert not torch.allclose(cond1, cond2)


def test_conditioning_mlp_different_timesteps():
    """Test different timesteps produce different conditioning."""
    mlp = ConditioningMLP(cond_dim=128)
    t1 = torch.tensor([10, 10])
    t2 = torch.tensor([500, 500])
    kind = torch.tensor([0, 0])
    cond1 = mlp(t1, kind)
    cond2 = mlp(t2, kind)
    assert not torch.allclose(cond1, cond2)
