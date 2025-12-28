"""Conditioning embeddings for diffusion models."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        args = t[:, None].float() * freqs[None, :]
        emb: Tensor = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb


class ConditioningMLP(nn.Module):
    """MLP that combines timestep and dataset kind into conditioning vector.

    Args:
        num_kinds: Number of dataset types.
        cond_dim: Output conditioning dimension.
        time_embed_dim: Intermediate timestep embedding dimension.
    """

    def __init__(
        self,
        num_kinds: int = 3,
        cond_dim: int = 128,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.kind_embed = nn.Embedding(num_kinds, cond_dim // 2)

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim + cond_dim // 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: Tensor, kind: Tensor) -> Tensor:
        """Embed timestep and kind into conditioning vector.

        Args:
            t: Timesteps of shape (B,).
            kind: Dataset kind indices of shape (B,).

        Returns:
            Conditioning vector of shape (B, cond_dim).
        """
        t_emb = self.time_embed(t)
        k_emb = self.kind_embed(kind)
        cond: Tensor = self.mlp(torch.cat([t_emb, k_emb], dim=-1))
        return cond
