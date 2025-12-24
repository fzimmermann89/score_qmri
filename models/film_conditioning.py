"""FiLM (Feature-wise Linear Modulation) conditioning for dataset kinds."""

import torch.nn as nn
from einops import rearrange
from torch import Tensor


def apply_film(features: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply FiLM conditioning to feature maps.

    Parameters
    ----------
    features : Tensor
        Feature maps of shape (B, C, H, W).
    scale : Tensor
        Scale parameters of shape (B, 1) or (B, C) or (B, C, 1, 1).
    shift : Tensor
        Shift parameters of shape (B, 1) or (B, C) or (B, C, 1, 1).

    Returns
    -------
    Tensor
        Modulated features: scale * features + shift.
    """
    if scale.ndim == 2:
        scale = rearrange(scale, "b c -> b c 1 1")
        shift = rearrange(shift, "b c -> b c 1 1")
    return scale * features + shift


class FilmConditioning(nn.Module):
    """MLP that maps dataset kind to FiLM scale/shift parameters.

    Uses learned embeddings for each kind and generates per-block
    FiLM parameters through a small MLP.
    """

    def __init__(
        self,
        num_kinds: int,
        embedding_dim: int = 64,
        num_blocks: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize FiLM conditioning MLP.

        Parameters
        ----------
        num_kinds : int
            Number of different dataset kinds.
        embedding_dim : int
            Dimension of kind embedding.
        num_blocks : int
            Number of UNet blocks to condition.
        hidden_dim : int
            Hidden dimension of MLP.
        """
        super().__init__()
        self.num_blocks = num_blocks

        self.kind_embedding = nn.Embedding(num_kinds, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_blocks * 2),
        )

    def forward(self, kind: Tensor) -> dict[str, Tensor]:
        """Generate FiLM parameters from dataset kind.

        Parameters
        ----------
        kind : Tensor
            Kind tensor of shape (B,) with integer values.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with 'scales' and 'shifts', each of shape (B, num_blocks).
        """
        embedded = self.kind_embedding(kind)
        film_params = rearrange(self.mlp(embedded), "b (n p) -> b n p", p=2)
        return {
            "scales": film_params[:, :, 0],
            "shifts": film_params[:, :, 1],
        }
