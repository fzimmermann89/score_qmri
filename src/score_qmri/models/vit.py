"""Vision Transformer for JEPA pretraining."""

import torch
from einops import rearrange
from torch import Tensor


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Computes in float32 for numerical stability, then casts back.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_fp32 * rms).to(x.dtype)
        return out * self.weight


class PatchEmbed(torch.nn.Module):
    """2D image to patch embedding.

    Parameters
    ----------
    img_size
        Input image size (assumes square).
    patch_size
        Patch size (assumes square).
    in_chans
        Number of input channels (2 for complex MRI).
    embed_dim
        Embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 2,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H', W')
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class Attention(torch.nn.Module):
    """Multi-head self-attention using SDPA.

    Parameters
    ----------
    dim
        Input embedding dimension.
    num_heads
        Number of attention heads.
    qkv_bias
        Whether to include bias in QKV projection.
    attn_drop
        Dropout rate for attention weights.
    proj_drop
        Dropout rate for output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:  # (B, N, D) -> (B, N, D)
        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3*D)
        q, k, v = rearrange(
            qkv,
            "b n (three h d) -> three b h n d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(0)

        dropout = self.attn_drop if self.training else 0.0
        out: Tensor = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=False)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(torch.nn.Module):
    """Feed-forward network.

    Parameters
    ----------
    in_features
        Input dimension.
    hidden_features
        Hidden layer dimension (defaults to 4x input).
    out_features
        Output dimension (defaults to input).
    drop
        Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(torch.nn.Module):
    """Transformer block with Pre-Norm.

    Parameters
    ----------
    dim
        Embedding dimension.
    num_heads
        Number of attention heads.
    mlp_ratio
        MLP hidden dimension ratio.
    qkv_bias
        Whether to include bias in QKV projection.
    drop
        Dropout rate for MLP and projection.
    attn_drop
        Dropout rate for attention weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(torch.nn.Module):
    """Vision Transformer for JEPA pretraining.

    Returns sequence of patch features (no CLS token).

    Parameters
    ----------
    img_size
        Input image size (assumes square).
    patch_size
        Patch size (assumes square).
    in_chans
        Number of input channels (2 for complex MRI).
    embed_dim
        Embedding dimension.
    depth
        Number of transformer blocks.
    num_heads
        Number of attention heads.
    mlp_ratio
        MLP hidden dimension ratio.
    qkv_bias
        Whether to include bias in QKV projection.
    drop_rate
        Dropout rate for MLP and embeddings.
    attn_drop_rate
        Dropout rate for attention weights.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 2,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = torch.nn.Dropout(drop_rate)

        self.blocks = torch.nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate) for _ in range(depth)]
        )

        self.norm = RMSNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, ids_keep: Tensor | None = None) -> Tensor:
        """Forward pass with optional token dropping.

        Parameters
        ----------
        x
            Input images: (B, C, H, W)
        ids_keep
            Optional indices of tokens to keep: (B, N_keep).
            If provided, only these tokens are processed (for masked encoding).

        Returns
        -------
        Tensor
            Output features: (B, N, D) or (B, N_keep, D) if ids_keep provided.
        """
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if ids_keep is not None:
            B, N_keep = ids_keep.shape
            D = x.shape[-1]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(B, N_keep, D))

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
