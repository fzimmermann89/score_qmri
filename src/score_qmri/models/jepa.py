"""JEPA model for self-supervised MRI pretraining.

Joint Embedding Predictive Architecture learns representations by predicting
target block embeddings from context block embeddings in latent space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import repeat
from torch import Tensor

from .vit import Block, RMSNorm, VisionTransformer


@dataclass
class JEPAConfig:
    """JEPA hyperparameters.

    Parameters
    ----------
    img_size
        Input image size (assumes square).
    patch_size
        Patch size (assumes square).
    in_chans
        Number of input channels (2 for complex MRI).
    embed_dim
        Encoder embedding dimension.
    encoder_depth
        Number of encoder transformer blocks.
    num_heads
        Number of attention heads.
    mlp_ratio
        MLP hidden dimension ratio.
    qkv_bias
        Whether to include bias in QKV projection.
    drop_rate
        Dropout rate.
    attn_drop_rate
        Attention dropout rate.
    predictor_embed_dim
        Predictor embedding dimension (None = same as encoder).
    predictor_depth
        Number of predictor transformer blocks.
    target_aspect_ratio
        Aspect ratio range for target blocks (min, max).
    target_scale
        Scale range for target blocks as fraction of total patches.
    num_targets
        Number of target blocks to predict.
    ema_start
        Initial EMA momentum.
    ema_end
        Final EMA momentum.
    """

    img_size: int = 256
    patch_size: int = 16
    in_chans: int = 2

    embed_dim: int = 384
    encoder_depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    predictor_embed_dim: int | None = None
    predictor_depth: int = 6

    target_aspect_ratio: tuple[float, float] = (0.75, 1.5)
    target_scale: tuple[float, float] = (0.15, 0.2)
    num_targets: int = 4

    ema_start: float = 0.996
    ema_end: float = 1.0


def cosine_ema_schedule(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Cosine schedule for EMA momentum.

    Parameters
    ----------
    step
        Current training step.
    total_steps
        Total number of training steps.
    start
        Starting momentum value.
    end
        Ending momentum value.

    Returns
    -------
    float
        Current momentum value.
    """
    progress = step / max(total_steps, 1)
    return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2


def sample_block_mask(
    grid_size: int,
    scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    device: torch.device,
) -> Tensor:
    """Sample a single rectangular block mask.

    Parameters
    ----------
    grid_size
        Number of patches per side (H/patch_size).
    scale
        Scale range as fraction of total patches (min, max).
    aspect_ratio
        Aspect ratio range (min, max).
    device
        Target device.

    Returns
    -------
    Tensor
        Boolean mask (grid_size, grid_size) - True = in block.
    """
    num_patches = grid_size * grid_size

    # Sample scale and aspect ratio
    s = torch.empty(1, device=device).uniform_(scale[0], scale[1]).item()
    ar = torch.empty(1, device=device).uniform_(aspect_ratio[0], aspect_ratio[1]).item()

    # Compute block dimensions
    num_block_patches = int(num_patches * s)
    h = int(round(math.sqrt(num_block_patches * ar)))
    w = int(round(math.sqrt(num_block_patches / ar)))
    h = min(max(h, 1), grid_size)
    w = min(max(w, 1), grid_size)

    # Sample top-left corner
    top = torch.randint(0, max(grid_size - h + 1, 1), (1,), device=device).item()
    left = torch.randint(0, max(grid_size - w + 1, 1), (1,), device=device).item()

    # Create mask
    mask = torch.zeros(grid_size, grid_size, dtype=torch.bool, device=device)
    mask[top : top + h, left : left + w] = True

    return mask


class BlockMaskGenerator:
    """Generate target block masks for JEPA.

    Context = complement of union of target blocks.
    """

    def __init__(
        self,
        grid_size: int,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
    ) -> None:
        """
        Parameters
        ----------
        grid_size
            Number of patches per side.
        num_targets
            Number of target blocks.
        target_scale
            Scale range for each target block.
        target_aspect_ratio
            Aspect ratio range for each target block.
        """
        self.grid_size = grid_size
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio

    def __call__(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Generate context and target masks.

        Returns
        -------
        context_ids
            Indices of context tokens: (B, N_ctx)
        target_ids
            Indices of target tokens: (B, N_tgt)
        """
        all_context_ids = []
        all_target_ids = []

        for _ in range(batch_size):
            # Generate target blocks
            target_mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool, device=device)
            for _ in range(self.num_targets):
                block = sample_block_mask(
                    self.grid_size,
                    self.target_scale,
                    self.target_aspect_ratio,
                    device,
                )
                target_mask = target_mask | block

            # Context = complement of targets
            context_mask = ~target_mask

            # Convert to flat indices
            target_ids = target_mask.flatten().nonzero(as_tuple=False).squeeze(-1)
            context_ids = context_mask.flatten().nonzero(as_tuple=False).squeeze(-1)

            all_context_ids.append(context_ids)
            all_target_ids.append(target_ids)

        # Truncate to minimum size for batching
        min_ctx = min(ids.shape[0] for ids in all_context_ids)
        min_tgt = min(ids.shape[0] for ids in all_target_ids)

        context_ids = torch.stack([ids[:min_ctx] for ids in all_context_ids], dim=0)
        target_ids = torch.stack([ids[:min_tgt] for ids in all_target_ids], dim=0)

        return context_ids, target_ids


class Predictor(torch.nn.Module):
    """Predict target embeddings from context.

    Projects encoder dim -> predictor dim, runs transformer blocks,
    then projects back to encoder dim for loss computation.

    Parameters
    ----------
    num_patches
        Total number of patches.
    encoder_embed_dim
        Encoder embedding dimension.
    predictor_embed_dim
        Predictor embedding dimension (None = same as encoder).
    depth
        Number of transformer blocks.
    num_heads
        Number of attention heads.
    mlp_ratio
        MLP hidden dimension ratio.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        predictor_embed_dim: int | None = None,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        pred_dim = predictor_embed_dim or encoder_embed_dim

        # Project encoder -> predictor dim
        self.embed_proj = torch.nn.Linear(encoder_embed_dim, pred_dim)

        # Learnable mask token for target positions
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, pred_dim))

        # Positional embeddings for predictor
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, pred_dim))

        # Transformer blocks
        self.blocks = torch.nn.ModuleList([Block(pred_dim, num_heads, mlp_ratio, qkv_bias=True) for _ in range(depth)])

        self.norm = RMSNorm(pred_dim)

        # Project predictor -> encoder dim
        self.output_proj = torch.nn.Linear(pred_dim, encoder_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: Tensor,
        context_ids: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """Predict target embeddings from context.

        Parameters
        ----------
        context_tokens
            Encoded context patches: (B, N_ctx, D_enc)
        context_ids
            Indices of context patches: (B, N_ctx)
        target_ids
            Indices of target patches: (B, N_tgt)

        Returns
        -------
        Tensor
            Predicted target embeddings: (B, N_tgt, D_enc)
        """
        B, N_ctx, D = context_tokens.shape
        N_tgt = target_ids.shape[1]
        N = self.pos_embed.shape[1]
        pred_dim = self.pos_embed.shape[2]

        # Project context to predictor dim
        context_tokens = self.embed_proj(context_tokens)  # (B, N_ctx, pred_dim)

        # Build full sequence with mask tokens
        x = repeat(self.mask_token, "1 1 d -> b n d", b=B, n=N)  # (B, N, pred_dim)

        # Scatter context tokens into their positions
        context_ids_exp = context_ids.unsqueeze(-1).expand(B, N_ctx, pred_dim)
        x = x.scatter(1, context_ids_exp, context_tokens)

        # Add positional embeddings
        x = x + self.pos_embed

        # Run transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Extract target positions
        target_ids_exp = target_ids.unsqueeze(-1).expand(B, N_tgt, pred_dim)
        target_preds = torch.gather(x, dim=1, index=target_ids_exp)

        # Project back to encoder dim
        target_preds = self.output_proj(target_preds)

        return target_preds


class JEPA(torch.nn.Module):
    """Joint Embedding Predictive Architecture.

    Learns representations by predicting target block embeddings
    from context block embeddings in latent space.

    Parameters
    ----------
    config
        JEPA configuration. If None, uses default config.
    """

    def __init__(self, config: JEPAConfig | None = None) -> None:
        super().__init__()
        self.config = config or JEPAConfig()
        cfg = self.config

        grid_size = cfg.img_size // cfg.patch_size
        num_patches = grid_size * grid_size
        pred_embed_dim = cfg.predictor_embed_dim or cfg.embed_dim

        # Context encoder (trainable)
        self.context_encoder = VisionTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
        )

        # Target encoder (EMA copy, frozen)
        self.target_encoder = VisionTransformer(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.encoder_depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
        )

        # Predictor
        self.predictor = Predictor(
            num_patches=num_patches,
            encoder_embed_dim=cfg.embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=cfg.predictor_depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
        )

        # Mask generator
        self.mask_generator = BlockMaskGenerator(
            grid_size=grid_size,
            num_targets=cfg.num_targets,
            target_scale=cfg.target_scale,
            target_aspect_ratio=cfg.target_aspect_ratio,
        )

        # Initialize target encoder
        self._init_target_encoder()

    @property
    def num_patches(self) -> int:
        """Total number of patches."""
        return self.context_encoder.num_patches

    @property
    def embed_dim(self) -> int:
        """Encoder embedding dimension."""
        return self.config.embed_dim

    def _init_target_encoder(self) -> None:
        """Copy context encoder weights to target encoder and freeze."""
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self._freeze_target_encoder()

    def _freeze_target_encoder(self) -> None:
        """Freeze target encoder parameters."""
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x
            Input images: (B, C, H, W)

        Returns
        -------
        pred
            Predicted target embeddings: (B, N_tgt, D)
        target
            True target embeddings (normalized): (B, N_tgt, D)
        context_ids
            Context token indices: (B, N_ctx)
        target_ids
            Target token indices: (B, N_tgt)
        """
        B = x.shape[0]
        device = x.device

        # Generate masks
        context_ids, target_ids = self.mask_generator(B, device)

        # Target encoder: encode full image, extract targets
        with torch.no_grad():
            target_tokens = self.target_encoder(x)  # (B, N, D)
            # Layer normalize for stability (prevents collapse)
            target_tokens = torch.nn.functional.layer_norm(target_tokens, (target_tokens.size(-1),))
            # Extract target positions
            N_tgt = target_ids.shape[1]
            target_ids_exp = target_ids.unsqueeze(-1).expand(B, N_tgt, target_tokens.size(-1))
            target = torch.gather(target_tokens, dim=1, index=target_ids_exp)

        # Context encoder: encode only context patches
        context_tokens = self.context_encoder(x, ids_keep=context_ids)  # (B, N_ctx, D)

        # Predictor: predict target embeddings from context
        pred = self.predictor(context_tokens, context_ids, target_ids)

        return pred, target, context_ids, target_ids

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA update of target encoder.

        Parameters
        ----------
        momentum
            EMA momentum (typically 0.996-1.0).
        """
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=True,
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)

    def load_encoder_weights(
        self,
        state_dict: dict[str, Tensor],
        strict: bool = True,
    ) -> None:
        """Load pretrained encoder weights into both encoders.

        Parameters
        ----------
        state_dict
            State dict with encoder weights.
        strict
            Whether to require exact key matching.
        """
        self.context_encoder.load_state_dict(state_dict, strict=strict)
        self.target_encoder.load_state_dict(state_dict, strict=strict)
        self._freeze_target_encoder()

    def load_state_dict(
        self,
        state_dict: dict[str, Tensor],  # type: ignore[override]
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load state dict with support for Lightning checkpoint format.

        Automatically strips 'model.' prefix if present (Lightning checkpoints).

        Parameters
        ----------
        state_dict
            State dict to load.
        strict
            Whether to require exact key matching.
        assign
            Whether to assign tensors instead of copying.

        Returns
        -------
        _IncompatibleKeys
            Named tuple with missing and unexpected keys.
        """
        # Strip "model." prefix if present (Lightning checkpoint format)
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def jepa_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE loss between predicted and target embeddings.

    Parameters
    ----------
    pred
        Predicted embeddings: (B, N_tgt, D)
    target
        Target embeddings: (B, N_tgt, D)

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    return torch.nn.functional.mse_loss(pred, target)
