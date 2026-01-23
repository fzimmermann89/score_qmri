# Plan: JEPA (Joint Embedding Predictive Architecture) Implementation

**File:** `src/score_qmri/models/jepa.py`

## Decisions Made

| Question | Decision |
|----------|----------|
| Masking strategy | **Block masking** (contiguous rectangular regions) |
| EMA decay | **Cosine schedule** (0.996 → 1.0) |
| Predictor depth | **Configurable** via `JEPAConfig` |
| Loss function | **MSE** |
| Encoding strategy | **Full image** (simpler, mask after encoding for targets) |
| Weight loading | **Supported** via `load_encoder_weights()` |

## Key Observations from References

### nanojepa (simpler, our starting point)
- Uses **random masking** (not block) - we'll upgrade to block masking
- Predictor uses **scatter/gather** to place context tokens and mask tokens
- Forward returns `(pred_tokens, target_tokens, mask)` for all positions
- Loss computed on masked positions only
- EMA is fixed at 0.996 (we'll use cosine schedule)
- **Clean pattern**: `random_masking()` returns `(x_masked, mask, ids_keep)`

### I-JEPA (more complex, reference for block masking)
- Uses **MaskCollator** in dataloader (returns index tensors)
- Predictor uses **separate embed dim** (smaller than encoder)
- Predictor has fixed sinusoidal pos embeddings (not learnable)
- Target encoder encodes full image, then `apply_masks()` extracts target regions
- Context encoder receives mask indices, only encodes visible patches
- Uses `repeat_interleave_batch` for handling multiple context/target masks

### Key Architectural Differences

| Aspect | nanojepa | I-JEPA | Our Choice |
|--------|----------|--------|------------|
| Masking | Random | Block (MaskCollator) | Block (simplified) |
| Predictor dim | Same as encoder | Smaller (e.g., 384 vs 768) | Configurable |
| Pos embed in predictor | Learnable (shared w/ encoder) | Fixed sinusoidal | Learnable |
| Context encoding | Encode full, mask out | Encode only visible patches | Encode full (simpler) |
| Mask format | Binary mask (B, N) | Index lists | Binary mask (simpler) |

## Simplified Architecture (following nanojepa style, adding block masking)

We'll follow nanojepa's cleaner pattern but add:
1. Block masking instead of random
2. Cosine EMA schedule
3. MSE loss instead of smooth L1
4. Configurable predictor embed dim

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              JEPA Model                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  Input: x (B, 2, H, W)                                                        │
│                                                                               │
│  1. Generate block masks: context_mask, target_mask (B, N) booleans           │
│                                                                               │
│  2. Target encoder (frozen, EMA):                                             │
│     - Encode full image → (B, N, D)                                           │
│     - Layer normalize (collapse prevention)                                   │
│     - Extract target positions using target_mask                              │
│                                                                               │
│  3. Context encoder:                                                          │
│     - Encode full image → (B, N, D)                                           │
│     - Extract context positions (where ~target_mask, i.e., visible)           │
│                                                                               │
│  4. Predictor:                                                                │
│     - Project context tokens to predictor dim                                 │
│     - Build full sequence: scatter context + mask_tokens                      │
│     - Add positional embeddings                                               │
│     - Run transformer blocks                                                  │
│     - Extract target positions, project back to encoder dim                   │
│                                                                               │
│  5. Loss: MSE(pred_target, true_target)                                       │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. `JEPAConfig` (dataclass)

```python
@dataclass
class JEPAConfig:
    """JEPA hyperparameters."""
    # Image/patch
    img_size: int = 256
    patch_size: int = 16
    in_chans: int = 2
    
    # Encoder
    embed_dim: int = 384
    encoder_depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    
    # Predictor (can be smaller than encoder)
    predictor_embed_dim: int = 192
    predictor_depth: int = 6
    
    # Block masking
    target_aspect_ratio: tuple[float, float] = (0.75, 1.5)
    target_scale: tuple[float, float] = (0.15, 0.2)
    num_targets: int = 4
    
    # EMA
    ema_start: float = 0.996
    ema_end: float = 1.0
```

### 2. `sample_block_mask` (function)

Generate a single rectangular block mask:

```python
def sample_block_mask(
    grid_size: int,
    scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    device: torch.device,
) -> Tensor:
    """Sample a single rectangular block mask.
    
    Returns
    -------
    mask
        Boolean tensor (grid_size, grid_size) - True = in block
    """
```

### 3. `BlockMaskGenerator`

```python
class BlockMaskGenerator:
    """Generate target block masks.
    
    Context = complement of union of target blocks.
    """
    
    def __init__(
        self,
        grid_size: int,
        num_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
    ): ...
    
    def __call__(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Returns
        -------
        context_mask
            (B, N) - True = visible (context)
        target_mask
            (B, N) - True = predict (target)
        """
```

### 4. `Predictor`

Following nanojepa's simpler style but with optional dim reduction:

```python
class Predictor(torch.nn.Module):
    """Predict target embeddings from context.
    
    Projects encoder dim → predictor dim, runs transformer,
    projects back to encoder dim.
    """
    
    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        predictor_embed_dim: int | None = None,  # None = same as encoder
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ): ...
    
    def forward(
        self,
        context_tokens: Tensor,  # (B, N_ctx, D_enc)
        context_mask: Tensor,    # (B, N) bool - True = context position
        target_mask: Tensor,     # (B, N) bool - True = target position
        pos_embed: Tensor,       # (1, N, D_enc) from encoder
    ) -> Tensor:  # (B, N_tgt, D_enc)
        """Predict target embeddings."""
```

### 5. `JEPA`

```python
class JEPA(torch.nn.Module):
    """Joint Embedding Predictive Architecture.
    
    Learns representations by predicting target block embeddings
    from context block embeddings in latent space.
    """
    
    def __init__(self, config: JEPAConfig | None = None):
        self.config = config or JEPAConfig()
        
        # Encoders
        self.context_encoder = VisionTransformer(...)
        self.target_encoder = VisionTransformer(...)
        
        # Predictor
        self.predictor = Predictor(...)
        
        # Mask generator
        self.mask_generator = BlockMaskGenerator(...)
        
        # Initialize and freeze target encoder
        self._init_target_encoder()
    
    @property
    def num_patches(self) -> int:
        return self.context_encoder.num_patches
    
    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim
    
    def _init_target_encoder(self) -> None:
        """Copy context encoder weights to target encoder and freeze."""
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x
            Input images: (B, 2, H, W)
        
        Returns
        -------
        pred
            Predicted target embeddings: (B, N_tgt, D)
        target
            True target embeddings (normalized): (B, N_tgt, D)
        context_mask
            Context mask: (B, N)
        target_mask
            Target mask: (B, N)
        """
    
    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA update of target encoder."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.data)
    
    def load_encoder_weights(self, state_dict: dict[str, Tensor], strict: bool = True) -> None:
        """Load pretrained encoder weights into both encoders."""
        self.context_encoder.load_state_dict(state_dict, strict=strict)
        self.target_encoder.load_state_dict(state_dict, strict=strict)
        self._freeze_target_encoder()
    
    def _freeze_target_encoder(self) -> None:
        for param in self.target_encoder.parameters():
            param.requires_grad = False
```

### 6. `jepa_loss` and `cosine_ema_schedule`

```python
def jepa_loss(pred: Tensor, target: Tensor) -> Tensor:
    """MSE loss between predicted and target embeddings."""
    return torch.nn.functional.mse_loss(pred, target)


def cosine_ema_schedule(
    step: int,
    total_steps: int,
    start: float = 0.996,
    end: float = 1.0,
) -> float:
    """Cosine schedule for EMA momentum."""
    progress = step / max(total_steps, 1)
    return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2
```

## Code Structure

```python
"""JEPA model for self-supervised MRI pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from torch import Tensor

from .vit import VisionTransformer, Block, RMSNorm


@dataclass
class JEPAConfig:
    """JEPA hyperparameters."""
    ...


def cosine_ema_schedule(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    ...


def sample_block_mask(
    grid_size: int,
    scale: tuple[float, float],
    aspect_ratio: tuple[float, float],
    device: torch.device,
) -> Tensor:
    ...


class BlockMaskGenerator:
    ...


class Predictor(torch.nn.Module):
    ...


class JEPA(torch.nn.Module):
    ...


def jepa_loss(pred: Tensor, target: Tensor) -> Tensor:
    ...
```

## Style Alignment with Existing Code

Following `diffusion.py` and `vit.py` patterns:
- `@dataclass` for configs
- NumPy-style docstrings
- Type hints everywhere
- `torch.nn.Module` (not `nn.Module`)
- `einops` for tensor reshaping
- `RMSNorm` from `vit.py` (float32 stable)
- Shape comments: `# (B, N, D)`

## Test Plan

```python
def test_jepa_forward():
    config = JEPAConfig(img_size=64, patch_size=8, embed_dim=128, encoder_depth=4, predictor_depth=2)
    model = JEPA(config)
    x = torch.randn(2, 2, 64, 64)
    pred, target, ctx_mask, tgt_mask = model(x)
    
    assert pred.shape == target.shape
    assert pred.shape[0] == 2  # batch
    assert pred.shape[2] == 128  # embed_dim
    
def test_jepa_ema_update():
    model = JEPA()
    # Check target encoder frozen
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    
    # Check EMA update
    old_param = model.target_encoder.blocks[0].attn.qkv.weight.clone()
    model.update_target_encoder(momentum=0.5)
    new_param = model.target_encoder.blocks[0].attn.qkv.weight
    assert not torch.equal(old_param, new_param)

def test_block_mask_generator():
    gen = BlockMaskGenerator(grid_size=16, num_targets=4)
    ctx_mask, tgt_mask = gen(batch_size=8, device=torch.device('cpu'))
    
    assert ctx_mask.shape == (8, 256)
    assert tgt_mask.shape == (8, 256)
    # Context and target should not overlap
    assert not (ctx_mask & tgt_mask).any()
```

## Next Steps

1. ✅ Plan reviewed and approved
2. Implement `src/score_qmri/models/jepa.py`
3. Update `src/score_qmri/models/__init__.py`
4. Write tests
5. Verify with ruff/mypy
