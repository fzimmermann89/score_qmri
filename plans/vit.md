# Plan: Vision Transformer (ViT) Implementation

**File:** `src/score_qmri/models/vit.py`

## Design Goals
- **Simplicity:** Clean implementation following `references/nanojepa` but adapted for this codebase.
- **Stability:** Normalization layers must compute in `float32` (similar to `GroupNorm32` in `unet.py`).
- **Style:** Follow project conventions (no `import torch.nn as nn`, explicit types, `einops`, NumPy-style docstrings).
- **Performance:** Use `torch.nn.functional.scaled_dot_product_attention` (SDPA).

## Architecture Components

### 1. `RMSNorm32`
A custom normalization layer that ensures stability by computing statistics in `float32`.
- **Inherits:** `torch.nn.Module`
- **Logic:**
  1. Cast input to `float32`.
  2. Compute RMS norm: `x * rsqrt(mean(x**2) + eps)`.
  3. Apply learnable scale (`weight`).
  4. Cast back to original dtype.
- **Why:** Replaces `GroupNorm32` from UNet for Transformer architectures, matching `nanojepa`'s choice but with the requested stability fix.

### 2. `PatchEmbed`
Standard image-to-patch embedding.
- **Implementation:** `torch.nn.Conv2d` with kernel/stride = patch_size.
- **Input:** `(B, C, H, W)` where `C=2` (default for MRI).
- **Output:** `(B, N, D)` sequence.

### 3. `Attention` (SDPA)
Multi-head self-attention block.
- **Components:** `qkv` projection (Linear), `proj` (Linear).
- **Mechanism:** `torch.nn.functional.scaled_dot_product_attention`.
- **Note:** No explicit mask needed for standard ViT encoding.

### 4. `MLP`
Feed-forward network.
- **Components:** Linear -> GELU -> Dropout -> Linear -> Dropout.
- **Ratio:** Default expansion ratio of 4.0.

### 5. `Block`
Standard Transformer block with Pre-Norm configuration.
- **Flow:**
  ```python
  x = x + self.attn(self.norm1(x))
  x = x + self.mlp(self.norm2(x))
  ```
- **Norm:** Uses `RMSNorm32`.

### 6. `VisionTransformer`
The main model class.
- **Init Args:** `img_size`, `patch_size`, `in_chans` (default 2), `embed_dim`, `depth`, `num_heads`, etc.
- **Structure:**
  - `patch_embed`
  - `pos_embed` (learnable parameter)
  - `pos_drop`
  - `blocks` (`torch.nn.ModuleList`)
  - `norm` (final `RMSNorm32`)
- **Initialization:** Use `torch.nn.init.trunc_normal_` for weights.
- **Forward:** Returns `(B, N, D)` features.

## Proposed Code Structure

```python
import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

class RMSNorm32(torch.nn.Module):
    """Root Mean Square Layer Normalization in float32."""
    ...

class PatchEmbed(torch.nn.Module):
    """2D Image to Patch Embedding."""
    ...

class Attention(torch.nn.Module):
    """Multi-head Attention using SDPA."""
    ...

class MLP(torch.nn.Module):
    """Feed-forward network."""
    ...

class Block(torch.nn.Module):
    """Transformer Block with Pre-Norm."""
    ...

class VisionTransformer(torch.nn.Module):
    """Vision Transformer."""
    def __init__(...):
        ...

    def forward(self, x: Tensor) -> Tensor:
        ...
```

## Next Steps
1. Create `src/score_qmri/models/vit.py` implementing this plan.
2. Verify with a simple instantiation test script.
