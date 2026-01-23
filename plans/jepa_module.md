# Plan: JEPA Lightning Module

**File:** `src/score_qmri/training/jepa_module.py`

## Goal
Implement a PyTorch Lightning module for training the JEPA model, matching the style and robustness of `DiffusionLightningModule`.

## Requirements
1. **Configurable**: Use a `JEPAHparams` dataclass.
2. **Optimization**: AdamW with warmup + cosine schedule (for LR) and cosine schedule (for EMA).
3. **Training Loop**:
   - Forward pass (pred, target, masks)
   - Loss computation (MSE)
   - EMA update of target encoder
   - Logging (loss, LR, EMA decay)
4. **Validation/Viz**:
   - Visualize context vs. target predictions?
   - Or just linear probe accuracy (if we implement a probe)?
   - For now: **Reconstruction visualization** (visualize predicted target blocks placed back into the image).
5. **Checkpointing**: Save/load EMA state (target encoder is already in `model.target_encoder`, but we might want to save the `JEPA` object as a whole). Actually, `JEPA` model class stores both encoders. The `update_target_encoder` method updates it. So standard checkpointing works. `DiffusionLightningModule` uses a separate `EMA` class because `DiffusionModel` doesn't inherently have an EMA copy. `JEPA` *does* have an inherent EMA copy. So we don't need the `EMA` wrapper class!

## Architecture

### 1. `JEPAHparams`
```python
@dataclass
class JEPAHparams:
    # Model config
    img_size: int
    patch_size: int
    embed_dim: int
    # ... other model params ...
    
    # Optimization
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int  # Needed for cosine schedules
    
    # EMA
    ema_start: float
    ema_end: float
```

### 2. `JEPALightningModule`

**`__init__`**:
- Initialize `JEPA` model from hparams.
- Save hparams.

**`configure_optimizers`**:
- AdamW.
- LR Scheduler: Warmup + Cosine Decay.
- **Note**: The EMA schedule is *not* an optimizer scheduler. It's updated manually in `training_step` or `on_train_batch_end`.

**`training_step`**:
- Get batch `x`.
- `pred, target, ctx_ids, tgt_ids = self.model(x)`
- `loss = jepa_loss(pred, target)`
- Log loss.
- **EMA Update**:
  - Calculate current momentum based on global step.
  - `self.model.update_target_encoder(momentum)`
  - Log momentum.

**`validation_step`**:
- Compute validation loss.
- Log.

**`on_train_batch_end`**:
- Maybe visualization?

**Visualization Logic**:
- Reconstruct the "predicted image":
  - Start with context image (masked).
  - Fill in target patches with `pred` tokens (projected back to pixels? No, `pred` are embeddings).
  - JEPA predicts *embeddings*, not pixels. We cannot easily visualize "predicted images" without a separate decoder.
  - **Alternative**: Train a small decoder on top? Or just visualize the masks.
  - **Decision**: Just visualize the masks (Context vs Target) for sanity check.

## Comparison with `DiffusionLightningModule`

| Feature | Diffusion | JEPA |
|---------|-----------|------|
| Model | `DiffusionModel` (wraps UNet) | `JEPA` (wraps ViTs) |
| EMA | External `EMA` class | Internal `target_encoder` |
| Loss | Weighted MSE/SSIM | Simple MSE |
| Scheduler | Constant/Lambda | Cosine with Warmup |
| Sampler | EDM Sampler | N/A (predicts embeddings) |

## Implementation Plan

1. Create `src/score_qmri/training/jepa_module.py`.
2. Implement `JEPAHparams`.
3. Implement `JEPALightningModule`.
   - `training_step`: Handle 2ch/complex data inputs correctly (magnitude check?).
   - `configure_optimizers`: Implement cosine LR schedule.
   - `update_ema`: Implement cosine EMA schedule using `cosine_ema_schedule` from `jepa.py`.

## Questions/Refinements

- **Input Format**: `batch` from dataloader is a dict `{"image": ..., "kind": ...}`.
- JEPA handles `(B, 2, H, W)`. We need to handle `kind`?
- **Normalization**: `DiffusionModule` normalizes/clamps. JEPA should probably receive normalized data too.
- **Action**: Apply same normalization logic as diffusion?
  - `x0_raw = batch["image"]`
  - `x0_norm = self.normalize(x0_raw, kind)`
  - **Wait**, JEPA usually learns from *raw* or standard-scaled data. The `DiffusionModel` has specific `normalize/denormalize` methods.
  - `JEPA` model doesn't have these.
  - **Decision**: Add normalization logic to `JEPALightningModule` or just use raw data?
  - Diffusion uses `kind_scales`. This is important for MRI where intensities vary wildly.
  - **Decision**: Copy the normalization logic (kind scales) to `JEPALightningModule` to ensure the ViT sees reasonable values (~0-1 range).

## Code Skeleton

```python
class JEPALightningModule(LightningModule):
    def __init__(self, ...):
        # ...
        self.model = JEPA(config)
        # Load kind scales for normalization
        
    def training_step(self, batch, batch_idx):
        img = batch["image"]
        kind = batch["kind"]
        
        # Normalize
        img = self._normalize(img, kind)
        
        # Forward
        pred, target, _, _ = self.model(img)
        loss = jepa_loss(pred, target)
        
        # EMA
        m = cosine_ema_schedule(self.global_step, self.hparams.max_steps, ...)
        self.model.update_target_encoder(m)
        
        return loss
```

This looks solid. I will create `src/score_qmri/training/jepa_module.py`.
