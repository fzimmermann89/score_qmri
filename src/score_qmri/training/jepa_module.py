"""PyTorch Lightning module for JEPA pretraining."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor

from score_qmri.models.jepa import JEPA, JEPAConfig, cosine_ema_schedule, jepa_loss


@dataclass
class JEPAHparams:
    """Type-safe hyperparameter definitions for JEPALightningModule."""

    # Model architecture
    img_size: int
    patch_size: int
    in_chans: int
    embed_dim: int
    encoder_depth: int
    num_heads: int
    mlp_ratio: float
    predictor_embed_dim: int | None
    predictor_depth: int

    # Masking
    target_scale_min: float
    target_scale_max: float
    target_aspect_ratio_min: float
    target_aspect_ratio_max: float
    num_targets: int

    # Optimization
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    min_lr_ratio: float

    # EMA
    ema_start: float
    ema_end: float

    # Logging
    log_interval: int


class JEPALightningModule(LightningModule):
    """Lightning module for JEPA self-supervised pretraining.

    Uses raw data without normalization. The ViT learns to handle
    varying intensities directly.
    """

    hparams: JEPAHparams  # type: ignore[assignment]

    def __init__(
        self,
        # Model architecture
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 2,
        embed_dim: int = 384,
        encoder_depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        predictor_embed_dim: int | None = None,
        predictor_depth: int = 6,
        # Masking
        target_scale_min: float = 0.15,
        target_scale_max: float = 0.2,
        target_aspect_ratio_min: float = 0.75,
        target_aspect_ratio_max: float = 1.5,
        num_targets: int = 4,
        # Optimization
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        min_lr_ratio: float = 0.01,
        # EMA
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        # Logging
        log_interval: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        config = JEPAConfig(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            predictor_embed_dim=predictor_embed_dim,
            predictor_depth=predictor_depth,
            target_scale=(target_scale_min, target_scale_max),
            target_aspect_ratio=(target_aspect_ratio_min, target_aspect_ratio_max),
            num_targets=num_targets,
            ema_start=ema_start,
            ema_end=ema_end,
        )

        self.model = JEPA(config)

    def on_fit_start(self) -> None:
        if self.global_rank != 0:
            return

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        context_params = sum(p.numel() for p in self.model.context_encoder.parameters())
        predictor_params = sum(p.numel() for p in self.model.predictor.parameters())

        self.log_dict(
            {
                "model/total_params": float(total_params),
                "model/trainable_params": float(trainable_params),
                "model/context_encoder_params": float(context_params),
                "model/predictor_params": float(predictor_params),
            },
            rank_zero_only=True,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x = batch["image"]  # (B, 2, H, W)

        # Forward pass
        pred, target, context_ids, target_ids = self.model(x)

        # Compute loss
        loss = jepa_loss(pred, target)

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        if self.global_step % self.hparams.log_interval == 0:
            self._log_step_metrics(context_ids, target_ids)

        return loss

    def _log_step_metrics(self, context_ids: Tensor, target_ids: Tensor) -> None:
        """Log additional metrics periodically."""
        # Compute current EMA momentum
        momentum = cosine_ema_schedule(
            self.global_step,
            self.hparams.max_steps,
            self.hparams.ema_start,
            self.hparams.ema_end,
        )

        # Context and target sizes
        n_ctx = context_ids.shape[1]
        n_tgt = target_ids.shape[1]
        total_patches = self.model.num_patches

        self.log_dict(
            {
                "train/ema_momentum": momentum,
                "train/context_ratio": n_ctx / total_patches,
                "train/target_ratio": n_tgt / total_patches,
            },
            rank_zero_only=True,
        )

    def on_train_batch_end(
        self,
        outputs: Tensor | dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Update target encoder with EMA
        momentum = cosine_ema_schedule(
            self.global_step,
            self.hparams.max_steps,
            self.hparams.ema_start,
            self.hparams.ema_end,
        )
        self.model.update_target_encoder(momentum)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x = batch["image"]

        # Forward pass (no mask randomness during validation for consistency)
        pred, target, _, _ = self.model(x)

        # Compute loss
        loss = jepa_loss(pred, target)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        # Separate parameters: context encoder + predictor (no target encoder)
        params = [
            {"params": self.model.context_encoder.parameters()},
            {"params": self.model.predictor.parameters()},
        ]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
        )

        def lr_lambda(step: int) -> float:
            """Warmup + cosine decay schedule."""
            warmup = self.hparams.warmup_steps
            max_steps = self.hparams.max_steps
            min_ratio = self.hparams.min_lr_ratio

            if step < warmup:
                # Linear warmup
                return step / max(1, warmup)

            # Cosine decay
            progress = (step - warmup) / max(1, max_steps - warmup)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_ratio + (1 - min_ratio) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient norms periodically."""
        if self.global_step % self.hparams.log_interval != 0:
            return

        encoder_grads = []
        predictor_grads = []

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().flatten()
            if "context_encoder" in name:
                encoder_grads.append(grad)
            elif "predictor" in name:
                predictor_grads.append(grad)

        def norm(grads: list[Tensor]) -> Tensor:
            return torch.cat(grads).norm() if grads else torch.tensor(0.0, device=self.device)

        self.log_dict(
            {
                "gradients/encoder_norm": norm(encoder_grads),
                "gradients/predictor_norm": norm(predictor_grads),
                "gradients/total_norm": norm(encoder_grads + predictor_grads),
            },
            rank_zero_only=True,
        )
