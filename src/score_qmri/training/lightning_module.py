"""PyTorch Lightning module for diffusion training."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind, is_complex_kind
from score_qmri.models.diffusion import (
    DiffusionModel,
    NoiseSchedule,
    _apply_kind_constraints,
    _create_noise,
)
from score_qmri.sampling.ddim_sampler import DDIMSampler, SamplingStats, create_fixed_noise
from score_qmri.training.ema import EMA
from score_qmri.training.losses import create_channel_mask, l1_loss, noise_loss, ssim_loss

VALIDATION_KINDS: tuple[DatasetKind, ...] = (
    DatasetKind.T1W_MAGNITUDE,
    DatasetKind.T1MAP,
    DatasetKind.COMPLEX_T1W_COIL_COMBINED,
    DatasetKind.T2MAP,
    DatasetKind.M0_MAGNITUDE,
)

LOSS_ANOMALY_THRESHOLD = 100.0


@dataclass
class DiffusionHparams:
    """Type-safe hyperparameter definitions for DiffusionLightningModule."""

    base_channels: int
    num_res_blocks: int
    channel_mult: tuple[int, ...]
    cond_dim: int
    beta_min: float
    beta_max: float
    learning_rate: float
    weight_decay: float
    noise_weight: float
    x0_weight: float
    ssim_weight: float
    l1_weight: float
    mask_weight: float
    validation_size: int
    validation_steps: int
    validation_interval: int
    log_batch_interval: int
    warmup_steps: int
    ema_decay: float
    attention_levels: tuple[int, ...]
    head_dim: int
    noise_schedule: NoiseSchedule


class DiffusionLightningModule(LightningModule):
    """Lightning module for diffusion training with dual prediction heads."""

    hparams: DiffusionHparams  # type: ignore[assignment]

    def __init__(
        self,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8, 16),
        cond_dim: int = 128,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        noise_weight: float = 1.0,
        x0_weight: float = 0.1,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        mask_weight: float = 0.1,
        validation_size: int = 128,
        validation_steps: int = 250,
        validation_interval: int = 1000,
        log_batch_interval: int = 200,
        warmup_steps: int = 2000,
        ema_decay: float = 0.9999,
        attention_levels: tuple[int, ...] = (),
        head_dim: int = 64,
        noise_schedule: NoiseSchedule = "linear",
    ) -> None:
        """
        Parameters
        ----------
        base_channels
            UNet base channel count.
        num_res_blocks
            Number of ResBlocks per level.
        channel_mult
            Channel multipliers per level (e.g., (1, 2, 4, 8) gives 4 levels).
        cond_dim
            Conditioning embedding dimension.
        beta_min
            Minimum noise rate at t=0.
        beta_max
            Maximum noise rate at t=1.
        learning_rate
            Optimizer learning rate.
        weight_decay
            AdamW weight decay.
        noise_weight
            Weight for noise prediction loss.
        x0_weight
            Weight for clean image prediction loss.
        ssim_weight
            SSIM weight within x0 loss.
        l1_weight
            L1 weight within x0 loss.
        mask_weight
            Weight for masked-out regions in loss (0 = ignore, 1 = full weight).
        validation_size
            Image size for validation sampling.
        validation_steps
            DDIM sampling steps for validation.
        validation_interval
            Steps between validation samples.
        log_batch_interval
            Steps between training batch visualizations.
        warmup_steps
            Number of warmup steps for linear LR warmup.
        ema_decay
            EMA decay rate for weight averaging.
        attention_levels
            Tuple of level indices where self-attention is applied.
        head_dim
            Dimension per attention head.
        noise_schedule
            Noise schedule type ("linear" or "cosine").
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffusionModel(
            beta_min=beta_min,
            beta_max=beta_max,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            cond_dim=cond_dim,
            num_kinds=NUM_DATASET_KINDS,
            attention_levels=attention_levels,
            head_dim=head_dim,
            noise_schedule=noise_schedule,
        )
        self.sampler: DDIMSampler = DDIMSampler(self.model, num_steps=validation_steps)
        self._fixed_noise: dict[int, Tensor] | None = None
        self.ema: EMA[DiffusionModel] | None = None

    def on_fit_start(self) -> None:
        self.ema = EMA[DiffusionModel](self.model, decay=self.hparams.ema_decay)

        if not isinstance(self.logger, WandbLogger) or self.global_rank != 0:
            return

        import wandb

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        beta1, beta2 = 0.9, 0.999

        hparams = {
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "optimizer/weight_decay": self.hparams.weight_decay,
            "optimizer/beta1": beta1,
            "optimizer/beta2": beta2,
        }

        self.logger.log_hyperparams(hparams)

        if wandb.run is not None:
            project_root = Path(__file__).parent.parent
            wandb.run.log_code(
                str(project_root),
                include_fn=lambda p: p.endswith(".py") or p.endswith(".yaml") or p.endswith(".toml"),
            )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.global_step % 100 != 0:
            return

        x0_grads, noise_grads, backbone_grads = [], [], []

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().flatten()
            if "x0_head" in name:
                x0_grads.append(grad)
            elif "noise_head" in name:
                noise_grads.append(grad)
            else:
                backbone_grads.append(grad)

        def norm(grads: list[Tensor]) -> Tensor:
            return torch.cat(grads).norm() if grads else torch.tensor(0.0, device=self.device)

        self.log_dict(
            {
                "gradients/total_norm": norm(x0_grads + noise_grads + backbone_grads),
                "gradients/x0_head_norm": norm(x0_grads),
                "gradients/noise_head_norm": norm(noise_grads),
                "gradients/backbone_norm": norm(backbone_grads),
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x0_raw = batch["image"]
        mask = batch["mask"]
        kind = batch["kind"]

        x0_raw = _apply_kind_constraints(x0_raw, kind)

        x0_pre_clamp = self.model.normalize(x0_raw, kind, mask)
        x0 = x0_pre_clamp.clamp(-6, 6)

        eps = 1e-5
        t = torch.rand(x0.shape[0], device=self.device) * (1.0 - eps) + eps

        noise = _create_noise(x0.shape, kind, self.device, x0.dtype)
        x_t = self.model.q_sample(x0, t, kind, noise)

        output = self.model(x_t, t, kind)

        valid_counts = mask.sum(dim=(-2, -1))
        valid_samples = valid_counts >= 1000
        n_valid = int(valid_samples.sum().item())

        if n_valid == 0:
            print(f"[Rank {self.global_rank}] Skipping step {self.global_step}: no valid samples")
            return (output.x0.sum() + output.noise.sum()).nan_to_num() * 0.0

        x0_v = x0[valid_samples]
        noise_v = noise[valid_samples]
        mask_v = mask[valid_samples]
        kind_v = kind[valid_samples]
        pred_noise_v = output.noise[valid_samples]
        pred_x0_v = output.x0[valid_samples]

        weighted_mask = torch.where(mask_v.unsqueeze(1), 1.0, self.hparams.mask_weight)
        channel_mask = create_channel_mask(kind_v, self.device)

        t_v = t[valid_samples]
        signal_weight = self.model.alpha_bar(t_v)

        loss_noise = noise_loss(pred_noise_v, noise_v, weighted_mask, channel_mask)
        loss_l1 = l1_loss(pred_x0_v, x0_v, weighted_mask, channel_mask, reduction="none")
        loss_ssim = ssim_loss(pred_x0_v, x0_v, weighted_mask, channel_mask, reduction="none").squeeze(-1)

        loss_l1 = (loss_l1 * signal_weight).mean()
        loss_ssim = (loss_ssim * signal_weight).mean()

        x0_loss = self.hparams.ssim_weight * loss_ssim + self.hparams.l1_weight * loss_l1
        total_loss = self.hparams.noise_weight * loss_noise + self.hparams.x0_weight * x0_loss

        if not torch.isfinite(total_loss) or total_loss > LOSS_ANOMALY_THRESHOLD:
            loss_val = total_loss.item() if torch.isfinite(total_loss) else float("nan")
            print(f"[Rank {self.global_rank}] Skipping step {self.global_step}: loss={loss_val:.2f}")
            return (output.x0.sum() + output.noise.sum()).nan_to_num() * 0.0

        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log_dict(
            {
                "train/noise_mse": loss_noise,
                "train/x0_l1": loss_l1,
                "train/x0_ssim": 1.0 - loss_ssim,
            },
            prog_bar=True,
            rank_zero_only=True,
        )

        if self.global_step > 0 and self.global_step % self.hparams.log_batch_interval == 0:
            x_t_v = x_t[valid_samples]
            x0_tweedie_v = self.model.predict_x0(x_t_v, t_v, pred_noise_v, kind_v)
            self._log_training_batch(x0_v, x_t_v, noise_v, pred_noise_v, x0_tweedie_v, pred_x0_v, kind_v, t_v)

        return total_loss

    def _log_training_batch(
        self,
        x0_gt: Tensor,
        x_t: Tensor,
        noise_gt: Tensor,
        noise_pred: Tensor,
        x0_tweedie: Tensor,
        x0_pred: Tensor,
        kind: Tensor,
        t: Tensor,
    ) -> None:
        if not isinstance(self.logger, WandbLogger) or self.global_rank != 0:
            return

        import matplotlib.pyplot as plt
        import wandb

        B = min(x0_gt.shape[0], 8)
        fig, axes = plt.subplots(6, B, figsize=(2.5 * B, 15), squeeze=False)

        def to_np(x: Tensor, kind_tensor: Tensor) -> list:
            result = []
            for i in range(B):
                sample = x[i]
                k_idx = int(kind_tensor[i].item())
                mag = (sample[0] ** 2 + sample[1] ** 2).sqrt() if is_complex_kind(k_idx) else sample[0]
                result.append(mag.detach().cpu().numpy())
            return result

        gt = to_np(x0_gt, kind)
        noised = to_np(x_t, kind)
        n_gt = to_np(noise_gt, kind)
        n_pred = to_np(noise_pred, kind)
        x0_tw = to_np(x0_tweedie, kind)
        x0_p = to_np(x0_pred, kind)
        kind_names = [DatasetKind(int(kind[i].item())).name for i in range(B)]

        row_labels = ["x0 GT", "x_t (noised)", "noise GT", "noise pred", "x0 Tweedie", "x0 pred"]
        for i in range(B):
            for row, data in enumerate([gt, noised, n_gt, n_pred, x0_tw, x0_p]):
                axes[row, i].imshow(data[i], cmap="gray")
                axes[row, i].axis("off")
                if i == 0:
                    axes[row, i].text(
                        -0.05,
                        0.5,
                        row_labels[row],
                        fontsize=8,
                        transform=axes[row, i].transAxes,
                        ha="right",
                        va="center",
                    )
            axes[0, i].set_title(f"{kind_names[i]}\nt={t[i].item():.2f}", fontsize=7)

        fig.suptitle(f"Training batch @ step {self.global_step}", fontsize=10)
        fig.tight_layout()

        self.logger.experiment.log({"train/batch_visualization": wandb.Image(fig)})
        plt.close(fig)

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema is not None:
            self.ema.update(self.model)

        if self.global_step > 0 and self.global_step % 100 == 0:
            self.model.normalizer.sync_stats()

        if self.global_step > 0 and self.global_step % self.hparams.validation_interval == 0:
            self._log_samples()

    def _log_samples(self) -> None:
        if not isinstance(self.logger, WandbLogger) or self.global_rank != 0:
            return

        import matplotlib.pyplot as plt
        import wandb

        size = self.hparams.validation_size
        num_samples = 4

        if self._fixed_noise is None:
            self._fixed_noise = create_fixed_noise((size, size), num_samples=num_samples)

        sample_model: DiffusionModel = self.ema.shadow if self.ema is not None else self.model
        sample_model.eval()
        sampler = DDIMSampler(sample_model, num_steps=self.hparams.validation_steps)

        with torch.no_grad():
            for kind_enum in VALIDATION_KINDS:
                k = int(kind_enum)
                kind_tensor = torch.full((num_samples,), k, device=self.device, dtype=torch.long)
                collect = kind_enum == DatasetKind.T1W_MAGNITUDE

                samples = sampler.sample(
                    (num_samples, 2, size, size),
                    kind_tensor,
                    self._fixed_noise[k],
                    collect_stats=collect,
                )
                samples_denorm: Tensor = sample_model.denormalize(samples, kind_tensor)

                fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
                if num_samples == 1:
                    axes = [axes]
                for i in range(num_samples):
                    sample_i = samples_denorm[i]
                    if is_complex_kind(k):
                        img = (sample_i[0] ** 2 + sample_i[1] ** 2).sqrt().cpu().numpy()
                    else:
                        img = sample_i[0].cpu().numpy()
                    axes[i].imshow(img, cmap="gray")
                    axes[i].axis("off")
                fig.suptitle(f"{kind_enum.name} @ step {self.global_step}", fontsize=10)
                fig.tight_layout()

                self.logger.experiment.log({f"samples/{kind_enum.name}": wandb.Image(fig), "global_step": self.global_step})
                plt.close(fig)

                if collect and sampler.last_stats is not None:
                    self._log_sampling_stats(kind_enum.name, sampler.last_stats)

        if self.ema is None:
            sample_model.train()

    def _log_sampling_stats(self, kind_name: str, stats: SamplingStats) -> None:
        if not isinstance(self.logger, WandbLogger) or self.global_rank != 0:
            return

        import matplotlib.pyplot as plt
        import wandb

        n = len(stats.intermediates)
        if n == 0:
            return

        fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5), squeeze=False)

        for j, (inter, step, t_val) in enumerate(zip(stats.intermediates, stats.steps, stats.t_values, strict=True)):
            img = (inter[0, 0] ** 2 + inter[0, 1] ** 2).sqrt().numpy() if inter.shape[1] == 2 else inter[0, 0].numpy()
            axes[0, j].imshow(img, cmap="gray")
            axes[0, j].axis("off")
            axes[0, j].set_title(f"step {step}\nt={t_val:.3f}", fontsize=7)

        ax_stats = axes[1, 0]
        for ax in axes[1, 1:]:
            ax.axis("off")

        ax_stats.plot(stats.steps, stats.x0_std, "b-", label="x0 std", linewidth=1.5)
        ax_stats.plot(stats.steps, stats.x_std, "g--", label="x std", linewidth=1.5)
        ax_stats.plot(stats.steps, stats.noise_std, "r:", label="noise std", linewidth=1.5)
        ax_stats.set_xlabel("step")
        ax_stats.set_ylabel("std")
        ax_stats.legend(fontsize=6)
        ax_stats.set_title("Sampling statistics", fontsize=8)

        fig.suptitle(f"Sampling progression: {kind_name} @ step {self.global_step}", fontsize=10)
        fig.tight_layout()

        self.logger.experiment.log({"samples/progression": wandb.Image(fig), "global_step": self.global_step})
        plt.close(fig)

        table_data = [
            [step, f"{t:.4f}", f"{ab:.6f}", f"{x0_mean:.3f}", f"{x0_std:.3f}", f"{n_std:.3f}"]
            for step, t, ab, x0_mean, x0_std, n_std in zip(
                stats.steps, stats.t_values, stats.alpha_bar, stats.x0_mean, stats.x0_std, stats.noise_std, strict=True
            )
        ]
        table = wandb.Table(
            columns=["step", "t", "alpha_bar", "x0_mean", "x0_std", "noise_std"],
            data=table_data,
        )
        self.logger.experiment.log({"samples/stats_table": table, "global_step": self.global_step})

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        def warmup_lambda(step: int) -> float:
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
