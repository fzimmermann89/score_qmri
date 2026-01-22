"""PyTorch Lightning module for VE/EDM diffusion training."""

from __future__ import annotations

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
    EDMConfig,
    NormalizationConfig,
    _apply_kind_constraints,
    _create_noise,
)
from score_qmri.sampling.edm_sampler import EDMSampler, SamplingStats, create_fixed_noise
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
    learning_rate: float
    weight_decay: float

    main_weight: float
    aux_weight: float
    ssim_weight: float
    l1_weight: float

    penumbra_weight: float
    background_weight: float

    validation_size: int
    validation_steps: int
    validation_interval: int
    log_batch_interval: int
    warmup_steps: int
    ema_decay: float

    cond_dropout_rate: float

    attention_levels: tuple[int, ...]
    head_dim: int

    sigma_min: float
    sigma_max: float
    sigma_data: float
    p_mean: float
    p_std: float
    rho: float

    normalization: NormalizationConfig


class DiffusionLightningModule(LightningModule):
    """Lightning module for VE/EDM diffusion training with two x0 heads."""

    hparams: DiffusionHparams  # type: ignore[assignment]

    def __init__(
        self,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8, 16),
        cond_dim: int = 128,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        main_weight: float = 1.0,
        aux_weight: float = 0.1,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        penumbra_weight: float = 0.0,
        background_weight: float = 0.5,
        validation_size: int = 128,
        validation_steps: int = 50,
        validation_interval: int = 1000,
        log_batch_interval: int = 200,
        warmup_steps: int = 2000,
        ema_decay: float = 0.9999,
        cond_dropout_rate: float = 0.1,
        attention_levels: tuple[int, ...] = (),
        head_dim: int = 64,
        sigma_min: float = 0.002,
        sigma_max: float = 10.0,
        sigma_data: float = 0.2,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        rho: float = 7.0,
        normalization: NormalizationConfig | None = None,
    ) -> None:
        super().__init__()
        normalization = normalization or NormalizationConfig()
        self.save_hyperparameters()

        edm = EDMConfig(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            p_mean=p_mean,
            p_std=p_std,
        )

        self.model = DiffusionModel(
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            cond_dim=cond_dim,
            num_kinds=NUM_DATASET_KINDS,
            attention_levels=attention_levels,
            head_dim=head_dim,
            edm=edm,
            normalization=normalization,
        )

        self.sampler: EDMSampler = EDMSampler(self.model, num_steps=validation_steps, rho=rho)
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

        self.logger.log_hyperparams(
            {
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "optimizer/weight_decay": self.hparams.weight_decay,
                "optimizer/beta1": beta1,
                "optimizer/beta2": beta2,
            }
        )

        if wandb.run is not None:
            project_root = Path(__file__).parent.parent
            wandb.run.log_code(
                str(project_root),
                include_fn=lambda p: p.endswith(".py") or p.endswith(".yaml") or p.endswith(".toml"),
            )

    def on_after_batch_transfer(self, batch: dict[str, Tensor], dataloader_idx: int) -> dict[str, Tensor]:
        for key, value in batch.items():
            if isinstance(value, Tensor) and value.ndim == 4:
                batch[key] = value.to(memory_format=torch.channels_last)
        return batch

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.global_step % 100 != 0:
            return

        main_grads, aux_grads, backbone_grads = [], [], []

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach().flatten()
            if "main_head" in name:
                main_grads.append(grad)
            elif "aux_head" in name:
                aux_grads.append(grad)
            else:
                backbone_grads.append(grad)

        def norm(grads: list[Tensor]) -> Tensor:
            return torch.cat(grads).norm() if grads else torch.tensor(0.0, device=self.device)

        self.log_dict(
            {
                "gradients/total_norm": norm(main_grads + aux_grads + backbone_grads),
                "gradients/main_head_norm": norm(main_grads),
                "gradients/aux_head_norm": norm(aux_grads),
                "gradients/backbone_norm": norm(backbone_grads),
            },
            on_step=True,
            on_epoch=False,
            rank_zero_only=True,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x0_raw = batch["image"]
        penumbra_mask = batch["penumbra_mask"]
        background_mask = batch["background_mask"]
        kind = batch["kind"]

        signal_mask = ~background_mask

        x0_raw = _apply_kind_constraints(x0_raw, kind)

        x0_norm = self.model.normalize(x0_raw, kind)
        x0_norm = torch.where(signal_mask.unsqueeze(1), x0_norm, torch.zeros_like(x0_norm))
        x0 = self.model.clamp_x0(x0_norm, kind, limit=self.model.train_clip)

        # Sample sigma per example (continuous)
        sigma = self.model.sample_sigmas(x0.shape[0], device=self.device)

        noise = _create_noise(x0.shape, kind, self.device, x0.dtype)
        x_sigma = x0 + sigma.view(-1, 1, 1, 1) * noise

        # Classifier-free guidance dropout (drop only kind embedding)
        cond_mask = None
        if self.hparams.cond_dropout_rate > 0.0:
            keep_prob = 1.0 - self.hparams.cond_dropout_rate
            cond_mask = torch.bernoulli(torch.full((x0.shape[0],), keep_prob, device=self.device)).unsqueeze(1)

        output = self.model(x_sigma, sigma, kind, cond_mask=cond_mask)

        valid_counts = signal_mask.sum(dim=(-2, -1))
        valid_samples = valid_counts >= 1000
        n_valid = int(valid_samples.sum().item())
        if n_valid == 0:
            print(f"[Rank {self.global_rank}] Skipping step {self.global_step}: no valid samples")
            return torch.nan_to_num(output.x0_main.sum() + output.x0_aux.sum()) * 0.0

        x0_v = x0[valid_samples]
        x_sigma_v = x_sigma[valid_samples]
        sigma_v = sigma[valid_samples]
        penumbra_mask_v = penumbra_mask[valid_samples]
        background_mask_v = background_mask[valid_samples]
        kind_v = kind[valid_samples]
        x0_main_v = output.x0_main[valid_samples]
        x0_aux_v = output.x0_aux[valid_samples]

        loss_weight = torch.ones_like(background_mask_v, dtype=x0_v.dtype)
        loss_weight = torch.where(background_mask_v, self.hparams.background_weight, loss_weight)
        loss_weight = torch.where(penumbra_mask_v, self.hparams.penumbra_weight, loss_weight)
        loss_weight = loss_weight.unsqueeze(1)

        channel_mask = create_channel_mask(kind_v)

        # EDM main loss (weighted L2 on x0_main)
        sigma_data = torch.tensor(self.model.sigma_data, device=self.device, dtype=torch.float32)
        sigma_f = sigma_v.float()
        w = (sigma_f.square() + sigma_data.square()) / (sigma_f * sigma_data).square().clamp(min=1e-8)

        main_mse = noise_loss(x0_main_v, x0_v, loss_weight, channel_mask, reduction="none")
        loss_main = (main_mse * w).mean()

        # Auxiliary SSIM+L1 (weighted towards low sigma)
        sigma_floor = float(self.model.sigma_min)
        aux_w = 1.0 / (sigma_f.square() + sigma_floor**2)

        loss_l1 = l1_loss(x0_aux_v, x0_v, loss_weight, channel_mask, reduction="none")
        loss_ssim = ssim_loss(x0_aux_v, x0_v, loss_weight, channel_mask, reduction="none")
        aux_loss = self.hparams.ssim_weight * loss_ssim + self.hparams.l1_weight * loss_l1
        loss_aux = (aux_loss * aux_w).mean()

        total_loss = self.hparams.main_weight * loss_main + self.hparams.aux_weight * loss_aux

        if not torch.isfinite(total_loss) or total_loss > LOSS_ANOMALY_THRESHOLD:
            loss_val = total_loss.item() if torch.isfinite(total_loss) else float("nan")
            print(f"[Rank {self.global_rank}] Skipping step {self.global_step}: loss={loss_val:.2f}")
            return torch.nan_to_num(output.x0_main.sum() + output.x0_aux.sum()) * 0.0

        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log_dict(
            {
                "train/main_l2": loss_main,
                "train/aux_l1": loss_l1.mean(),
                "train/aux_ssim": 1.0 - loss_ssim.mean(),
                "train/sigma_mean": sigma_f.mean(),
                "train/sigma_p95": torch.quantile(sigma_f, 0.95),
            },
            prog_bar=True,
            rank_zero_only=True,
        )

        if self.global_step > 0 and self.global_step % self.hparams.log_batch_interval == 0:
            self._log_training_batch(x0_v, x_sigma_v, x0_main_v, x0_aux_v, kind_v, sigma_v)

        return total_loss

    def _log_training_batch(
        self,
        x0_gt: Tensor,
        x_sigma: Tensor,
        x0_main: Tensor,
        x0_aux: Tensor,
        kind: Tensor,
        sigma: Tensor,
    ) -> None:
        if not isinstance(self.logger, WandbLogger) or self.global_rank != 0:
            return

        import matplotlib.pyplot as plt
        import wandb

        B = min(x0_gt.shape[0], 8)
        fig, axes = plt.subplots(4, B, figsize=(2.5 * B, 10), squeeze=False)

        def to_np(x: Tensor, kind_tensor: Tensor) -> list:
            result = []
            for i in range(B):
                sample = x[i]
                k_idx = int(kind_tensor[i].item())
                img = (sample[0] ** 2 + sample[1] ** 2).sqrt() if is_complex_kind(k_idx) else sample[0]
                result.append(img.detach().cpu().numpy())
            return result

        rows = [to_np(x0_gt, kind), to_np(x_sigma, kind), to_np(x0_main, kind), to_np(x0_aux, kind)]
        row_labels = ["x0 GT", "x_sigma", "x0 main", "x0 aux"]

        kind_names = [DatasetKind(int(kind[i].item())).name for i in range(B)]
        for i in range(B):
            for r, data in enumerate(rows):
                axes[r, i].imshow(data[i], cmap="gray")
                axes[r, i].axis("off")
                if i == 0:
                    axes[r, i].text(-0.05, 0.5, row_labels[r], fontsize=8, transform=axes[r, i].transAxes, ha="right", va="center")
            axes[0, i].set_title(f"{kind_names[i]}\nσ={sigma[i].item():.3f}", fontsize=7)

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
        sampler = EDMSampler(sample_model, num_steps=self.hparams.validation_steps, rho=self.hparams.rho)

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
                    mag = (sample_i[0] ** 2 + sample_i[1] ** 2).sqrt() if is_complex_kind(k) else sample_i[0]
                    axes[i].imshow(mag.cpu().numpy(), cmap="gray")
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

        for j, (inter, step, sigma_val) in enumerate(zip(stats.intermediates, stats.steps, stats.sigma, strict=True)):
            img = (inter[0, 0] ** 2 + inter[0, 1] ** 2).sqrt().numpy() if inter.shape[1] == 2 else inter[0, 0].numpy()
            axes[0, j].imshow(img, cmap="gray")
            axes[0, j].axis("off")
            axes[0, j].set_title(f"step {step}\nσ={sigma_val:.3f}", fontsize=7)

        ax_stats = axes[1, 0]
        for ax in axes[1, 1:]:
            ax.axis("off")

        ax_stats.plot(stats.steps, stats.x0_std, "b-", label="x0 std", linewidth=1.5)
        ax_stats.plot(stats.steps, stats.x_std, "g--", label="x std", linewidth=1.5)
        ax_stats.set_xlabel("step")
        ax_stats.set_ylabel("std")
        ax_stats.legend(fontsize=6)
        ax_stats.set_title("Sampling statistics", fontsize=8)

        fig.suptitle(f"Sampling progression: {kind_name} @ step {self.global_step}", fontsize=10)
        fig.tight_layout()

        self.logger.experiment.log({"samples/progression": wandb.Image(fig), "global_step": self.global_step})
        plt.close(fig)

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
