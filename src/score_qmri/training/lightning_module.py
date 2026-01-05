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

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.ddim_sampler import DDIMSampler, create_fixed_noise
from score_qmri.training.losses import l1_loss, noise_loss, ssim_loss

VALIDATION_KINDS: tuple[DatasetKind, ...] = (
    DatasetKind.T1W_MAGNITUDE,
    DatasetKind.T1MAP,
    DatasetKind.COMPLEX_T1W_COIL_COMBINED,
    DatasetKind.T2MAP,
    DatasetKind.M0_MAGNITUDE,
)


@dataclass
class DiffusionHparams:
    """Type-safe hyperparameter definitions for DiffusionLightningModule."""

    base_channels: int
    num_blocks: int
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
    warmup_steps: int


class DiffusionLightningModule(LightningModule):
    """Lightning module for diffusion training with dual prediction heads."""

    hparams: DiffusionHparams  # type: ignore[assignment]

    def __init__(
        self,
        base_channels: int = 64,
        num_blocks: int = 4,
        cond_dim: int = 128,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        noise_weight: float = 1.0,
        x0_weight: float = 0.1,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        mask_weight: float = 1e-3,
        validation_size: int = 128,
        validation_steps: int = 100,
        validation_interval: int = 1000,
        warmup_steps: int = 1000,
    ) -> None:
        """
        Parameters
        ----------
        base_channels
            UNet base channel count.
        num_blocks
            UNet encoder/decoder depth.
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
        warmup_steps
            Number of warmup steps for linear LR warmup.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffusionModel(
            beta_min=beta_min,
            beta_max=beta_max,
            base_channels=base_channels,
            num_blocks=num_blocks,
            cond_dim=cond_dim,
            num_kinds=NUM_DATASET_KINDS,
        )
        self.sampler: DDIMSampler = DDIMSampler(self.model, num_steps=validation_steps)
        self._fixed_noise: dict[int, Tensor] | None = None

    def on_fit_start(self) -> None:
        if not isinstance(self.logger, WandbLogger):
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

        if wandb.run is not None:  # type: ignore[attr-defined]
            project_root = Path(__file__).parent.parent
            wandb.run.log_code(  # type: ignore[attr-defined]
                str(project_root),
                include_fn=lambda p: p.endswith(".py") or p.endswith(".yaml") or p.endswith(".toml"),
            )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return

        all_grads = torch.cat(grads)
        grad_norm = all_grads.norm()
        grad_variance = all_grads.var()

        self.log_dict(
            {"gradients/norm": grad_norm, "gradients/variance": grad_variance},
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor | None:
        x0_raw = batch["image"]
        mask = batch["mask"]
        kind = batch["kind"]

        x0 = self.model.normalize(x0_raw, kind, mask)

        self._log_normalization_stats(x0_raw, x0, mask, kind)

        t = torch.rand(x0.shape[0], device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.model.q_sample(x0, t, noise)

        output = self.model(x_t, t, kind)

        weighted_mask = torch.where(mask, 1.0, self.hparams.mask_weight)

        loss_noise = noise_loss(output.noise, noise, weighted_mask)
        loss_l1 = l1_loss(output.x0, x0, weighted_mask)
        loss_ssim = ssim_loss(output.x0, x0, weighted_mask)

        x0_loss = self.hparams.ssim_weight * loss_ssim + self.hparams.l1_weight * loss_l1
        total_loss = self.hparams.noise_weight * loss_noise + self.hparams.x0_weight * x0_loss

        if not torch.isfinite(total_loss) or total_loss > 100.0:
            self._log_debug_batch(x0_raw, output.x0, noise, output.noise, kind, t, loss_noise, loss_l1, loss_ssim, total_loss)
            self.log("train/skipped_step", 1.0, sync_dist=True)
            return None

        self.log_dict(
            {
                "train/loss": total_loss,
                "train/noise_mse": loss_noise,
                "train/x0_l1": loss_l1,
                "train/x0_ssim": loss_ssim,
            },
            prog_bar=True,
            sync_dist=True,
        )

        if self.global_step > 0 and self.global_step % 1000 == 0:
            x0_tweedie = self.model.predict_x0(x_t, t, output.noise)
            self._log_training_batch(x0, x_t, noise, output.noise, x0_tweedie, output.x0, kind, t)

        return total_loss

    def _log_normalization_stats(self, x0_raw: Tensor, x0_norm: Tensor, mask: Tensor, kind: Tensor) -> None:
        # TODO: Debug logging for normalization issues - remove once training is stable
        x_raw = x0_raw.abs() if x0_raw.is_complex() else x0_raw
        x_norm = x0_norm.abs() if x0_norm.is_complex() else x0_norm
        mask_f = mask.float()

        masked_raw = x_raw * mask_f
        masked_norm = x_norm * mask_f
        n_valid = mask_f.sum().clamp(min=1)

        raw_mean = masked_raw.sum() / n_valid
        raw_var = ((masked_raw - raw_mean * mask_f).pow(2) * mask_f).sum() / n_valid

        norm_mean = masked_norm.sum() / n_valid
        norm_var = ((masked_norm - norm_mean * mask_f).pow(2) * mask_f).sum() / n_valid

        normalizer_means = []
        normalizer_vars = []
        for k in kind.unique():
            mean_buf, var_buf, n_buf = self.model.normalizer._get_buffers(k.item())
            if n_buf.item() > 0:
                normalizer_means.append(mean_buf[:1].abs().mean())
                normalizer_vars.append(var_buf[:1].mean())

        avg_normalizer_mean = torch.stack(normalizer_means).mean() if normalizer_means else torch.tensor(0.0)
        avg_normalizer_var = torch.stack(normalizer_vars).mean() if normalizer_vars else torch.tensor(1.0)

        self.log_dict(
            {
                "norm/raw_mean": raw_mean,
                "norm/raw_var": raw_var,
                "norm/raw_min": x_raw[mask].min() if mask.any() else x_raw.min(),
                "norm/raw_max": x_raw[mask].max() if mask.any() else x_raw.max(),
                "norm/post_mean": norm_mean,
                "norm/post_var": norm_var,
                "norm/post_min": x_norm[mask].min() if mask.any() else x_norm.min(),
                "norm/post_max": x_norm[mask].max() if mask.any() else x_norm.max(),
                "norm/normalizer_mean": avg_normalizer_mean,
                "norm/normalizer_var": avg_normalizer_var,
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

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
        if not isinstance(self.logger, WandbLogger):
            return

        import matplotlib.pyplot as plt

        import wandb

        B = min(x0_gt.shape[0], 8)
        fig, axes = plt.subplots(6, B, figsize=(2.5 * B, 15), squeeze=False)

        def to_np(x: Tensor) -> list:
            return [(x[i].abs() if x.is_complex() else x[i]).detach().cpu().numpy() for i in range(B)]

        gt = to_np(x0_gt)
        noised = to_np(x_t)
        n_gt = to_np(noise_gt)
        n_pred = to_np(noise_pred)
        x0_tw = to_np(x0_tweedie)
        x0_p = to_np(x0_pred)
        kind_names = [DatasetKind(int(kind[i].item())).name for i in range(B)]

        row_labels = ["x0 GT", "x_t (noised)", "noise GT", "noise pred", "x0 Tweedie", "x0 pred"]
        for i in range(B):
            for row, data in enumerate([gt, noised, n_gt, n_pred, x0_tw, x0_p]):
                axes[row, i].imshow(data[i], cmap="gray")
                axes[row, i].axis("off")
                if i == 0:
                    axes[row, i].set_ylabel(row_labels[row], fontsize=8)
            axes[0, i].set_title(f"{kind_names[i]}\nt={t[i].item():.2f}", fontsize=7)

        fig.suptitle(f"Training batch @ step {self.global_step}", fontsize=10)
        fig.tight_layout()

        self.logger.experiment.log({"train/batch_visualization": wandb.Image(fig)})  # type: ignore[attr-defined]
        plt.close(fig)

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.global_step > 0 and self.global_step % self.hparams.validation_interval == 0:
            self._log_samples()

    def _log_debug_batch(
        self,
        x0_raw: Tensor,
        x0_pred: Tensor,
        noise: Tensor,
        noise_pred: Tensor,
        kind: Tensor,
        t: Tensor,
        loss_noise: Tensor,
        loss_l1: Tensor,
        loss_ssim: Tensor,
        total_loss: Tensor,
    ) -> None:
        """Log debug info for batches with anomalous loss values."""
        import matplotlib.pyplot as plt

        import wandb

        kind_names = [DatasetKind(k.item()).name for k in kind]
        self.log_dict(
            {
                "debug/total_loss": total_loss.detach() if torch.isfinite(total_loss) else 999.0,
                "debug/noise_loss": loss_noise.detach() if torch.isfinite(loss_noise) else 999.0,
                "debug/l1_loss": loss_l1.detach() if torch.isfinite(loss_l1) else 999.0,
                "debug/ssim_loss": loss_ssim.detach() if torch.isfinite(loss_ssim) else 999.0,
                "debug/t_mean": t.mean(),
                "debug/t_min": t.min(),
                "debug/t_max": t.max(),
                "debug/x0_raw_min": x0_raw.abs().min() if x0_raw.is_complex() else x0_raw.min(),
                "debug/x0_raw_max": x0_raw.abs().max() if x0_raw.is_complex() else x0_raw.max(),
                "debug/x0_raw_std": x0_raw.abs().std() if x0_raw.is_complex() else x0_raw.std(),
                "debug/x0_pred_min": x0_pred.abs().min() if x0_pred.is_complex() else x0_pred.min(),
                "debug/x0_pred_max": x0_pred.abs().max() if x0_pred.is_complex() else x0_pred.max(),
                "debug/x0_pred_has_nan": x0_pred.isnan().any().float(),
                "debug/x0_pred_has_inf": x0_pred.isinf().any().float(),
                "debug/noise_pred_has_nan": noise_pred.isnan().any().float(),
                "debug/noise_pred_has_inf": noise_pred.isinf().any().float(),
            },
            sync_dist=True,
        )

        if not isinstance(self.logger, WandbLogger):
            return

        B = min(x0_raw.shape[0], 8)
        fig, axes = plt.subplots(4, B, figsize=(2.5 * B, 10), squeeze=False)

        for i in range(B):
            gt = x0_raw[i].abs().cpu().numpy() if x0_raw.is_complex() else x0_raw[i].cpu().numpy()
            pred = x0_pred[i].abs().detach().cpu().numpy() if x0_pred.is_complex() else x0_pred[i].detach().cpu().numpy()
            noise_gt = noise[i].abs().cpu().numpy() if noise.is_complex() else noise[i].cpu().numpy()
            noise_p = noise_pred[i].abs().detach().cpu().numpy() if noise_pred.is_complex() else noise_pred[i].detach().cpu().numpy()

            axes[0, i].imshow(gt, cmap="gray")
            axes[0, i].set_title(f"GT\n{kind_names[i]}\nt={t[i].item():.3f}", fontsize=8)
            axes[0, i].axis("off")

            axes[1, i].imshow(pred, cmap="gray")
            axes[1, i].set_title("x0 pred", fontsize=8)
            axes[1, i].axis("off")

            axes[2, i].imshow(noise_gt, cmap="gray")
            axes[2, i].set_title("noise GT", fontsize=8)
            axes[2, i].axis("off")

            axes[3, i].imshow(noise_p, cmap="gray")
            axes[3, i].set_title("noise pred", fontsize=8)
            axes[3, i].axis("off")

        loss_str = f"{total_loss.item():.2f}" if torch.isfinite(total_loss) else "NaN/Inf"
        fig.suptitle(f"Step {self.global_step} | Loss: {loss_str}", fontsize=10)
        fig.tight_layout()

        self.logger.experiment.log({"debug/high_loss_batch": wandb.Image(fig)})  # type: ignore[attr-defined]
        plt.close(fig)

    def _log_samples(self) -> None:
        if not isinstance(self.logger, WandbLogger):
            return

        import wandb

        self.model.eval()
        size = self.hparams.validation_size

        if self._fixed_noise is None:
            self._fixed_noise = create_fixed_noise((size, size))

        with torch.no_grad():
            for kind_enum in VALIDATION_KINDS:
                k = int(kind_enum)
                kind_tensor = torch.tensor([k], device=self.device)
                sample = self.sampler.sample((1, size, size), kind_tensor, self._fixed_noise[k])
                sample = self.model.denormalize(sample, kind_tensor)

                img_np = sample[0].abs().cpu().numpy() if sample.is_complex() else sample[0].cpu().numpy()

                image = wandb.Image(img_np, caption=f"step_{self.global_step}")  # type: ignore[attr-defined]
                self.logger.experiment.log({f"samples/{kind_enum.name}": image, "global_step": self.global_step})

        self.model.train()

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
