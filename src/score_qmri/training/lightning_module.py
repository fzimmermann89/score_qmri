"""PyTorch Lightning module for diffusion training."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
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

        if wandb.run is not None:
            project_root = Path(__file__).parent.parent
            wandb.run.log_code(
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

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x0 = batch["image"]
        mask = batch["mask"]
        kind = batch["kind"]

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
        return total_loss

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.global_step > 0 and self.global_step % self.hparams.validation_interval == 0:
            self._log_samples()

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

                img_np = sample[0].abs().cpu().numpy() if sample.is_complex() else sample[0].cpu().numpy()

                image = wandb.Image(img_np, caption=f"step_{self.global_step}")
                self.logger.experiment.log({f"samples/{kind_enum.name}": image, "global_step": self.global_step})

        self.model.train()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
