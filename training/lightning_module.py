"""PyTorch Lightning module for diffusion training."""

from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor

from models.diffusion import DiffusionModel
from sampling.ddpm_sampler import DDPMSampler
from training.losses import LossScheduler, MSELoss


class DiffusionLightningModule(pl.LightningModule):
    """Lightning module for continuous diffusion training.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model instance.
    loss_scheduler : LossScheduler
        Loss scheduler for interpolating between MSE and SSIM+L1.
    config : Any
        Training configuration with learning_rate and validation_freq attributes.
    num_validation_samples : int
        Number of samples to generate during validation.
    """

    def __init__(
        self,
        model: DiffusionModel,
        loss_scheduler: LossScheduler,
        config: Any,
        num_validation_samples: int = 4,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_scheduler = loss_scheduler
        self.config = config
        self.num_validation_samples = num_validation_samples
        self.sampler = DDPMSampler(model, num_steps=50)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute single training step."""
        images, masks, kinds = batch["image"], batch["mask"], batch["kind"]
        B = images.shape[0]

        x0 = self.model.prepare_input(images, kinds)
        t = torch.randint(0, self.model.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.model.q_sample(x0, t, noise)

        pred_x0 = self.model(x_t, t, kinds)
        mask_4d = rearrange(masks, "b h w -> b 1 h w").expand_as(x0)
        loss = self.loss_scheduler(pred_x0, x0, mask_4d)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Trigger validation at specified intervals."""
        if self.global_step > 0 and self.global_step % self.config.validation_freq == 0:
            self.trainer.validate(self, self.trainer.val_dataloaders)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Execute validation step with sampling."""
        images, masks, kinds = batch["image"], batch["mask"], batch["kind"]
        B = min(images.shape[0], self.num_validation_samples)

        x0 = self.model.prepare_input(images[:B], kinds[:B])
        sampled = self.sampler.sample(
            (B, 2, x0.shape[2], x0.shape[3]), self.device, kinds[:B]
        )

        mask_4d = rearrange(masks[:B], "b h w -> b 1 h w")
        mse = MSELoss()(sampled[:, :1], x0[:, :1], mask_4d)
        self.log("val/mse", mse, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
