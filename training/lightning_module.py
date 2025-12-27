"""PyTorch Lightning module for diffusion training."""

import torch
from einops import rearrange
from lightning import LightningModule
from torch import Tensor

from models.diffusion import DiffusionModel
from models.unet import UNet
from sampling.ddpm_sampler import DDPMSampler
from training.losses import LossScheduler, MSELoss, SSIML1Loss


class DiffusionLightningModule(LightningModule):
    """Lightning module for continuous diffusion training.

    All hyperparameters are exposed for CLI configuration via jsonargparse.
    """

    def __init__(
        self,
        # UNet architecture
        unet_base_channels: int = 64,
        unet_num_blocks: int = 4,
        # Diffusion parameters
        num_kinds: int = 3,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # Training parameters
        learning_rate: float = 1e-4,
        # Loss parameters
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        loss_switch_step: int = 10000,
        # Validation parameters
        num_validation_samples: int = 4,
        sampler_num_steps: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        unet = UNet(
            in_channels=2,
            out_channels=2,
            base_channels=unet_base_channels,
            num_blocks=unet_num_blocks,
            use_film=True,
        )
        self.model = DiffusionModel(
            unet=unet,
            num_kinds=num_kinds,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.loss_scheduler = LossScheduler(
            mse_loss=MSELoss(),
            ssim_l1_loss=SSIML1Loss(ssim_weight=ssim_weight, l1_weight=l1_weight),
            schedule_fn=lambda step: 0.0 if step < loss_switch_step else 1.0,
        )

        self.sampler = DDPMSampler(self.model, num_steps=sampler_num_steps)

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

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Execute validation step with sampling."""
        images, masks, kinds = batch["image"], batch["mask"], batch["kind"]
        B = min(images.shape[0], self.hparams.num_validation_samples)  # type: ignore[attr-defined]

        x0 = self.model.prepare_input(images[:B], kinds[:B])
        sampled = self.sampler.sample(
            (B, 2, x0.shape[2], x0.shape[3]), self.device, kinds[:B]
        )

        mask_4d = rearrange(masks[:B], "b h w -> b 1 h w")
        mse = MSELoss()(sampled[:, :1], x0[:, :1], mask_4d)
        self.log("val/mse", mse, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)  # type: ignore[attr-defined]
