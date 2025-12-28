"""PyTorch Lightning module for diffusion training."""

from collections.abc import Mapping
from typing import Any

import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from models.diffusion import DiffusionModel
from sampling.ddim_sampler import DDIMSampler, create_fixed_noise
from training.losses import compute_loss


class DiffusionLightningModule(LightningModule):
    """Lightning module for diffusion training with dual prediction heads."""

    def __init__(
        self,
        base_channels: int = 64,
        num_blocks: int = 4,
        cond_dim: int = 128,
        num_kinds: int = 3,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        learning_rate: float = 1e-4,
        noise_weight: float = 1.0,
        x0_weight: float = 0.1,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        validation_size: int = 128,
        validation_steps: int = 100,
        validation_interval: int = 1000,
    ) -> None:
        """
        Initialize diffusion training module.

        Parameters
        ----------
        base_channels
            UNet base channel count.
        num_blocks
            UNet encoder/decoder depth.
        cond_dim
            Conditioning embedding dimension.
        num_kinds
            Number of dataset types for conditioning.
        num_timesteps
            Total diffusion timesteps.
        beta_start
            Starting noise schedule value.
        beta_end
            Ending noise schedule value.
        learning_rate
            Optimizer learning rate.
        noise_weight
            Weight for noise prediction loss.
        x0_weight
            Weight for clean image prediction loss.
        ssim_weight
            SSIM weight within x0 loss.
        l1_weight
            L1 weight within x0 loss.
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
            num_kinds=num_kinds,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            base_channels=base_channels,
            num_blocks=num_blocks,
            cond_dim=cond_dim,
        )
        self.sampler = DDIMSampler(self.model, num_steps=validation_steps)
        self._fixed_noise: dict[int, Tensor] | None = None

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute training loss and log metrics."""
        x0 = DiffusionModel.prepare_input(batch["image"])
        mask = rearrange(batch["mask"], "b h w -> b 1 h w").expand_as(x0)
        kind = batch["kind"]

        t = torch.randint(0, self.model.num_timesteps, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.model.q_sample(x0, t, noise)

        output = self.model(x_t, t, kind)
        hp = self.hparams
        losses = compute_loss(
            output.x0, x0, output.noise, noise, mask,
            hp.noise_weight, hp.x0_weight, hp.ssim_weight, hp.l1_weight,
        )

        self.log_dict({
            "train/loss": losses.total,
            "train/noise_mse": losses.noise_mse,
            "train/x0_l1": losses.x0_l1,
            "train/x0_ssim": losses.x0_ssim,
        }, prog_bar=True)
        return losses.total

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log validation samples at specified intervals."""
        hp = self.hparams
        if self.global_step > 0 and self.global_step % hp.validation_interval == 0:
            self._log_samples()

    def _log_samples(self) -> None:
        """Generate and log samples for each dataset kind to W&B."""
        if not isinstance(self.logger, WandbLogger):
            return

        import wandb

        self.model.eval()
        hp = self.hparams
        size = hp.validation_size

        if self._fixed_noise is None:
            self._fixed_noise = create_fixed_noise(hp.num_kinds, (size, size))

        images = []
        for k in range(hp.num_kinds):
            kind = torch.tensor([k], device=self.device)
            sample = self.sampler.sample((1, 2, size, size), self.device, kind, self._fixed_noise[k])
            images.append(wandb.Image(sample[0, 0].cpu().numpy(), caption=f"kind_{k}"))

        self.logger.experiment.log({"validation/samples": images, "global_step": self.global_step})
        self.model.train()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
