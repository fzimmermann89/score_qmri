"""PyTorch Lightning module for diffusion training."""

from collections.abc import Mapping
from typing import Any

import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from models.diffusion import DiffusionModel
from models.unet import UNet
from sampling.ddim_sampler import DDIMSampler, create_fixed_noise
from training.losses import compute_diffusion_loss


class DiffusionLightningModule(LightningModule):
    """Lightning module for diffusion training with dual prediction heads.

    Predicts both clean image (x0) and noise (epsilon).
    - Noise prediction uses L_simple (MSE) loss
    - Clean image prediction uses SSIM + L1 loss (auxiliary task)

    Validation samples one image per dataset kind using DDIM with fixed noise.
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
        # Loss weights
        noise_loss_weight: float = 1.0,
        x0_loss_weight: float = 0.1,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5,
        # Validation parameters
        validation_image_size: int = 128,
        ddim_sampling_steps: int = 100,
        validation_interval_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        unet = UNet(
            in_channels=2,
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

        self.sampler = DDIMSampler(self.model, num_steps=ddim_sampling_steps, eta=0.0)
        self._fixed_noise: dict[int, Tensor] | None = None

    def _get_fixed_noise(self) -> dict[int, Tensor]:
        """Lazily create fixed noise for validation sampling."""
        if self._fixed_noise is None:
            self._fixed_noise = create_fixed_noise(
                num_kinds=self.hparams.num_kinds,  # type: ignore[attr-defined]
                image_shape=(self.hparams.validation_image_size, self.hparams.validation_image_size),  # type: ignore[attr-defined]
                seed=42,
            )
        return self._fixed_noise

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute single training step.

        Logs L_simple (noise MSE), L1, and SSIM losses.
        """
        images, masks, kinds = batch["image"], batch["mask"], batch["kind"]
        B = images.shape[0]

        x0 = self.model.prepare_input(images, kinds)
        t = torch.randint(0, self.model.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.model.q_sample(x0, t, noise)

        output = self.model(x_t, t, kinds)

        mask_4d = rearrange(masks, "b h w -> b 1 h w").expand_as(x0)
        losses = compute_diffusion_loss(
            x0_pred=output.x0_pred,
            x0_target=x0,
            noise_pred=output.noise_pred,
            noise_target=noise,
            mask=mask_4d,
            noise_weight=self.hparams.noise_loss_weight,  # type: ignore[attr-defined]
            x0_weight=self.hparams.x0_loss_weight,  # type: ignore[attr-defined]
            ssim_weight=self.hparams.ssim_weight,  # type: ignore[attr-defined]
            l1_weight=self.hparams.l1_weight,  # type: ignore[attr-defined]
        )

        self.log("train/loss", losses.total, prog_bar=True)
        self.log("train/l_simple", losses.l_simple)
        self.log("train/l1", losses.l1)
        self.log("train/ssim", losses.ssim)

        return losses.total

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Run validation sampling at specified intervals."""
        if self.global_step > 0 and self.global_step % self.hparams.validation_interval_steps == 0:  # type: ignore[attr-defined]
            self._log_validation_samples()

    def _log_validation_samples(self) -> None:
        """Generate and log samples for each dataset kind to W&B."""
        if not isinstance(self.logger, WandbLogger):
            return

        import wandb

        self.model.eval()
        fixed_noise = self._get_fixed_noise()
        image_size = self.hparams.validation_image_size  # type: ignore[attr-defined]

        images_to_log = []
        for kind_idx in range(self.hparams.num_kinds):  # type: ignore[attr-defined]
            kind_tensor = torch.tensor([kind_idx], device=self.device)
            noise = fixed_noise[kind_idx]

            sample = self.sampler.sample(
                shape=(1, 2, image_size, image_size),
                device=self.device,
                kind=kind_tensor,
                initial_noise=noise,
            )

            # Take first channel (real part / magnitude) for visualization
            img = sample[0, 0].cpu().numpy()
            images_to_log.append(wandb.Image(img, caption=f"kind_{kind_idx}"))

        self.logger.experiment.log(
            {"validation/samples": images_to_log, "global_step": self.global_step}
        )
        self.model.train()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)  # type: ignore[attr-defined]
