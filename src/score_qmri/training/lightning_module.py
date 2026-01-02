"""PyTorch Lightning module for diffusion training."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.ddim_sampler import DDIMSampler, create_fixed_noise
from score_qmri.training.losses import compute_loss

# Subset of kinds to generate validation samples for
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
        )
        self.sampler: DDIMSampler = DDIMSampler(self.model, num_steps=validation_steps)
        self._fixed_noise: dict[int, Tensor] | None = None

    def on_fit_start(self) -> None:
        """Log model info, optimizer config, and source code at training start."""
        if not isinstance(self.logger, WandbLogger):
            return

        import wandb

        # Model parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Optimizer betas (AdamW defaults)
        beta1, beta2 = 0.9, 0.999

        hparams = {
            "model/total_params": total_params,
            "model/trainable_params": trainable_params,
            "optimizer/weight_decay": self.hparams.weight_decay,
            "optimizer/beta1": beta1,
            "optimizer/beta2": beta2,
        }

        self.logger.log_hyperparams(hparams)

        # Log source code
        project_root = Path(__file__).parent.parent
        wandb.run.log_code(
            str(project_root),
            include_fn=lambda p: p.endswith(".py") or p.endswith(".yaml") or p.endswith(".toml"),
        )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient statistics before optimizer step."""
        grads = [p.grad.detach().flatten() for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return

        all_grads = torch.cat(grads)
        grad_norm = all_grads.norm()
        grad_variance = all_grads.var()

        self.log_dict(
            {
                "gradients/norm": grad_norm,
                "gradients/variance": grad_variance,
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute training loss and log metrics."""
        x0 = DiffusionModel.prepare_input(batch["image"])
        mask = rearrange(batch["mask"], "b h w -> b 1 h w").expand_as(x0)
        kind = batch["kind"]

        t = torch.rand(x0.shape[0], device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.model.q_sample(x0, t, noise)

        output = self.model(x_t, t, kind)
        losses = compute_loss(
            output.x0,
            x0,
            output.noise,
            noise,
            mask,
            self.hparams.noise_weight,
            self.hparams.x0_weight,
            self.hparams.ssim_weight,
            self.hparams.l1_weight,
        )

        self.log_dict(
            {
                "train/loss": losses.total,
                "train/noise_mse": losses.noise_mse,
                "train/x0_l1": losses.x0_l1,
                "train/x0_ssim": losses.x0_ssim,
            },
            prog_bar=True,
            sync_dist=True,
        )
        return losses.total

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log validation samples at specified intervals."""
        if self.global_step > 0 and self.global_step % self.hparams.validation_interval == 0:
            self._log_samples()

    def _log_samples(self) -> None:
        """Generate and log samples for selected dataset kinds to W&B."""
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
                sample = self.sampler.sample((1, 2, size, size), kind_tensor, self._fixed_noise[k])

                # Log real channel as image, use enum name as key
                image = wandb.Image(sample[0, 0].cpu().numpy(), caption=f"step_{self.global_step}")
                self.logger.experiment.log({f"samples/{kind_enum.name}": image, "global_step": self.global_step})

        self.model.train()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
