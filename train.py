"""Main training script for diffusion models."""

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from config import TrainingConfig
from models.diffusion import DiffusionModel
from models.unet import UNet
from training.lightning_module import DiffusionLightningModule
from training.losses import LossScheduler, MSELoss, SSIML1Loss


def create_model(config: TrainingConfig) -> DiffusionModel:
    """Create diffusion model from config."""
    unet = UNet(
        in_channels=2,
        out_channels=2,
        base_channels=config.unet_base_channels,
        num_blocks=config.unet_num_blocks,
        use_film=True,
    )
    return DiffusionModel(
        unet=unet,
        num_kinds=config.num_kinds,
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    )


def create_loss_scheduler(config: TrainingConfig) -> LossScheduler:
    """Create loss scheduler from config."""
    return LossScheduler(
        mse_loss=MSELoss(),
        ssim_l1_loss=SSIML1Loss(ssim_weight=config.ssim_weight, l1_weight=config.l1_weight),
        schedule_fn=config.loss_schedule_fn,
    )


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--validation-freq", type=int, default=1000)
    parser.add_argument("--wandb-project", type=str, default="mri-diffusion")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        validation_freq=args.validation_freq,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    model = create_model(config)
    loss_scheduler = create_loss_scheduler(config)
    lightning_module = DiffusionLightningModule(model, loss_scheduler, config)

    # DataModule should be created from dataset configs (not implemented here)
    datamodule = None

    logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_run_name,
        entity=config.wandb_entity,
    )

    trainer = pl.Trainer(
        max_steps=config.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        use_distributed_sampler=False,  # Required for custom batch sampler
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=config.checkpoint_dir,
                save_top_k=config.save_top_k,
                monitor="val/mse",
                mode="min",
            )
        ],
    )

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
