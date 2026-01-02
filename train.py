"""CLI application for MRI diffusion model training."""

import os
import sys
import warnings

import jsonargparse
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from datasets.lightning import MRIDataModule
from training.lightning_module import DiffusionLightningModule

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class DiffusionLightningCLI(LightningCLI):
    """Custom CLI for diffusion model training with WandB logging."""

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
        """Add custom arguments to the parser."""
        parser.add_argument("--experiment", type=str, default="mri-diffusion", help="Name of the experiment.")
        parser.add_argument("--wandb_project", type=str, default="mri-diffusion", help="WandB project name.")
        parser.add_argument("--wandb_entity", type=str | None, default=None, help="WandB entity name.")
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path for checkpoints.")
        parser.set_defaults({"data": {"class_path": "datasets.lightning.MRIDataModule"}})

    def before_instantiate_classes(self) -> None:
        """Set up logger and callbacks before instantiating classes."""
        if self.subcommand is None or self.subcommand != "fit":
            return

        config = getattr(self.config, self.subcommand)
        experiment_name = config.experiment

        logger = WandbLogger(
            project=config.wandb_project,
            name=experiment_name,
            entity=config.wandb_entity,
            log_model=False,
            save_dir="logs",
        )

        config.trainer.logger = logger

        checkpoint_callback = ModelCheckpoint(
            monitor="train/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            dirpath=f"{config.checkpoint_dir}/{experiment_name}",
            filename="{step:06d}-{train_loss:.4f}",
            every_n_train_steps=1000,
        )

        config.trainer.callbacks = [
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
        ]

    def after_instantiate_classes(self) -> None:
        """Log config to W&B after instantiation."""
        if self.subcommand is None:
            return
        if self.trainer.logger is not None and hasattr(self.trainer.logger, "experiment"):
            config = getattr(self.config, self.subcommand)
            experiment = self.trainer.logger.experiment
            if hasattr(experiment, "config"):
                update = getattr(experiment.config, "update", None)
                if callable(update):
                    update({"cli_config": config.as_dict()})

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define trainer functions to use as subcommands of the CLI."""
        return {
            "fit": {"model", "train_dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
        }


def main() -> None:
    """Run the training CLI."""
    torch.set_float32_matmul_precision("high")

    DiffusionLightningCLI(
        DiffusionLightningModule,
        MRIDataModule,
        save_config_callback=None,
        subclass_mode_data=True,
        seed_everything_default=42,
        trainer_defaults={
            "max_steps": 100000,
            "accelerator": "auto",
            "devices": "auto",
            "precision": "16-mixed",
            "log_every_n_steps": 10,
            "gradient_clip_val": 1.0,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
        },
    )


if __name__ == "__main__":
    main()
