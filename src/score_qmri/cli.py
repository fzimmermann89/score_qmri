"""CLI application for MRI diffusion model training."""

import os
import sys
import warnings
from pathlib import Path

import jsonargparse
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

from score_qmri.datasets.lightning import MRIDataModule
from score_qmri.training.lightning_module import DiffusionLightningModule

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class WandbSaveConfigCallback(SaveConfigCallback):
    """Save config and log as W&B artifact."""

    def save_config(self, trainer: "lightning.Trainer", pl_module: "lightning.LightningModule", stage: str) -> None:  # type: ignore[name-defined]  # noqa: F821
        """Save config to file and log as W&B artifact."""
        super().save_config(trainer, pl_module, stage)

        if not isinstance(trainer.logger, WandbLogger):
            return

        import wandb

        config_path = Path(trainer.log_dir) / self.config_filename if trainer.log_dir else None
        if config_path is None or not config_path.exists():
            return

        # Log config as artifact
        artifact = wandb.Artifact(f"config-{wandb.run.id}", type="config")
        artifact.add_file(str(config_path))
        wandb.run.log_artifact(artifact)


class DiffusionLightningCLI(LightningCLI):
    """Custom CLI for diffusion model training with WandB logging."""

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
        """Add custom arguments to the parser."""
        parser.add_argument("--job_name", type=str, required=True, help="Unique job name for resumption.")
        parser.add_argument("--experiment", type=str, default="mri-diffusion", help="Name of the experiment.")
        parser.add_argument("--wandb_project", type=str, default="mri-diffusion", help="WandB project name.")
        parser.add_argument("--wandb_entity", type=str | None, default=None, help="WandB entity name.")
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path for checkpoints.")
        parser.set_defaults({"data": {"class_path": "score_qmri.datasets.lightning.MRIDataModule"}})

    def before_instantiate_classes(self) -> None:
        """Set up logger and callbacks before instantiating classes."""
        if self.subcommand is None or self.subcommand != "fit":
            return

        config = getattr(self.config, self.subcommand)

        run_id = config.job_name
        run_dir = Path(config.checkpoint_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config.trainer.default_root_dir = str(run_dir)

        logger = WandbLogger(
            id=run_id,
            resume="allow",
            project=config.wandb_project,
            name=config.experiment,
            entity=config.wandb_entity,
            log_model=False,
            save_dir="logs",
        )

        config.trainer.logger = logger

        last_ckpt_path = run_dir / "last.ckpt"
        if last_ckpt_path.exists() and last_ckpt_path.is_file() and not config.ckpt_path:
            print(f"Resuming from checkpoint: {last_ckpt_path}")
            config.ckpt_path = str(last_ckpt_path)

        checkpoint_callback = ModelCheckpoint(
            monitor="train/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            dirpath=str(run_dir),
            filename="{step:06d}-{train_loss:.4f}",
            every_n_train_steps=1000,
        )

        config.trainer.callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=3),
            checkpoint_callback,
        ]

    def after_instantiate_classes(self) -> None:
        """Log flattened hyperparams to W&B after instantiation."""
        if self.subcommand is None:
            return

        logger = self.trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        config = getattr(self.config, self.subcommand)
        experiment = logger.experiment

        # Log full CLI config
        if hasattr(experiment, "config"):
            update = getattr(experiment.config, "update", None)
            if callable(update):
                update({"cli_config": config.as_dict()})

        # Log flattened datamodule config
        if isinstance(self.datamodule, MRIDataModule):
            data_summary = self.datamodule.get_config_summary()
            logger.log_hyperparams(data_summary)

        # Log trainer config
        trainer_hparams = {
            "trainer/max_steps": self.trainer.max_steps,
            "trainer/max_epochs": self.trainer.max_epochs,
            "trainer/precision": str(self.trainer.precision),
            "trainer/gradient_clip_val": self.trainer.gradient_clip_val,
            "trainer/accumulate_grad_batches": self.trainer.accumulate_grad_batches,
            "trainer/num_devices": self.trainer.num_devices,
            "trainer/accelerator": str(self.trainer.accelerator),
        }
        logger.log_hyperparams(trainer_hparams)

        # Log system info
        system_info = {
            "system/cuda_available": torch.cuda.is_available(),
            "system/cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "system/torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            system_info["system/cuda_version"] = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                system_info[f"system/gpu_{i}_name"] = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                system_info[f"system/gpu_{i}_memory_gb"] = round(mem, 2)

        logger.log_hyperparams(system_info)

        # Log experiment metadata
        experiment_hparams = {
            "experiment/name": config.experiment,
            "experiment/job_name": config.job_name,
            "experiment/wandb_project": config.wandb_project,
            "experiment/checkpoint_dir": config.checkpoint_dir,
        }
        logger.log_hyperparams(experiment_hparams)

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define trainer functions to use as subcommands of the CLI."""
        return {"fit": {"model", "train_dataloaders", "datamodule"}}


def main() -> None:
    """Run the training CLI."""
    torch.set_float32_matmul_precision("high")

    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "fit")

    DiffusionLightningCLI(
        DiffusionLightningModule,
        MRIDataModule,
        save_config_callback=WandbSaveConfigCallback,
        subclass_mode_data=True,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
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
