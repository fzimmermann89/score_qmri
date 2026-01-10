"""CLI application for MRI diffusion model training. DEBUG BRANCH for DDP hang investigation."""

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

torch.set_float32_matmul_precision("high")


def _dbg(msg: str) -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    print(f"[DDP-DEBUG][rank={rank}] {msg}", flush=True)


class WandbSaveConfigCallback(SaveConfigCallback):
    """Save config and log as W&B artifact."""

    def save_config(
        self,
        trainer: "lightning.Trainer",  # type: ignore[name-defined]  # noqa: F821
        pl_module: "lightning.LightningModule",  # type: ignore[name-defined]  # noqa: F821
        stage: str,
    ) -> None:
        """Save config to file and log as W&B artifact."""
        _dbg("WandbSaveConfigCallback.save_config: ENTER")
        _dbg("WandbSaveConfigCallback.save_config: calling super().save_config")
        super().save_config(trainer, pl_module, stage)
        _dbg("WandbSaveConfigCallback.save_config: super().save_config DONE")

        if not isinstance(trainer.logger, WandbLogger):
            _dbg("WandbSaveConfigCallback.save_config: not WandbLogger, returning")
            return

        import wandb

        config_path = Path(trainer.log_dir) / self.config_filename if trainer.log_dir else None
        if config_path is None or not config_path.exists():
            _dbg("WandbSaveConfigCallback.save_config: config_path missing, returning")
            return
        if wandb.run is None:
            _dbg("WandbSaveConfigCallback.save_config: wandb.run is None, returning")
            return
        _dbg("WandbSaveConfigCallback.save_config: logging artifact")
        artifact = wandb.Artifact(f"config-{wandb.run.id}", type="config")
        artifact.add_file(str(config_path))
        wandb.run.log_artifact(artifact)
        _dbg("WandbSaveConfigCallback.save_config: EXIT")


class DiffusionLightningCLI(LightningCLI):
    """Custom CLI for diffusion model training with WandB logging."""

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
        """Add custom arguments to the parser."""
        parser.add_argument(
            "--job_name",
            type=str | None,
            default=None,
            help="Unique job name for resumption. WandB auto-generates if omitted.",
        )
        parser.add_argument("--wandb_project", type=str, default="mri-diffusion", help="WandB project name.")
        parser.add_argument("--wandb_entity", type=str | None, default=None, help="WandB entity name.")
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Base path for checkpoints.")
        parser.set_defaults({"data": {"class_path": "score_qmri.datasets.lightning.MRIDataModule"}})

    def before_instantiate_classes(self) -> None:
        """Set up logger and callbacks before instantiating classes."""
        _dbg("before_instantiate_classes: ENTER")
        if self.subcommand != "fit":
            _dbg("before_instantiate_classes: not fit, returning")
            return

        config = getattr(self.config, self.subcommand)

        _dbg("before_instantiate_classes: creating WandbLogger")
        logger = WandbLogger(
            id=config.job_name,
            name=config.job_name,
            resume="allow" if config.job_name else None,
            project=config.wandb_project,
            entity=config.wandb_entity,
            log_model=False,
            save_dir="logs",
        )
        config.trainer.logger = logger
        _dbg("before_instantiate_classes: WandbLogger created and assigned")

        checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            monitor="train/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{step:06d}-{train_loss:.4f}",
            every_n_train_steps=1000,
        )

        config.trainer.callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=3),
            checkpoint_callback,
        ]
        _dbg("before_instantiate_classes: EXIT")

    def after_instantiate_classes(self) -> None:
        """Set checkpoint directory using WandB's run name (synced across ranks)."""
        _dbg("after_instantiate_classes: ENTER")
        if self.subcommand != "fit":
            _dbg("after_instantiate_classes: not fit, returning")
            return

        config = getattr(self.config, self.subcommand)
        logger = self.trainer.logger

        if not isinstance(logger, WandbLogger):
            _dbg("after_instantiate_classes: not WandbLogger, returning")
            return

        _dbg("after_instantiate_classes: calling _get_synced_run_name")
        run_name = self._get_synced_run_name(config.job_name)
        _dbg(f"after_instantiate_classes: run_name={run_name}")
        run_dir = Path(config.checkpoint_dir) / run_name

        _dbg(f"after_instantiate_classes: global_rank={self.trainer.global_rank}")
        if self.trainer.global_rank == 0:
            _dbg("after_instantiate_classes: rank 0 creating run_dir")
            run_dir.mkdir(parents=True, exist_ok=True)

        _dbg("after_instantiate_classes: BEFORE strategy.barrier()")
        self.trainer.strategy.barrier()
        _dbg("after_instantiate_classes: AFTER strategy.barrier()")

        for callback in self.trainer.callbacks:  # type: ignore[attr-defined]
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = str(run_dir)
                break

        last_ckpt = run_dir / "last.ckpt"
        if last_ckpt.exists() and not config.ckpt_path:
            print(f"Auto-resuming from: {last_ckpt}")
            self.trainer.ckpt_path = str(last_ckpt)

        _dbg("after_instantiate_classes: calling _log_hyperparams")
        self._log_hyperparams(config, run_name)
        _dbg("after_instantiate_classes: EXIT")

    def _get_synced_run_name(self, job_name: str | None) -> str:
        """Get WandB run name, synchronized across all DDP ranks."""
        _dbg(f"_get_synced_run_name: ENTER, job_name={job_name}")
        if job_name:
            _dbg("_get_synced_run_name: using provided job_name")
            return job_name

        _dbg(f"_get_synced_run_name: global_rank={self.trainer.global_rank}")
        if self.trainer.global_rank == 0:
            _dbg("_get_synced_run_name: rank 0, accessing logger.experiment.name")
            logger = self.trainer.logger
            assert isinstance(logger, WandbLogger)
            _dbg("_get_synced_run_name: BEFORE logger.experiment")
            run_name: str | None = logger.experiment.name
            _dbg(f"_get_synced_run_name: AFTER logger.experiment, name={run_name}")
        else:
            _dbg("_get_synced_run_name: non-zero rank, run_name=None")
            run_name = None

        _dbg("_get_synced_run_name: BEFORE strategy.broadcast")
        synced_name = self.trainer.strategy.broadcast(run_name, src=0)
        _dbg(f"_get_synced_run_name: AFTER strategy.broadcast, synced_name={synced_name}")
        assert isinstance(synced_name, str)
        return synced_name

    def _log_hyperparams(self, config: jsonargparse.Namespace, run_name: str) -> None:
        """Log hyperparams to WandB. Safe to call from all ranks."""
        _dbg("_log_hyperparams: ENTER")
        logger = self.trainer.logger
        if not isinstance(logger, WandbLogger):
            _dbg("_log_hyperparams: not WandbLogger, returning")
            return

        _dbg("_log_hyperparams: accessing logger.experiment")
        experiment = logger.experiment
        _dbg("_log_hyperparams: accessed logger.experiment")
        if hasattr(experiment, "config"):
            update = getattr(experiment.config, "update", None)
            if callable(update):
                _dbg("_log_hyperparams: updating experiment.config")
                update({"cli_config": config.as_dict()})

        if isinstance(self.datamodule, MRIDataModule):
            _dbg("_log_hyperparams: logging datamodule config summary")
            logger.log_hyperparams(self.datamodule.get_config_summary())

        _dbg("_log_hyperparams: logging trainer hyperparams")
        logger.log_hyperparams(
            {
                "trainer/max_steps": self.trainer.max_steps,
                "trainer/max_epochs": self.trainer.max_epochs,
                "trainer/precision": str(self.trainer.precision),
                "trainer/gradient_clip_val": self.trainer.gradient_clip_val,
                "trainer/accumulate_grad_batches": self.trainer.accumulate_grad_batches,
                "trainer/num_devices": self.trainer.num_devices,
                "trainer/accelerator": str(self.trainer.accelerator),
            }
        )

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
        _dbg("_log_hyperparams: logging system info")
        logger.log_hyperparams(system_info)

        _dbg("_log_hyperparams: logging experiment info")
        logger.log_hyperparams(
            {
                "experiment/job_name": config.job_name,
                "experiment/run_name": run_name,
                "experiment/wandb_project": config.wandb_project,
                "experiment/checkpoint_dir": config.checkpoint_dir,
            }
        )
        _dbg("_log_hyperparams: EXIT")

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define trainer functions to use as subcommands of the CLI."""
        return {"fit": {"model", "train_dataloaders", "datamodule"}}


def main() -> None:
    """Run the training CLI."""
    _dbg("main: ENTER")
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "fit")

    _dbg("main: creating DiffusionLightningCLI")
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
            "strategy": "ddp_find_unused_parameters_true",
            "enable_progress_bar": True,
        },
    )
    _dbg("main: EXIT")


if __name__ == "__main__":
    main()
