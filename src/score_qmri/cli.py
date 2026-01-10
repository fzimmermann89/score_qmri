"""CLI - DEBUG BRANCH for DDP hang investigation."""

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
    import datetime
    import torch.distributed as dist

    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    print(f"[{ts}][DDP][r{rank}] {msg}", flush=True)


class WandbSaveConfigCallback(SaveConfigCallback):

    def setup(
        self,
        trainer: "lightning.Trainer",  # type: ignore[name-defined]  # noqa: F821
        pl_module: "lightning.LightningModule",  # type: ignore[name-defined]  # noqa: F821
        stage: str,
    ) -> None:
        _dbg(f"WandbSaveConfigCallback.setup: ENTER stage={stage}")
        _dbg(f"  trainer.global_rank={trainer.global_rank}")
        _dbg(f"  trainer.is_global_zero={trainer.is_global_zero}")
        super().setup(trainer, pl_module, stage)
        _dbg("WandbSaveConfigCallback.setup: EXIT")

    def save_config(
        self,
        trainer: "lightning.Trainer",  # type: ignore[name-defined]  # noqa: F821
        pl_module: "lightning.LightningModule",  # type: ignore[name-defined]  # noqa: F821
        stage: str,
    ) -> None:
        _dbg(f"WandbSaveConfigCallback.save_config: ENTER stage={stage}")
        _dbg("  calling super().save_config...")
        super().save_config(trainer, pl_module, stage)
        _dbg("  super().save_config DONE")
        # NOTE: artifact logging moved to on_train_start to avoid DDP hang
        _dbg("WandbSaveConfigCallback.save_config: EXIT (no artifact here)")

    def on_train_start(
        self,
        trainer: "lightning.Trainer",  # type: ignore[name-defined]  # noqa: F821
        pl_module: "lightning.LightningModule",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        _dbg("WandbSaveConfigCallback.on_train_start: ENTER")
        _dbg(f"  is_global_zero={trainer.is_global_zero}")

        if not trainer.is_global_zero:
            _dbg("  not global_zero, skipping artifact")
            return

        if not isinstance(trainer.logger, WandbLogger):
            _dbg("  not WandbLogger, skipping")
            return

        import wandb

        config_path = Path(trainer.log_dir) / self.config_filename if trainer.log_dir else None
        _dbg(f"  config_path={config_path}")

        if config_path is None or not config_path.exists():
            _dbg("  config_path missing, skipping")
            return
        if wandb.run is None:
            _dbg("  wandb.run is None, skipping")
            return

        _dbg("  logging artifact...")
        artifact = wandb.Artifact(f"config-{wandb.run.id}", type="config")
        artifact.add_file(str(config_path))
        wandb.run.log_artifact(artifact)
        _dbg("WandbSaveConfigCallback.on_train_start: EXIT (artifact logged)")


class DiffusionLightningCLI(LightningCLI):

    def __init__(self, *args, **kwargs):
        _dbg("DiffusionLightningCLI.__init__: ENTER")
        super().__init__(*args, **kwargs)
        _dbg("DiffusionLightningCLI.__init__: EXIT")

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
        _dbg("add_arguments_to_parser: ENTER")
        parser.add_argument("--job_name", type=str | None, default=None)
        parser.add_argument("--wandb_project", type=str, default="mri-diffusion")
        parser.add_argument("--wandb_entity", type=str | None, default=None)
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
        parser.set_defaults({"data": {"class_path": "score_qmri.datasets.lightning.MRIDataModule"}})
        _dbg("add_arguments_to_parser: EXIT")

    def before_instantiate_classes(self) -> None:
        _dbg("before_instantiate_classes: ENTER")
        if self.subcommand != "fit":
            _dbg("  not fit, returning")
            return

        config = getattr(self.config, self.subcommand)

        _dbg("  creating WandbLogger...")
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
        _dbg("  WandbLogger created")

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

    def instantiate_classes(self) -> None:
        _dbg("instantiate_classes: ENTER (super)")
        super().instantiate_classes()
        _dbg("instantiate_classes: EXIT")

    def after_instantiate_classes(self) -> None:
        _dbg("after_instantiate_classes: ENTER")
        if self.subcommand != "fit":
            _dbg("  not fit, returning")
            return

        config = getattr(self.config, self.subcommand)
        logger = self.trainer.logger
        _dbg(f"  trainer.global_rank={self.trainer.global_rank}")
        _dbg(f"  logger type={type(logger).__name__}")

        if not isinstance(logger, WandbLogger):
            _dbg("  not WandbLogger, returning")
            return

        _dbg("  calling _get_synced_run_name...")
        run_name = self._get_synced_run_name(config.job_name)
        _dbg(f"  run_name={run_name}")
        run_dir = Path(config.checkpoint_dir) / run_name

        if self.trainer.global_rank == 0:
            _dbg(f"  rank 0: creating run_dir={run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)

        _dbg("  BEFORE strategy.barrier('after_instantiate_mkdir')")
        self.trainer.strategy.barrier("after_instantiate_mkdir")
        _dbg("  AFTER strategy.barrier('after_instantiate_mkdir')")

        for callback in self.trainer.callbacks:  # type: ignore[attr-defined]
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = str(run_dir)
                break

        last_ckpt = run_dir / "last.ckpt"
        if last_ckpt.exists() and not config.ckpt_path:
            _dbg(f"  auto-resuming from {last_ckpt}")
            self.trainer.ckpt_path = str(last_ckpt)

        _dbg("  calling _log_hyperparams...")
        self._log_hyperparams(config, run_name)
        _dbg("after_instantiate_classes: EXIT")

    def _get_synced_run_name(self, job_name: str | None) -> str:
        _dbg(f"_get_synced_run_name: ENTER job_name={job_name}")
        if job_name:
            _dbg("  using provided job_name")
            return job_name

        rank = self.trainer.global_rank
        _dbg(f"  global_rank={rank}")

        if rank == 0:
            _dbg("  rank 0: accessing logger.experiment.name...")
            logger = self.trainer.logger
            assert isinstance(logger, WandbLogger)
            _dbg("  BEFORE logger.experiment")
            run_name: str | None = logger.experiment.name
            _dbg(f"  AFTER logger.experiment, name={run_name}")
        else:
            _dbg("  non-zero rank: run_name=None")
            run_name = None

        _dbg("  BEFORE strategy.broadcast(run_name)")
        synced_name = self.trainer.strategy.broadcast(run_name, src=0)
        _dbg(f"  AFTER strategy.broadcast, synced_name={synced_name}")
        assert isinstance(synced_name, str)
        return synced_name

    def _log_hyperparams(self, config: jsonargparse.Namespace, run_name: str) -> None:
        _dbg("_log_hyperparams: ENTER")
        logger = self.trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        _dbg("  accessing logger.experiment...")
        experiment = logger.experiment
        _dbg("  accessed logger.experiment")

        if hasattr(experiment, "config"):
            update = getattr(experiment.config, "update", None)
            if callable(update):
                _dbg("  updating experiment.config")
                update({"cli_config": config.as_dict()})

        if isinstance(self.datamodule, MRIDataModule):
            _dbg("  logging datamodule config summary")
            logger.log_hyperparams(self.datamodule.get_config_summary())

        _dbg("  logging trainer hyperparams")
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
        _dbg("  logging system info")
        logger.log_hyperparams(system_info)

        _dbg("  logging experiment info")
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
        return {"fit": {"model", "train_dataloaders", "datamodule"}}


def main() -> None:
    _dbg("main: ENTER")
    _dbg(f"  sys.argv={sys.argv}")
    _dbg(f"  LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
    _dbg(f"  RANK={os.environ.get('RANK')}")
    _dbg(f"  WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1].startswith("-")):
        sys.argv.insert(1, "fit")

    _dbg("  creating DiffusionLightningCLI...")
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
