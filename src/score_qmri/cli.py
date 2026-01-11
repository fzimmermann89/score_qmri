"""CLI application for MRI diffusion model training and generation."""

import os
import sys
import warnings
from pathlib import Path

import jsonargparse
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

from score_qmri.datasets.lightning import MRIDataModule
from score_qmri.training.lightning_module import DiffusionLightningModule
from score_qmri.utils.config import normalize_config

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

torch.set_float32_matmul_precision("high")


class WandbSaveConfigCallback(SaveConfigCallback):
    """Save config and log as W&B artifact."""

    def save_config(
        self,
        trainer: "Trainer",
        pl_module: "DiffusionLightningModule",
        stage: str,
    ) -> None:
        super().save_config(trainer, pl_module, stage)

    def on_train_start(
        self,
        trainer: "Trainer",
        pl_module: "DiffusionLightningModule",
    ) -> None:
        """Log config as W&B artifact after training starts (DDP-safe)."""
        if not isinstance(trainer.logger, WandbLogger):
            return

        log_dir = trainer.log_dir
        trainer.strategy.barrier("on_train_start_log_dir")

        if not trainer.is_global_zero:
            return

        import wandb

        config_path = Path(log_dir) / self.config_filename if log_dir else None
        if config_path is None or not config_path.exists():
            return
        if wandb.run is None:
            return
        artifact = wandb.Artifact(f"config-{wandb.run.id}", type="config")
        artifact.add_file(str(config_path))
        wandb.run.log_artifact(artifact)


class DiffusionTrainer(Trainer):
    """Trainer with generate subcommand for sample generation."""

    def generate(
        self,
        model: DiffusionLightningModule,
        ckpt_path: str,
        kind: str = "T1W_MAGNITUDE",
        n: int = 4,
        size: int = 256,
        sampler: str = "ddim",
        steps: int = 100,
        eta: float = 0.0,
        seed: int | None = None,
        output: str | None = None,
        device: str = "cuda",
    ) -> None:
        """Generate samples from a trained diffusion model.

        Parameters
        ----------
        model
            The model instance (instantiated from config).
        ckpt_path
            Path to trained model checkpoint (.ckpt file).
        kind
            Dataset kind name for conditioning (e.g., T1W_MAGNITUDE, T1MAP).
        n
            Number of samples to generate.
        size
            Image size (height = width).
        sampler
            Sampling algorithm: "ddim" or "ddpm".
        steps
            Number of denoising steps.
        eta
            DDIM stochasticity (0 = deterministic, 1 = full stochastic).
        seed
            Random seed for reproducibility.
        output
            Output PNG path. If None, displays interactive window.
        device
            Device for generation ("cuda" or "cpu").
        """
        from score_qmri.datasets.kind import DatasetKind
        from score_qmri.sampling.generate import generate_samples
        from score_qmri.utils.plot import plot_samples_grid

        try:
            kind_enum = DatasetKind[kind.upper()]
        except KeyError:
            valid = ", ".join(k.name for k in DatasetKind)
            raise SystemExit(f"Unknown kind: {kind!r}. Valid: {valid}") from None

        diffusion_model = model.model
        diffusion_model.load_checkpoint(Path(ckpt_path))

        diffusion_model.to(device)
        diffusion_model.eval()

        print(f"Generating {n} samples of {kind_enum.name}...")
        print(f"  Sampler: {sampler.upper()}, Steps: {steps}, Size: {size}x{size}")

        samples = generate_samples(
            model=diffusion_model,
            kind=kind_enum,
            n_samples=n,
            size=size,
            sampler_type=sampler,
            num_steps=steps,
            eta=eta,
            seed=seed,
        )

        title = f"{kind_enum.name} ({sampler.upper()}, {steps} steps)"
        output_path = Path(output) if output else None
        plot_samples_grid(samples, output_path, title)


class DiffusionLightningCLI(LightningCLI):
    """Custom CLI for diffusion model training and generation."""

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
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
        if self.subcommand != "fit":
            return

        config = getattr(self.config, self.subcommand)

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

    def after_instantiate_classes(self) -> None:
        if self.subcommand != "fit":
            return

        config = getattr(self.config, self.subcommand)
        logger = self.trainer.logger

        if not isinstance(logger, WandbLogger):
            return

        run_name = self._get_synced_run_name(config.job_name)
        run_dir = Path(config.checkpoint_dir) / run_name

        if self.trainer.global_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.strategy.barrier("after_instantiate_mkdir")

        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.dirpath = str(run_dir)
                break

        last_ckpt = run_dir / "last.ckpt"
        if last_ckpt.exists() and not config.ckpt_path:
            print(f"Auto-resuming from: {last_ckpt}")
            self.trainer.ckpt_path = str(last_ckpt)

        self._log_hyperparams(config, run_name)

    def _get_synced_run_name(self, job_name: str | None) -> str:
        if job_name:
            return job_name

        if self.trainer.global_rank == 0:
            logger = self.trainer.logger
            assert isinstance(logger, WandbLogger)
            run_name: str | None = logger.experiment.name
        else:
            run_name = None

        synced_name = self.trainer.strategy.broadcast(run_name, src=0)
        assert isinstance(synced_name, str)
        return synced_name

    def _log_hyperparams(self, config: jsonargparse.Namespace, run_name: str) -> None:
        logger = self.trainer.logger
        if not isinstance(logger, WandbLogger):
            return

        experiment = logger.experiment
        if hasattr(experiment, "config"):
            update = getattr(experiment.config, "update", None)
            if callable(update):
                normalized_config = normalize_config(config.as_dict())
                update({"cli_config": normalized_config})

        if isinstance(self.datamodule, MRIDataModule):
            logger.log_hyperparams(self.datamodule.get_config_summary())

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

        system_info: dict[str, object] = {
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

        logger.log_hyperparams(
            {
                "experiment/job_name": config.job_name,
                "experiment/run_name": run_name,
                "experiment/wandb_project": config.wandb_project,
                "experiment/checkpoint_dir": config.checkpoint_dir,
            }
        )

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        return {
            "fit": {"model", "train_dataloaders", "datamodule"},
            "generate": {"model"},
        }


def main() -> None:
    """Run the CLI."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    elif len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        sys.argv.insert(1, "fit")

    DiffusionLightningCLI(
        DiffusionLightningModule,
        MRIDataModule,
        trainer_class=DiffusionTrainer,
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
            "strategy": "ddp",
            "enable_progress_bar": True,
        },
    )


if __name__ == "__main__":
    main()
