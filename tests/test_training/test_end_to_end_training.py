"""End-to-end training tests with mocked datasets."""

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.cli import LightningCLI

from score_qmri.cli import DiffusionLightningCLI
from score_qmri.datasets.config import AugmentationConfig
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import ConstantSchedule, MagnitudeDatasetConfig, MRIDataModule
from score_qmri.training.lightning_module import DiffusionLightningModule


@pytest.fixture
def training_yaml_config(sample_h5_file_magnitude: str) -> Generator[str, None, None]:
    """Create temporary YAML config file for training tests."""
    config_content = f"""model:
  base_channels: 16
  num_res_blocks: 2
  channel_mult:
    - 1
    - 2
  cond_dim: 64
  sigma_min: 0.01
  sigma_max: 10.0
  sigma_data: 0.5
  p_mean: -1.2
  p_std: 1.2
  rho: 7.0
  learning_rate: 1e-4
  main_weight: 1.0
  aux_weight: 0.1
  ssim_weight: 0.5
  l1_weight: 0.5
  validation_steps: 10
  validation_interval: 100

data:
  class_path: score_qmri.datasets.lightning.MRIDataModule
  init_args:
    dataset_configs:
      - type: magnitude
        kind: T1W_MAGNITUDE
        h5_path: {sample_h5_file_magnitude}
        schedule:
          type: constant
          weight: 1.0
    batch_size: 2
    num_workers: 0
    base_seed: 42

trainer:
  max_steps: 2
  max_epochs: 1
  accelerator: auto
  devices: 1
  precision: 32-true
  log_every_n_steps: 1
  logger: null
  enable_checkpointing: false
  enable_progress_bar: false
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    yield config_path

    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def test_augmentation_config():
    return AugmentationConfig(
        geometric_prob=0.0,
        contrast_prob=0.0,
        phase_prob=0.0,
    )


@pytest.fixture
def magnitude_dataset_config(
    sample_h5_file_magnitude: str,
    test_augmentation_config: AugmentationConfig,
) -> MagnitudeDatasetConfig:
    return MagnitudeDatasetConfig(
        type="magnitude",
        kind=DatasetKind.T1W_MAGNITUDE,
        h5_path=sample_h5_file_magnitude,
        augmentation_config=test_augmentation_config,
        schedule=ConstantSchedule(type="constant", weight=1.0),
    )


@pytest.fixture
def fast_trainer():
    return pl.Trainer(
        max_steps=2,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        precision="32-true",
        log_every_n_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


@pytest.fixture
def small_model_kwargs():
    return {
        "base_channels": 16,
        "num_res_blocks": 2,
        "channel_mult": (1, 2),
        "cond_dim": 64,
        "sigma_min": 0.01,
        "sigma_max": 10.0,
        "sigma_data": 0.5,
        "p_mean": -1.2,
        "p_std": 1.2,
        "rho": 7.0,
        "learning_rate": 1e-4,
        "main_weight": 1.0,
        "aux_weight": 0.1,
        "ssim_weight": 0.5,
        "l1_weight": 0.5,
        "validation_steps": 10,
        "validation_interval": 100,
    }


def test_training_with_single_dataset(
    magnitude_dataset_config,
    fast_trainer,
    small_model_kwargs,
):
    """Verify training completes with single dataset and parameters receive gradients."""
    try:
        from score_qmri.training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    datamodule = MRIDataModule(
        dataset_configs=[magnitude_dataset_config],
        batch_size=2,
        num_workers=0,
        base_seed=42,
    )
    model = DiffusionLightningModule(**small_model_kwargs)

    datamodule.setup("fit")
    fast_trainer.fit(model, datamodule)

    assert fast_trainer.global_step == 2
    assert fast_trainer.logged_metrics is not None

    params_have_grad = any(param.grad is not None for param in model.parameters())
    assert params_have_grad


def test_training_with_multiple_datasets(
    magnitude_dataset_config: MagnitudeDatasetConfig,
    sample_h5_file_mapping: str,
    test_augmentation_config: AugmentationConfig,
    fast_trainer: pl.Trainer,
    small_model_kwargs: dict[str, Any],
) -> None:
    """Verify multi-dataset concat works and both dataset configs are retained."""
    try:
        from score_qmri.training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    from score_qmri.datasets.lightning import MappingDatasetConfig

    mapping_config = MappingDatasetConfig(
        type="mapping",
        kind=DatasetKind.T1MAP,
        h5_path=sample_h5_file_mapping,
        scaling_factor=2000.0,
        augmentation_config=test_augmentation_config,
        schedule=ConstantSchedule(type="constant", weight=0.5),
    )

    datamodule = MRIDataModule(
        dataset_configs=[magnitude_dataset_config, mapping_config],
        batch_size=2,
        num_workers=0,
        base_seed=42,
    )
    model = DiffusionLightningModule(**small_model_kwargs)

    datamodule.setup("fit")
    fast_trainer.fit(model, datamodule)

    assert fast_trainer.global_step == 2
    assert datamodule.concat_ds is not None
    assert len(datamodule.dataset_configs) == 2


def test_training_with_yaml_config(
    training_yaml_config: str,
    fast_trainer: pl.Trainer,
) -> None:
    try:
        from score_qmri.training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    original_argv = sys.argv.copy()

    try:
        sys.argv = ["test", "--config", training_yaml_config]

        cli = LightningCLI(
            DiffusionLightningModule,
            run=False,
            parser_kwargs={"parser_mode": "yaml"},
        )

        fast_trainer.fit(cli.model, cli.datamodule)

        assert fast_trainer.global_step == 2
        assert "train/loss" in fast_trainer.logged_metrics

    finally:
        sys.argv = original_argv


def test_cli_parses_default_config() -> None:
    """Test that default config.yaml can be parsed by Lightning CLI."""
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.exists():
        pytest.skip("config.yaml not found")

    original_argv = sys.argv.copy()

    try:
        sys.argv = ["test", "--config", str(config_path), "--job_name", "test-run"]

        from unittest.mock import patch

        with patch("score_qmri.cli.WandbLogger"):
            cli = DiffusionLightningCLI(
                DiffusionLightningModule,
                MRIDataModule,
                run=False,
                subclass_mode_data=True,
            )

        assert isinstance(cli.model, DiffusionLightningModule)
        assert cli.model.model.train_clip == 1.2
        assert cli.model.model.sample_clip == 1.0
        assert cli.datamodule is not None
        assert cli.datamodule.dataset_configs

    finally:
        sys.argv = original_argv
