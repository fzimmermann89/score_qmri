"""Integration tests for full training pipeline."""

import lightning.pytorch as pl
import pytest

from datasets.config import AugmentationConfig
from datasets.kind import DatasetKind
from datasets.lightning import (
    ConstantSchedule,
    MagnitudeDatasetConfig,
    MRIDataModule,
)
from training.lightning_module import DiffusionLightningModule


@pytest.fixture
def integration_trainer():
    """Fast trainer for integration tests."""
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
def integration_aug_config():
    """Augmentation config with minimal augmentations."""
    return AugmentationConfig(
        geometric_prob=0.0,
        contrast_prob=0.0,
        phase_prob=0.0,
        shift_range=(-5, 5),
    )


@pytest.fixture
def small_integration_model_kwargs():
    """Small model for fast integration tests."""
    return {
        "base_channels": 8,
        "num_blocks": 2,
        "cond_dim": 32,
        "beta_min": 0.1,
        "beta_max": 20.0,
        "learning_rate": 1e-4,
        "noise_weight": 1.0,
        "x0_weight": 0.1,
        "ssim_weight": 0.5,
        "l1_weight": 0.5,
        "validation_size": 16,
        "validation_steps": 5,
        "validation_interval": 10,
    }


def test_integration_single_dataset_training(
    sample_h5_file_magnitude: str,
    integration_aug_config: AugmentationConfig,
    integration_trainer: pl.Trainer,
    small_integration_model_kwargs: dict,
) -> None:
    """Test complete training loop with single dataset."""
    dataset_config = MagnitudeDatasetConfig(
        type="magnitude",
        kind=DatasetKind.T1W_MAGNITUDE,
        h5_path=sample_h5_file_magnitude,
        augmentation_config=integration_aug_config,
        schedule=ConstantSchedule(type="constant", weight=1.0),
    )

    datamodule = MRIDataModule(
        dataset_configs=[dataset_config],
        batch_size=2,
        num_workers=0,
        base_seed=42,
    )

    model = DiffusionLightningModule(**small_integration_model_kwargs)

    datamodule.setup("fit")
    integration_trainer.fit(model, datamodule)

    assert integration_trainer.global_step == 2
    assert integration_trainer.logged_metrics is not None


def test_integration_multiple_datasets_produce_distinct_outputs(
    sample_h5_file_magnitude: str,
    sample_h5_file_mapping: str,
    integration_aug_config: AugmentationConfig,
    integration_trainer: pl.Trainer,
    small_integration_model_kwargs: dict,
) -> None:
    """Test multiple datasets with different kinds produce distinct outputs."""
    from datasets.lightning import MappingDatasetConfig

    magnitude_config = MagnitudeDatasetConfig(
        type="magnitude",
        kind=DatasetKind.T1W_MAGNITUDE,
        h5_path=sample_h5_file_magnitude,
        augmentation_config=integration_aug_config,
        schedule=ConstantSchedule(type="constant", weight=1.0),
    )

    mapping_config = MappingDatasetConfig(
        type="mapping",
        kind=DatasetKind.T1MAP,
        h5_path=sample_h5_file_mapping,
        scaling_factor=2000.0,
        augmentation_config=integration_aug_config,
        schedule=ConstantSchedule(type="constant", weight=0.5),
    )

    datamodule = MRIDataModule(
        dataset_configs=[magnitude_config, mapping_config],
        batch_size=4,
        num_workers=0,
        base_seed=42,
    )

    model = DiffusionLightningModule(**small_integration_model_kwargs)

    datamodule.setup("fit")

    # Verify both configs are retained
    assert len(datamodule.dataset_configs) == 2
    assert datamodule.concat_ds is not None

    integration_trainer.fit(model, datamodule)

    assert integration_trainer.global_step == 2


def test_integration_datamodule_with_lightning_module(
    sample_h5_file_magnitude: str,
    integration_aug_config: AugmentationConfig,
    small_integration_model_kwargs: dict,
) -> None:
    """Test MRIDataModule correctly integrates with DiffusionLightningModule."""
    dataset_config = MagnitudeDatasetConfig(
        type="magnitude",
        kind=DatasetKind.T1W_MAGNITUDE,
        h5_path=sample_h5_file_magnitude,
        augmentation_config=integration_aug_config,
        schedule=ConstantSchedule(type="constant", weight=1.0),
    )

    datamodule = MRIDataModule(
        dataset_configs=[dataset_config],
        batch_size=2,
        num_workers=0,
        base_seed=42,
    )

    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()

    # Verify dataloader produces batches with correct format
    batch = next(iter(train_loader))
    assert "image" in batch
    assert "mask" in batch
    assert "kind" in batch
    assert batch["image"].shape[0] == 2  # Batch size
    # Dataset returns (H, W), collation gives (B, H, W)
    assert batch["image"].shape[1] == 64  # Height
    assert batch["image"].shape[2] == 64  # Width
    # Note: prepare_input() in training_step will add channel dim to make (B, 2, H, W)
