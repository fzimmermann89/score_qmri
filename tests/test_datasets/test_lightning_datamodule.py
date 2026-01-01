"""Tests for MRIDataModule."""

from datasets.kind import DatasetKind
from datasets.lightning import (
    ConstantSchedule,
    DatasetConfig,
    MagnitudeDatasetConfig,
    MRIDataModule,
)


def test_datamodule_init_stores_config():
    """Test MRIDataModule stores configuration."""
    configs: list[DatasetConfig] = [
        MagnitudeDatasetConfig(h5_path="dummy.h5", kind=DatasetKind.T1W_MAGNITUDE, schedule=ConstantSchedule(weight=1.0))
    ]

    dm = MRIDataModule(
        dataset_configs=configs,
        batch_size=4,
        num_workers=0,
    )
    assert dm.batch_size == 4
    assert dm.num_workers == 0


def test_datamodule_setup_creates_concat_dataset(sample_h5_file_magnitude):
    """Test setup creates ConcatDataset and size labels."""
    configs: list[DatasetConfig] = [
        MagnitudeDatasetConfig(h5_path=sample_h5_file_magnitude, kind=DatasetKind.T1W_MAGNITUDE, schedule=ConstantSchedule(weight=1.0))
    ]

    dm = MRIDataModule(
        dataset_configs=configs,
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")

    assert dm.concat_ds is not None
    assert dm.size_labels is not None
    assert len(dm.size_labels) == 1


def test_datamodule_hparams_saved():
    """Test hyperparameters are saved correctly."""
    configs: list[DatasetConfig] = [
        MagnitudeDatasetConfig(h5_path="dummy.h5", kind=DatasetKind.T1W_MAGNITUDE, schedule=ConstantSchedule(weight=1.0))
    ]

    dm = MRIDataModule(
        dataset_configs=configs,
        batch_size=8,
        num_workers=2,
        base_seed=123,
    )

    # The hparams are properly typed via the class-level annotation
    assert dm.hparams.batch_size == 8
    assert dm.hparams.num_workers == 2
    assert dm.hparams.base_seed == 123


def test_datamodule_empty_configs():
    """Test MRIDataModule handles empty configs."""
    dm = MRIDataModule(dataset_configs=[])
    dm.setup("fit")
    assert dm.concat_ds is None
