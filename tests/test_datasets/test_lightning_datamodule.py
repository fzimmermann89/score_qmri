"""Tests for MRIDataModule."""

import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import (
    Collate,
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
    """Test setup creates ConcatDataset."""
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
    assert dm.dataset_lengths is not None
    assert len(dm.dataset_lengths) == 1


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

    assert dm.hparams["batch_size"] == 8
    assert dm.hparams["num_workers"] == 2
    assert dm.hparams["base_seed"] == 123


def test_datamodule_empty_configs():
    """Test MRIDataModule raises error when no configs provided."""
    import pytest

    dm = MRIDataModule(dataset_configs=[])
    with pytest.raises(ValueError, match="No datasets configured"):
        dm.setup("fit")


def test_datamodule_get_config_summary():
    """Test get_config_summary returns flattened config dict."""
    from score_qmri.datasets.config import AugmentationConfig

    configs: list[DatasetConfig] = [
        MagnitudeDatasetConfig(
            h5_path="dummy.h5",
            kind=DatasetKind.T1W_MAGNITUDE,
            schedule=ConstantSchedule(weight=1.0),
            augmentation_config=AugmentationConfig(geometric_prob=0.5),
        )
    ]

    dm = MRIDataModule(dataset_configs=configs, batch_size=4, num_workers=2)
    summary = dm.get_config_summary()

    assert summary["data/batch_size"] == 4
    assert summary["data/num_workers"] == 2
    assert summary["data/num_datasets"] == 1
    assert summary["data/dataset_kinds"] == ["T1W_MAGNITUDE"]
    assert summary["data/dataset_0/kind"] == "T1W_MAGNITUDE"
    assert summary["data/dataset_0/aug/geometric_prob"] == 0.5


def test_collate_background_padding_constant_true():
    """Background padding should be constant True after Collate."""
    background_mask = torch.ones(10, 10, dtype=torch.bool)
    background_mask[3:7, 3:7] = False

    sample = {
        "image": torch.zeros(10, 10),
        "penumbra_mask": torch.zeros(10, 10, dtype=torch.bool),
        "background_mask": background_mask,
        "kind": torch.tensor(DatasetKind.T1W_MAGNITUDE.value, dtype=torch.long),
        "h5_path": "dummy.h5",
    }

    collate = Collate(14, 14)
    batch = collate([sample])
    padded_background = batch["background_mask"][0]

    assert padded_background[0, 0].item()
    assert not padded_background[5:9, 5:9].any().item()
