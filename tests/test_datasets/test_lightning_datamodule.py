"""Tests for MRIDataModule."""

from datasets.lightning import MRIDataModule


def test_datamodule_init_stores_config():
    """Test MRIDataModule stores configuration."""
    configs = [{"type": "magnitude", "h5_path": "dummy.h5", "kind": 9}]

    dm = MRIDataModule(
        dataset_configs=configs,
        schedule_ramp_steps=[10000],
        batch_size=4,
        num_workers=0,
    )
    assert dm.batch_size == 4
    assert dm.num_workers == 0


def test_datamodule_setup_creates_concat_dataset(sample_h5_file_magnitude):
    """Test setup creates ConcatDataset and size labels."""
    configs = [{"type": "magnitude", "h5_path": sample_h5_file_magnitude, "kind": 9}]

    dm = MRIDataModule(
        dataset_configs=configs,
        schedule_ramp_steps=[10000],
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")

    assert dm.concat_ds is not None
    assert dm.size_labels is not None
    assert len(dm.size_labels) == 1


def test_datamodule_hparams_saved():
    """Test hyperparameters are saved correctly."""
    configs = [{"type": "magnitude", "h5_path": "dummy.h5", "kind": 9}]

    dm = MRIDataModule(
        dataset_configs=configs,
        schedule_ramp_steps=[5000],
        batch_size=8,
        num_workers=2,
        base_seed=123,
    )

    assert dm.hparams.batch_size == 8
    assert dm.hparams.num_workers == 2
    assert dm.hparams.base_seed == 123
    assert dm.hparams.schedule_ramp_steps == [5000]


def test_datamodule_empty_configs():
    """Test MRIDataModule handles empty configs."""
    dm = MRIDataModule(dataset_configs=[])
    dm.setup("fit")
    assert dm.concat_ds is None
