"""Tests for MRIDataModule."""

from datasets.lightning import MRIDataModule


def test_datamodule_init_stores_config():
    """Test MRIDataModule stores configuration."""
    configs = [{"type": "magnitude", "h5_path": "dummy.h5", "dataset_path": "mag", "kind": 1}]
    schedules = [lambda s: 1.0]

    dm = MRIDataModule(configs, schedules, batch_size=4, num_workers=0)
    assert dm.batch_size == 4
    assert dm.num_workers == 0


def test_datamodule_setup_creates_concat_dataset(sample_h5_file):
    """Test setup creates ConcatDataset and size labels."""
    configs = [{"type": "magnitude", "h5_path": sample_h5_file, "dataset_path": "magnitude", "kind": 1}]
    schedules = [lambda s: 1.0]

    dm = MRIDataModule(configs, schedules, batch_size=4, num_workers=0)
    dm.setup("fit")

    assert dm.concat_ds is not None
    assert dm.size_labels is not None
    assert len(dm.size_labels) == 1
