"""Fixed sample-number holdouts, including aliases of a source slice."""

from dataclasses import replace

import pytest
import torch
from torch.utils.data import Subset

from score_qmri.datasets.lightning import Collate, ComplexDatasetConfig, MagnitudeDatasetConfig, MRIDataModule


def test_fixed_samples_are_withheld_and_loaded_without_augmentation(sample_h5_file_magnitude):
    config = MagnitudeDatasetConfig(h5_path=sample_h5_file_magnitude, validation_samples=[1, 6])
    module = MRIDataModule(dataset_configs=[config], sample_size=(64, 64), num_workers=0)
    module.setup("fit")
    assert module.dataset_lengths == [6]
    training = module.concat_ds.datasets[0]
    assert isinstance(training, Subset)
    assert training.indices == [0, 2, 3, 4, 5, 7]
    fixed = replace(config, augmentation_config=None).create()
    expected = Collate(64, 64)([fixed[1], fixed[6]])
    assert module.validation_batch is not None
    assert torch.equal(module.validation_batch["image"], expected["image"])
    first = module.validation_batch["image"].clone()
    module.setup("fit")
    assert module.validation_batch is not None
    assert torch.equal(first, module.validation_batch["image"])
    assert config.augmentation_config is not None


def test_combined_holdout_is_removed_from_all_coils_of_the_same_slice(sample_h5_file_complex):
    coilwise = ComplexDatasetConfig(h5_path=sample_h5_file_complex, mode="coilwise")
    combined = ComplexDatasetConfig(h5_path=sample_h5_file_complex, mode="combined", validation_samples=[1])
    module = MRIDataModule(dataset_configs=[coilwise, combined], sample_size=(64, 64))
    module.setup("fit")
    assert module.dataset_lengths == [16, 4]
    for training in module.concat_ds.datasets:
        assert all(training.dataset._slice_index[index][:2] != ("subject_001", 1) for index in training.indices)


@pytest.mark.parametrize("indices", [[-1], [1, 1]])
def test_invalid_sample_numbers_fail_early(indices):
    with pytest.raises(ValueError, match="validation_samples"):
        MagnitudeDatasetConfig(h5_path="unused.h5", validation_samples=indices)


def test_out_of_bounds_and_empty_training_selection_are_rejected(sample_h5_file_magnitude):
    for indices in ([8], list(range(8))):
        config = MagnitudeDatasetConfig(h5_path=sample_h5_file_magnitude, validation_samples=indices)
        with pytest.raises(ValueError, match="validation_samples"):
            MRIDataModule(dataset_configs=[config]).setup("fit")


def test_no_selection_preserves_original_training_dataset(sample_h5_file_magnitude):
    module = MRIDataModule(dataset_configs=[MagnitudeDatasetConfig(h5_path=sample_h5_file_magnitude)])
    module.setup("fit")
    assert module.validation_batch is None
    assert module.dataset_lengths == [8]
    assert not isinstance(module.concat_ds.datasets[0], Subset)


def test_randomized_brainweb_cannot_claim_fixed_validation():
    from score_qmri.datasets.lightning import BrainwebDatasetConfig

    with pytest.raises(ValueError, match="randomized Brainweb"):
        BrainwebDatasetConfig(h5_path="unused.h5", validation_samples=[0])
