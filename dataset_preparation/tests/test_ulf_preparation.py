"""Recovered ULF 3T reference processing and output safety."""

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

from dataset_preparation.ulf import clean_mask_2d, prepare, prepare_volume


def test_processing_matches_original_cell() -> None:
    volume: np.ndarray = np.zeros((64, 64, 4), dtype=np.float64)
    volume[3:-3, 3:-3] = 1.0
    volume[20:40, 20:40] = 2.0
    expected = np.transpose(volume, (2, 1, 0))
    expected = np.flip(expected, axis=1)
    expected = np.pad(expected, ((0, 0), (16, 16), (16, 16)))
    expected /= np.percentile(expected[expected > 0.05 * expected.max()], 99.5)
    mask = torch.stack([clean_mask_2d(m) for m in expected > 0.05])
    selection = mask.float().mean((-1, -2)) > 0.3
    mask = mask[selection]
    expected = expected[selection]
    expected[~mask] = 0
    np.testing.assert_array_equal(prepare_volume(volume), expected.astype(np.float32))


def test_one_nifti_and_no_overwrite(tmp_path: Path) -> None:
    folder = tmp_path / "raw" / "POCEMR001" / "3T"
    folder.mkdir(parents=True)
    volume: np.ndarray = np.zeros((64, 64, 4), dtype=np.float32)
    volume[3:-3, 3:-3] = 1.0
    nifti = folder / "POCEMR001_T1.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), nifti)
    before = nifti.read_bytes()
    output = tmp_path / "output"
    prepare(tmp_path / "raw", output, modalities=["t1"], limit=1)
    h5 = output / "ulf_t1.h5"
    with h5py.File(str(h5), "r") as file:
        assert list(file) == ["POCEMR001"]
        np.testing.assert_array_equal(file["POCEMR001"][:], prepare_volume(volume.astype(np.float64)))
    written = h5.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "raw", output, modalities=["t1"])
    assert h5.read_bytes() == written
    assert nifti.read_bytes() == before
