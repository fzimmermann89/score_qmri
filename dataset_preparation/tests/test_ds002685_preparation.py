"""Check the historical subject exclusions and native output widths."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dataset_preparation import ds002685


def test_t2w_preparation_skips_tse_before_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    excluded = "sub-01_ses-00_acq-tse_T2w.nii.gz"
    retained = "sub-02_ses-00_acq-spc_T2w.nii.gz"
    for filename in (excluded, retained):
        (tmp_path / filename).touch()
    data = torch.ones(2, 256, 176)
    monkeypatch.setattr(ds002685, "volume", lambda *_: data)
    output = tmp_path / "output"
    ds002685.prepare(tmp_path, output, ["t2w"], limit=1)
    with h5py.File(str(output / "ds002685_t2w.h5"), "r") as handle:
        assert list(handle) == [retained.removesuffix(".gz")]
        np.testing.assert_array_equal(handle[retained.removesuffix(".gz")][:], data)


@pytest.mark.parametrize("width", [160, 176, 192])
def test_flair_preparation_retains_native_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    filename = "sub-01_ses-00_FLAIR.nii.gz"
    (tmp_path / filename).touch()
    data = torch.ones(2, 256, width)
    monkeypatch.setattr(ds002685, "volume", lambda *_: data)
    output = tmp_path / "output"
    ds002685.prepare(tmp_path, output, ["flair"])
    with h5py.File(str(output / "ds002685_flair.h5"), "r") as handle:
        np.testing.assert_array_equal(handle[filename.removesuffix(".gz")][:], data)
