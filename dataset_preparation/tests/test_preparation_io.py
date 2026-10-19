"""Safety and numerical equivalence of standalone dataset preparation I/O."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset_preparation.clean_h5 import clean_file
from dataset_preparation.common import exclusive_h5


def test_exclusive_output_refuses_files_and_dangling_symlinks(tmp_path: Path) -> None:
    existing = tmp_path / "existing.h5"
    existing.write_bytes(b"preserve me")
    dangling = tmp_path / "dangling.h5"
    dangling.symlink_to(tmp_path / "missing.h5")
    for output in (existing, dangling):
        with pytest.raises(FileExistsError), exclusive_h5(output):
            pass
    assert existing.read_bytes() == b"preserve me"
    assert not (tmp_path / "missing.h5").exists()


def test_cleanup_matches_legacy_boundaries_and_drops_attributes(tmp_path: Path) -> None:
    source, target = tmp_path / "source.h5", tmp_path / "target.h5"
    data: np.ndarray = np.zeros((5, 40, 40), dtype=np.float32)
    data[1].flat[:999] = 1
    data[2].flat[:1000] = -1
    data[3] = 1e-6
    data[4].flat[:1000] = 1e-5
    with exclusive_h5(source) as handle:
        handle.attrs["original"] = 10
        ds = handle.create_dataset("subject", data=data)
        ds.attrs["scaling_factor"] = 1000
        handle.create_dataset("empty", data=np.zeros_like(data))
    original_bytes = source.read_bytes()
    report = clean_file(source, target)
    assert report["subject"] == {"input_slices": 5, "output_slices": 2}
    with h5py.File(target, "r") as handle:
        assert list(handle) == ["subject"]
        np.testing.assert_array_equal(handle["subject"][:], data[[2, 4]])
        assert not list(handle.attrs)
        assert not list(handle["subject"].attrs)
        assert handle["subject"].chunks == (1, 40, 40)
        assert handle["subject"].compression == "lzf"
        assert handle["subject"].shuffle
    assert source.read_bytes() == original_bytes


def test_grouped_cleanup_uses_combined_indices_for_all_coils(tmp_path: Path) -> None:
    source, target = tmp_path / "source.h5", tmp_path / "target.h5"
    combined: np.ndarray = np.zeros((3, 40, 40), dtype=np.complex64)
    combined[1].flat[:1000] = 1j
    coilwise: np.ndarray = np.ones((3, 2, 40, 40), dtype=np.complex64)
    with exclusive_h5(source) as handle:
        handle.create_dataset("combined/subj", data=combined)
        handle.create_dataset("coilwise/subj", data=coilwise)
    clean_file(source, target)
    with h5py.File(target, "r") as handle:
        np.testing.assert_array_equal(handle["combined/subj"][:], combined[[1]])
        np.testing.assert_array_equal(handle["coilwise/subj"][:], coilwise[[1]])
        assert handle["coilwise/subj"].chunks == (1, 1, 40, 40)


def test_coilwise_only_filter_counts_largest_coil_not_sum(tmp_path: Path) -> None:
    source, target = tmp_path / "source.h5", tmp_path / "target.h5"
    data: np.ndarray = np.zeros((2, 2, 40, 40), dtype=np.complex64)
    data[0, :, :20] = 1
    data[1, 1].flat[:1000] = 1
    with exclusive_h5(source) as handle:
        handle.create_dataset("coilwise/subj", data=data)
    clean_file(source, target)
    with h5py.File(target, "r") as handle:
        np.testing.assert_array_equal(handle["coilwise/subj"][:], data[[1]])
