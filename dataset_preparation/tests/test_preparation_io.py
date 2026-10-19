"""Safety and numerical equivalence of standalone dataset preparation I/O."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset_preparation.clean_h5 import clean_file
from dataset_preparation.common import exclusive_h5, selected_paths, write_dataset
from dataset_preparation.verify_h5 import compare_h5


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


def test_full_verifier_detects_unsampled_change_and_separates_attrs(tmp_path: Path) -> None:
    left, right = tmp_path / "left.h5", tmp_path / "right.h5"
    data: np.ndarray = np.zeros((5, 40, 40), dtype=np.float32)
    with exclusive_h5(left) as handle:
        write_dataset(handle, "subject", data)
    data[1, 0, 0] = 99
    with exclusive_h5(right) as handle:
        ds = write_dataset(handle, "subject", data)
        ds.attrs["note"] = "different"
    sampled = compare_h5(left, right, sample_slices=2)
    assert sampled["equal"] and sampled["mode"] == "sampled"
    assert not compare_h5(left, right)["equal"]
    assert any(item["reason"] == "attributes" for item in compare_h5(left, right, check_attributes=True)["issues"])


def test_selection_is_sorted_bounded_and_exact(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("s02.h5", "s01.h5", "s03.h5")]
    assert selected_paths(paths, ["s01", "s02.h5"], limit=1) == [tmp_path / "s01.h5"]
    with pytest.raises(ValueError):
        selected_paths(paths, limit=0)


def test_verifier_reports_missing_subject_and_layout_only_difference(tmp_path: Path) -> None:
    left, right = tmp_path / "left.h5", tmp_path / "right.h5"
    data: np.ndarray = np.zeros((2, 4, 4), dtype=np.float32)
    with exclusive_h5(left) as handle:
        write_dataset(handle, "subject", data)
    with exclusive_h5(right) as handle:
        handle.create_dataset("subject", data=data)
    assert compare_h5(left, right)["equal"]
    assert not compare_h5(left, right, check_layout=True)["equal"]
    assert not compare_h5(left, right, subjects=["absent"])["equal"]


def test_verifier_tolerance_is_explicit_and_still_reports_bit_difference(tmp_path: Path) -> None:
    left, right = tmp_path / "left.h5", tmp_path / "right.h5"
    for path, value in ((left, 1.0), (right, 1.0 + 1e-7)):
        with exclusive_h5(path) as handle:
            handle.create_dataset("subject", data=np.full((3, 4, 4), value, np.float64))
    assert not compare_h5(left, right)["equal"]
    report = compare_h5(left, right, atol=2e-7)
    assert report["equal"] and not report["bit_exact"]
    assert report["checked_slices"] == 3
    assert report["checked_values"] == 48
    assert report["max_abs_difference"] == pytest.approx(1e-7)
    assert not compare_h5(left, right, dataset_paths=["missing"])["equal"]


def test_verifier_distinguishes_signed_zero_bytes_from_numerical_equality(tmp_path: Path) -> None:
    left, right = tmp_path / "left.h5", tmp_path / "right.h5"
    for path, sign in ((left, 1.0), (right, -1.0)):
        with exclusive_h5(path) as handle:
            handle.create_dataset("subject", data=np.zeros((2, 3, 3), np.float32) * sign)
    report = compare_h5(left, right)
    assert report["equal"]
    assert not report["bit_exact"]
    assert report["max_abs_difference"] == 0.0
