"""Tests for BrainwebQuantitativeSampler."""

from pathlib import Path

import h5py
import numpy as np
import torch

from score_qmri.datasets.brainweb import BrainwebQualitativeSampler, BrainwebQuantitativeSampler, BrainwebTissue
from score_qmri.datasets.kind import DatasetKind


def _write_brainweb_h5_group(h5_file: h5py.File, group_name: str, classes: np.ndarray, classnames: list[str], subject: int) -> None:
    """Write subject data to a group within an HDF5 file."""
    group = h5_file.create_group(group_name)
    group.create_dataset("classes", data=classes, dtype=np.uint8)
    group.attrs["classnames"] = np.array(classnames, dtype="S")
    group.attrs["subject"] = subject
    group.attrs["version"] = 1


def _write_brainweb_h5(path: Path, classes: np.ndarray, classnames: list[str], subject: int = 1) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["version"] = 1
        f.attrs["num_subjects"] = 1
        _write_brainweb_h5_group(f, f"subject_{subject:02d}", classes, classnames, subject=subject)


def _fixed_parameters(classnames: list[str]) -> dict[str, BrainwebTissue]:
    params = {}
    for i, name in enumerate(classnames):
        params[name] = BrainwebTissue(t1=1.0 + i * 0.5, t2=0.5 + i * 0.1, m0_abs=1.0, m0_phase=0.0)
    return params


def _value(value: float | tuple[float, float]) -> float:
    if isinstance(value, tuple):
        return sum(value) / 2.0
    return float(value)


def _make_classes(shape: tuple[int, int, int, int], class_index: int = 0) -> np.ndarray:
    classes = np.zeros(shape, dtype=np.uint8)
    classes[0, :, :, class_index] = 255
    return classes


def test_brainweb_t1_t2_maps(tmp_path: Path) -> None:
    classnames = ["skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves"]
    classes = _make_classes((2, 4, 5, len(classnames)))
    h5_path = tmp_path / "brainweb.h5"
    _write_brainweb_h5(h5_path, classes, classnames)
    params = _fixed_parameters(classnames)

    t1_ds = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.T1MAP,
        parameters=params,
        scaling_factor=0.5,
        skip_slices=(0, 0),
        subject_id=1,
    )
    t1_sample = t1_ds[0]
    assert t1_sample["image"].dtype.is_floating_point
    expected_t1 = _value(params[classnames[0]].t1) * 0.5
    assert torch.allclose(t1_sample["image"].mean(), torch.tensor(expected_t1))
    assert t1_sample["kind"].item() == DatasetKind.T1MAP.value

    t2_ds = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.T2MAP,
        parameters=params,
        scaling_factor=2.0,
        skip_slices=(0, 0),
        subject_id=1,
    )
    t2_sample = t2_ds[0]
    expected_t2 = _value(params[classnames[0]].t2) * 2.0
    assert torch.allclose(t2_sample["image"].mean(), torch.tensor(expected_t2))
    assert t2_sample["kind"].item() == DatasetKind.T2MAP.value

    background_sample = t1_ds[1]
    assert torch.all(background_sample["image"] == 0)
    assert torch.all(background_sample["background_mask"])


def test_brainweb_m0_complex(tmp_path: Path) -> None:
    classnames = ["skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves"]
    classes = _make_classes((2, 4, 5, len(classnames)))
    h5_path = tmp_path / "brainweb.h5"
    _write_brainweb_h5(h5_path, classes, classnames)
    params = _fixed_parameters(classnames)

    dataset = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.M0_COMPLEX,
        parameters=params,
        skip_slices=(0, 0),
        subject_id=1,
    )
    sample = dataset[0]
    assert sample["image"].is_complex()
    assert torch.allclose(sample["image"].abs().mean(), torch.tensor(1.0))
    assert sample["kind"].item() == DatasetKind.M0_COMPLEX.value


def test_brainweb_qualitative_t1w_t2w(tmp_path: Path) -> None:
    classnames = ["skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves"]
    classes = _make_classes((2, 4, 5, len(classnames)), class_index=0)
    h5_path = tmp_path / "brainweb_multi.h5"
    _write_brainweb_h5(h5_path, classes, classnames, subject=1)
    params = _fixed_parameters(classnames)

    tr = 0.010
    te = 0.003
    flip = 20.0

    t1 = _value(params[classnames[0]].t1)
    t2 = _value(params[classnames[0]].t2)
    m0 = _value(params[classnames[0]].m0_abs)

    e1 = np.exp(-tr / t1)
    expected_t1w = m0 * np.sin(np.deg2rad(flip)) * (1.0 - e1) / (1.0 - np.cos(np.deg2rad(flip)) * e1)
    expected_t1w *= np.exp(-te / t2)

    t1_ds = BrainwebQualitativeSampler(
        str(h5_path),
        kind=DatasetKind.T1W_MAGNITUDE,
        parameters=params,
        subject_id=1,
        skip_slices=(0, 0),
        tr_range=(tr, tr),
        te_range=(te, te),
        flip_range=(flip, flip),
    )
    t1_sample = t1_ds[0]
    assert t1_sample["kind"].item() == DatasetKind.T1W_MAGNITUDE.value
    expected_t1w_tensor = torch.tensor(expected_t1w, dtype=t1_sample["image"].dtype)
    assert torch.allclose(t1_sample["image"].mean(), expected_t1w_tensor)

    tr_t2w = 3.0
    te_t2w = 0.090
    expected_t2w = m0 * (1.0 - np.exp(-tr_t2w / t1)) * np.exp(-te_t2w / t2)

    t2_ds = BrainwebQualitativeSampler(
        str(h5_path),
        kind=DatasetKind.T2W_MAGNITUDE,
        parameters=params,
        subject_id=1,
        skip_slices=(0, 0),
        tr_range=(tr_t2w, tr_t2w),
        te_range=(te_t2w, te_t2w),
    )
    t2_sample = t2_ds[0]
    assert t2_sample["kind"].item() == DatasetKind.T2W_MAGNITUDE.value
    expected_t2w_tensor = torch.tensor(expected_t2w, dtype=t2_sample["image"].dtype)
    assert torch.allclose(t2_sample["image"].mean(), expected_t2w_tensor)


def test_brainweb_multi_subject_format(tmp_path: Path) -> None:
    """Test reading from multi-subject HDF5 file with groups."""
    classnames = ["skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves"]
    params = _fixed_parameters(classnames)

    # Create multi-subject file
    h5_path = tmp_path / "brainweb_multi.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["version"] = 1
        f.attrs["num_subjects"] = 2

        classes1 = _make_classes((2, 4, 5, len(classnames)), class_index=1)
        _write_brainweb_h5_group(f, "subject_01", classes1, classnames, subject=1)

        classes2 = _make_classes((2, 4, 5, len(classnames)), class_index=2)
        _write_brainweb_h5_group(f, "subject_02", classes2, classnames, subject=2)

    ds1 = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.T1MAP,
        parameters=params,
        subject_id=1,
        scaling_factor=0.5,
        skip_slices=(0, 0),
    )
    sample1 = ds1[0]
    assert sample1["kind"].item() == DatasetKind.T1MAP.value
    expected_t1_subject1 = _value(params[classnames[1]].t1) * 0.5
    assert torch.allclose(sample1["image"].mean(), torch.tensor(expected_t1_subject1))

    ds2 = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.T1MAP,
        parameters=params,
        subject_id=2,
        scaling_factor=0.5,
        skip_slices=(0, 0),
    )
    sample2 = ds2[0]
    assert sample2["kind"].item() == DatasetKind.T1MAP.value
    expected_t1_subject2 = _value(params[classnames[2]].t1) * 0.5
    assert torch.allclose(sample2["image"].mean(), torch.tensor(expected_t1_subject2))

    # Verify they come from different subjects
    assert not torch.allclose(sample1["image"], sample2["image"])

    ds_all = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.T1MAP,
        parameters=params,
        scaling_factor=0.5,
        skip_slices=(0, 0),
    )
    assert len(ds_all) == len(ds1) + len(ds2)
    assert torch.allclose(ds_all[0]["image"], sample1["image"])
    assert torch.allclose(ds_all[len(ds1)]["image"], sample2["image"])
