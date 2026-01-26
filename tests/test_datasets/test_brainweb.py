"""Tests for BrainwebQuantitativeSampler."""

from pathlib import Path

import h5py
import numpy as np
import torch

from score_qmri.datasets.brainweb import BrainwebQuantitativeSampler, BrainwebTissue
from score_qmri.datasets.kind import DatasetKind


def _write_brainweb_h5(path: Path, classes: np.ndarray, classnames: list[str]) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("classes", data=classes, dtype=np.uint8)
        f.attrs["classnames"] = np.array(classnames, dtype="S")
        f.attrs["subject"] = 1
        f.attrs["version"] = 1


def _fixed_parameters(classnames: list[str]) -> dict[str, BrainwebTissue]:
    return {name: BrainwebTissue(t1=2.0, t2=0.5, m0_abs=1.0, m0_phase=0.0) for name in classnames}


def _make_classes(shape: tuple[int, int, int, int]) -> np.ndarray:
    classes = np.zeros(shape, dtype=np.uint8)
    classes[0, :, :, 0] = 255
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
    )
    t1_sample = t1_ds[0]
    assert t1_sample["image"].dtype.is_floating_point
    assert torch.allclose(t1_sample["image"].mean(), torch.tensor(1.0))
    assert t1_sample["kind"].item() == DatasetKind.T1MAP.value

    t2_ds = BrainwebQuantitativeSampler(
        str(h5_path),
        kind=DatasetKind.T2MAP,
        parameters=params,
        scaling_factor=2.0,
        skip_slices=(0, 0),
    )
    t2_sample = t2_ds[0]
    assert torch.allclose(t2_sample["image"].mean(), torch.tensor(1.0))
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
    )
    sample = dataset[0]
    assert sample["image"].is_complex()
    assert torch.allclose(sample["image"].abs().mean(), torch.tensor(1.0))
    assert sample["kind"].item() == DatasetKind.M0_COMPLEX.value
