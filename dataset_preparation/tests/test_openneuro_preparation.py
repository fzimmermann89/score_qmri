"""Precision, orientation and non-destructive CLI regression checks."""

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

from dataset_preparation.ds003967 import prepare, volume
from dataset_preparation.ds004889 import centered
from dataset_preparation.mrf_zenodo8183344 import prepare as prepare_mrf
from dataset_preparation.mrf_zenodo8183344 import zenodo_subject


@pytest.mark.parametrize("dataset", ["ds000221", "ds004611"])
def test_limit_selects_subject_with_requested_contrast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: str) -> None:
    import importlib

    script = importlib.import_module("dataset_preparation." + dataset)
    missing = tmp_path / "raw" / "sub-01"
    present = tmp_path / "raw" / "sub-02"
    if dataset == "ds004611":
        missing, present = missing / "anat", present / "anat"
    missing.mkdir(parents=True)
    present.mkdir(parents=True)
    for filename in ("scan_T1map.nii.gz", "scan_inv-1.nii.gz", "scan_inv-2.nii.gz"):
        (missing / filename).touch()
        (present / filename).touch()
    (present / "scan_T2map.nii.gz").touch()
    (present / "scan_T2w.nii.gz").touch()
    seen = []

    def image(root: Path, modalities: set[str]) -> list[tuple[str, str, torch.Tensor]]:
        seen.append(root)
        return [(next(iter(modalities)), "sub-02", torch.ones(2, 4, 4))]

    monkeypatch.setattr(script, "subject_221" if dataset == "ds000221" else "subject_4611", image)
    modality = "t2w" if dataset == "ds000221" else "t2map"
    output = tmp_path / "output"
    script.prepare(tmp_path / "raw", output, modalities=[modality], limit=1)
    assert seen == [present]
    with h5py.File(output / f"{dataset}_{modality}.h5") as handle:
        assert list(handle) == ["sub-02"]


def test_t1_map_units_orientation_interpolation(tmp_path: Path) -> None:
    data: np.ndarray = np.arange(20 * 15 * 10, dtype=np.float64).reshape(20, 15, 10)
    path = tmp_path / "map.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    actual = volume(path).numpy()
    # Interior interpolation samples use align_corners=False's half-voxel rule.
    positions = [(np.arange(n) + 0.5) / 0.6 - 0.5 for n in (9, 12, 6)]
    expected = positions[0][:, None, None] * 10 + positions[1][None, :, None] * 150 + positions[2][None, None, :]
    np.testing.assert_array_equal(actual, (expected / 1000).astype(np.float32))


def test_output_exclusive_creation_preserves_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    path = source / "map.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((20, 15, 10)), np.eye(4)), path)
    output = tmp_path / "new"
    prepare(source, output, limit=1)
    target = output / "ds003967_t1map_cropped.h5"
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source, output, limit=1)
    assert target.read_bytes() == before
    with h5py.File(target) as file:
        assert file["map.nii"].dtype == np.float32


def test_zenodo_maps_seconds_and_distinct_crops(tmp_path: Path) -> None:
    root = tmp_path / "V01" / "MR6"
    root.mkdir(parents=True)
    shape = (64, 64, 20)
    for name, value in [("mprage", 100), ("t1", 1500), ("t2", 80), ("tse", 100)]:
        nib.save(nib.Nifti1Image(np.full(shape, value, dtype=np.float64), np.eye(4)), root / f"original_{name}.nii")
    outputs = {kind: image for kind, _, image in zenodo_subject(root, "original", {"t1map", "t2map", "t1w", "t2w"})}
    assert outputs["t1map"].shape[0] == 8
    assert outputs["t1w"].shape[0] == 12
    assert outputs["t1map"].max() == 1.5
    assert outputs["t2map"].max() == pytest.approx(0.08)


def test_mrpro_odd_negative_crop_convention() -> None:
    source = torch.arange(5 * 7).reshape(5, 7)
    # mrpro keeps one extra pixel on the leading side when cropping odd deltas.
    actual = centered(source, (2, 4))
    assert torch.equal(actual, source[1:3, 1:5])


def test_zenodo_curated_default_drops_empty_map_slice(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "V01" / "MR6"
    root.mkdir(parents=True)
    for acquisition in ("original", "reposition"):
        contrast = np.full((64, 64, 20), 100.0)
        t1 = np.full(contrast.shape, 1500.0)
        t1[:, :, 9] = 0
        for tag, array in (("mprage", contrast), ("t1", t1)):
            nib.save(nib.Nifti1Image(array, np.eye(4)), root / f"{acquisition}_{tag}.nii")
    output = tmp_path / "curated"
    prepare_mrf(tmp_path / "raw", output, modalities=["t1map"])
    with h5py.File(output / "mrf_zenodo8183344_t1map.h5") as handle:
        for name in ("V01_MR6_original", "V01_MR6_reposition"):
            assert handle[name].shape[0] == 7
            assert handle[name].dtype == np.float32
            assert handle[name][:].max() == 1.5
