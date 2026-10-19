"""Independent regression checks for Brainweb reproduction and file safety."""

import gzip
from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset_preparation import brainweb


def source_normalize(values: list[np.ndarray]) -> np.ndarray:
    """Literal newer mr2 routine, independently applied to a full fixture."""
    for i, x in enumerate(values):
        values[i] = np.clip(x - np.min(x[50], (0, 1)), 0, 4096)
    sum_values = sum(values)
    values.pop(0)
    for i, x in enumerate(values):
        x = np.divide(x, sum_values, out=np.zeros_like(x, dtype=float), where=sum_values != 0)
        x[sum_values == 0] = 0
        values[i] = (x * 255).astype(np.uint8)
    return np.stack(values, -1)


def test_streaming_matches_original_including_underflow(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    shape = (52, 3, 4)
    values = [rng.integers(0, 4096, shape, dtype=np.uint16) for _ in brainweb.CLASSES]
    for x in values:
        x[50, 0, 0] = 0
    values[-1][50] = 16
    values[-1][0, 0, 0] = 0
    paths = []
    for tissue, array in zip(brainweb.CLASSES, values, strict=True):
        path = tmp_path / f"{tissue}.gz"
        with gzip.open(path, "wb") as file:
            file.write(array.astype("<u2").tobytes())
        paths.append(path)
    expected = source_normalize([x.copy() for x in values])
    actual = np.concatenate([slab for _, slab in brainweb.iter_raw_slabs(paths, shape, slab_size=3)])
    np.testing.assert_array_equal(actual, expected)
    safe = np.concatenate([slab for _, slab in brainweb.iter_raw_slabs(paths, shape, "float-safe")])
    assert np.any(safe[0, 0, 0] != actual[0, 0, 0])
    assert actual[0, 0, 0, -1] > 0
    assert safe[0, 0, 0, -1] == 0


def test_pack_copies_values_and_attributes_and_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brainweb, "SHAPE", (3, 4, 5))
    source = tmp_path / "s04.h5"
    values: np.ndarray = np.arange(3 * 4 * 5 * 11, dtype=np.uint8).reshape(3, 4, 5, 11)
    with h5py.File(source, "x") as file:
        file.attrs["classnames"] = brainweb.CLASSES[1:]
        file.attrs["subject"] = 4
        file.attrs["version"] = 1
        data = file.create_dataset("classes", data=values)
        data.attrs["test"] = "preserve"
    before = source.read_bytes()
    output = tmp_path / "packed.h5"
    brainweb.pack_subjects(tmp_path, output, ["04"], slab_size=2)
    with h5py.File(output, "r") as file:
        group = file["subject_04"]
        np.testing.assert_array_equal(group["classes"][:], values)
        assert tuple(group.attrs["classnames"]) == brainweb.CLASSES[1:]
        assert group.attrs["subject"] == 4
        assert group.attrs["version"] == 1
        assert group["classes"].attrs["test"] == "preserve"
    packed_before = output.read_bytes()
    with pytest.raises(FileExistsError):
        brainweb.pack_subjects(tmp_path, output, ["04"])
    assert output.read_bytes() == packed_before
    assert source.read_bytes() == before


def test_raw_requires_explicit_download_and_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "output.h5"
    with pytest.raises(FileNotFoundError):
        brainweb.prepare_raw(tmp_path, output, ["04"])
    assert not output.exists()
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        brainweb.prepare_raw(tmp_path, output, ["04"], download=True)
    assert output.read_bytes() == b"preserve"
    assert not (tmp_path / "s04").exists()


def test_explicit_little_endian_decoding_and_invalid_input(tmp_path: Path) -> None:
    paths = []
    for tissue in brainweb.CLASSES:
        path = tmp_path / f"{tissue}.gz"
        with gzip.open(path, "wb") as file:
            # BE bytes misread as LE exceed the valid 12-bit range.
            file.write(np.full((51, 1, 1), 4095, dtype=">u2").tobytes())
        paths.append(path)
    with pytest.raises(ValueError, match="4095"):
        list(brainweb.iter_raw_slabs(paths, shape=(51, 1, 1)))


def test_cli_help_and_invalid_pack_options(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        brainweb.main(["--help"])
    assert result.value.code == 0
    assert "float-safe" in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        brainweb.main(["pack", "--input-root", str(tmp_path), "--output", str(tmp_path / "out.h5"), "--download"])
    assert result.value.code == 2
    assert not (tmp_path / "out.h5").exists()
