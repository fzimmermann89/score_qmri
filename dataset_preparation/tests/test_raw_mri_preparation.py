"""Small CPU regressions; no real raw reconstruction or CUDA workload."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dataset_preparation import raw_mri


def test_original_centered_fft_convention() -> None:
    torch.manual_seed(0)
    raw = torch.randn(2, 3, 1, 8, 10, dtype=torch.complex64)
    expected = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(raw.numpy(), axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    np.testing.assert_allclose(raw_mri.ift(raw).numpy(), expected, atol=1e-7)


def test_padding_preserves_original_odd_alignment() -> None:
    raw = torch.arange(5)
    # mrpro builds before=2/after=1 then reverses the flattened pad list.
    assert raw_mri.pad_or_crop(raw, (8,)).tolist() == [0, 0, 1, 2, 3, 4, 0, 0]
    assert raw_mri.pad_or_crop(raw, (2,)).tolist() == [1, 2]


def test_calibration_and_pocs_run_on_cpu() -> None:
    torch.manual_seed(1)
    raw = torch.randn(1, 2, 1, 6, 6, dtype=torch.complex64)
    csm = raw_mri.espirit_2d(raw, calib_width=4, kernel_width=2, max_iter=3)
    assert csm.shape == raw.shape
    assert csm.device.type == "cpu"
    assert torch.isfinite(csm).all()
    volume = torch.randn(1, 2, 4, 4, 4, dtype=torch.complex64)
    csm3 = raw_mri.espirit_3d(volume, calib_width=4, kernel_width=2, max_iter=3)
    assert csm3.shape == volume.shape
    assert torch.isfinite(csm3).all()
    acquired = torch.full((2, 6, 6, 2), 200 + 100j, dtype=torch.complex64)
    output = raw_mri.pocs_partial_fourier(acquired, n_iter=2)
    torch.testing.assert_close(output, acquired)


def test_legacy_cc_streaming_matches_original_axis_recipe(tmp_path: Path) -> None:
    source = tmp_path / "legacy.h5"
    values: np.ndarray = np.arange(72 * 4 * 5, dtype=np.float32).reshape(72, 4, 5).astype(np.complex64)
    coils: np.ndarray = np.arange(5 * 2 * 72 * 4, dtype=np.float32).reshape(5, 2, 72, 4).astype(np.complex64)
    with h5py.File(source, "x") as file:
        file.create_group("combined").create_dataset("subject", data=values)
        file.create_group("coilwise").create_dataset("subject", data=coils)
    before = source.read_bytes()
    output = tmp_path / "fixed.h5"
    raw_mri.fix_legacy_cc(source, output)
    with h5py.File(output, "r") as file:
        np.testing.assert_array_equal(file["combined/subject"][:], values[50:-20])
        expected = np.swapaxes(np.swapaxes(coils, 0, 2), -1, -2)[50:-20]
        np.testing.assert_array_equal(file["coilwise/subject"][:], expected)
    assert source.read_bytes() == before
    fixed_before = output.read_bytes()
    with pytest.raises(FileExistsError):
        raw_mri.fix_legacy_cc(source, output)
    assert output.read_bytes() == fixed_before


def test_no_inputs_no_outputs_and_existing_destination_refused(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="No matching"):
        raw_mri.prepare("m4raw", inputs, output)
    assert not output.exists()
    with h5py.File(inputs / "subject_T101.h5", "x") as file:
        file.create_dataset("kspace", data=np.ones((2, 1, 4, 4), dtype=np.complex64))
    output.mkdir()
    existing = output / "m4raw_t1w.h5"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        raw_mri.prepare("m4raw", inputs, output)
    assert existing.read_bytes() == b"preserve"


def test_cli_defaults_do_not_select_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(raw_mri, "prepare", lambda *args: calls.append(args))
    raw_mri.main(["m4raw", "--input-root", str(tmp_path), "--output-dir", str(tmp_path / "new"), "--limit", "1"])
    assert calls[0][6] == "cpu"
    assert torch.get_num_threads() == 1


def test_cc_prepared_family_uses_combined_slice_filter(tmp_path: Path) -> None:
    source = tmp_path / "prepared.h5"
    values: np.ndarray = np.ones((3, 40, 40), dtype=np.complex64)
    values[0] = 0
    coils: np.ndarray = np.stack([values, values * 2], axis=1)
    with h5py.File(source, "x") as file:
        file.create_group("combined").create_dataset("subject", data=values)
        file.create_group("coilwise").create_dataset("subject", data=coils)
    output = tmp_path / "output"
    raw_mri.prepare("cc-prepared", source, output, limit=1)
    with h5py.File(output / "cc.h5", "r") as file:
        np.testing.assert_array_equal(file["combined/subject"][:], values[1:])
        np.testing.assert_array_equal(file["coilwise/subject"][:], coils[1:])
    with pytest.raises(FileExistsError):
        raw_mri.prepare("cc-prepared", source, output, limit=1)


@pytest.mark.parametrize("family,modality", [("m4raw", "gre"), ("fastmri", "t1")])
def test_prepared_imports_retained_arrays(tmp_path: Path, family: str, modality: str) -> None:
    source = tmp_path / "prepared"
    source.mkdir()
    values: np.ndarray = np.ones((3, 40, 40), dtype=np.complex64)
    values[0] = 0
    with h5py.File(source / f"{family}_{modality}.h5", "x") as file:
        file.create_dataset("subject", data=values)
    output = tmp_path / "output"
    raw_mri.prepare(f"{family}-prepared", source, output, modalities=[modality], limit=1)
    with h5py.File(output / f"{family}_{modality}.h5", "r") as file:
        np.testing.assert_array_equal(file["subject"][:], values[1:])
    with pytest.raises(FileExistsError):
        raw_mri.prepare(f"{family}-prepared", source, output, modalities=[modality], limit=1)
