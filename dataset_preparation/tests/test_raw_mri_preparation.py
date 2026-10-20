"""Small CPU regressions; no real raw reconstruction or CUDA workload."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from dataset_preparation import cc, fastmri, m4raw


def test_original_centered_fft_convention() -> None:
    torch.manual_seed(0)
    raw = torch.randn(2, 3, 1, 8, 10, dtype=torch.complex64)
    expected = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(raw.numpy(), axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
    np.testing.assert_allclose(fastmri.ift(raw).numpy(), expected, atol=1e-7)


def test_padding_preserves_original_odd_alignment() -> None:
    raw = torch.arange(5)
    # mrpro builds before=2/after=1 then reverses the flattened pad list.
    assert fastmri.pad_or_crop(raw, (8,)).tolist() == [0, 0, 1, 2, 3, 4, 0, 0]
    assert fastmri.pad_or_crop(raw, (2,)).tolist() == [1, 2]


def test_calibration_and_pocs_run_on_cpu() -> None:
    torch.manual_seed(1)
    raw = torch.randn(1, 2, 1, 6, 6, dtype=torch.complex64)
    csm = fastmri.espirit_2d(raw, calib_width=4, kernel_width=2, max_iter=3)
    assert csm.shape == raw.shape
    assert csm.device.type == "cpu"
    assert torch.isfinite(csm).all()
    volume = torch.randn(1, 2, 4, 4, 4, dtype=torch.complex64)
    csm3 = cc.espirit_3d(volume, calib_width=4, kernel_width=2, max_iter=3)
    assert csm3.shape == volume.shape
    assert torch.isfinite(csm3).all()
    acquired = torch.full((2, 6, 6, 2), 200 + 100j, dtype=torch.complex64)
    output = cc.pocs_partial_fourier(acquired, n_iter=2)
    torch.testing.assert_close(output, acquired)


def test_cc_raw_output_is_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "raw"
    inputs.mkdir()
    with h5py.File(str(inputs / "subject.h5"), "w-") as source:
        source.create_dataset("kspace", data=np.ones((1, 2), dtype=np.float32))
    image = torch.ones((1, 32, 32), dtype=torch.complex64)
    monkeypatch.setattr(cc, "cc_images", lambda raw: (image, image[:, None]))
    output = tmp_path / "new"
    cc.reconstruct(inputs, output)
    path = output / "cc_reconstructed.h5"
    with h5py.File(str(path), "r") as target:
        np.testing.assert_array_equal(target["combined/subject"][:], image.numpy())
        np.testing.assert_array_equal(target["coilwise/subject"][:], image[:, None].numpy())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        cc.reconstruct(inputs, output)
    assert path.read_bytes() == before


@pytest.mark.parametrize("dataset", ["fastmri", "m4raw"])
def test_raw_limit_applies_to_each_modality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: str) -> None:
    inputs = tmp_path / "raw"
    inputs.mkdir()
    for sequence in ("T1", "T2"):
        for subject in ("001", "002"):
            filename = f"{'x' * 13}{sequence}_{subject}_scan.h5" if dataset == "fastmri" else f"{subject}_{sequence}01.h5"
            with h5py.File(str(inputs / filename), "w-") as source:
                source.create_dataset("kspace", data=np.ones((1, 1, 32, 32), dtype=np.complex64))
                source.create_dataset("ismrmrd_header", data=b"mock header")
    image = torch.ones((1, 32, 32), dtype=torch.complex64)
    module: ModuleType
    if dataset == "fastmri":
        header = SimpleNamespace(encoding=[SimpleNamespace(reconSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=32, y=32)))])
        monkeypatch.setitem(sys.modules, "ismrmrd", SimpleNamespace(xsd=SimpleNamespace(CreateFromDocument=lambda _: header)))
        monkeypatch.setattr(fastmri, "fastmri_images", lambda raw, matrix: (image, image[:, None]))
        module, modalities, subject = fastmri, ("t1", "t2"), "001_scan"
    else:
        monkeypatch.setattr(m4raw, "m4raw_image", lambda raw, gre: image)
        module, modalities, subject = m4raw, ("t1w", "t2w"), "001"
    output = tmp_path / "new"
    module.prepare(inputs, output, limit=1)
    assert sorted(path.name for path in output.iterdir()) == sorted(f"{dataset}_{modality}.h5" for modality in modalities)
    for modality in modalities:
        with h5py.File(str(output / f"{dataset}_{modality}.h5"), "r") as target:
            group = target["combined"] if dataset == "fastmri" else target
            assert list(group) == [subject]
            np.testing.assert_array_equal(group[subject][:], image.numpy())


def test_no_inputs_no_outputs_and_existing_destination_refused(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="No matching"):
        m4raw.prepare(inputs, output)
    assert not output.exists()
    with h5py.File(str(inputs / "subject_T101.h5"), "w-") as file:
        file.create_dataset("kspace", data=np.ones((2, 1, 4, 4), dtype=np.complex64))
    output.mkdir()
    existing = output / "m4raw_t1w.h5"
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        m4raw.prepare(inputs, output)
    assert existing.read_bytes() == b"preserve"


@pytest.mark.parametrize("dataset", ["fastmri", "m4raw"])
def test_all_raw_destinations_checked_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: str) -> None:
    inputs = tmp_path / "raw"
    inputs.mkdir()
    for sequence in ("T1", "T2"):
        filename = f"{'x' * 13}{sequence}_001_scan.h5" if dataset == "fastmri" else f"001_{sequence}01.h5"
        (inputs / filename).touch()
    output = tmp_path / "new"
    output.mkdir()
    second = "t2" if dataset == "fastmri" else "t2w"
    existing = output / f"{dataset}_{second}.h5"
    existing.write_bytes(b"preserve")
    monkeypatch.setitem(sys.modules, "ismrmrd", SimpleNamespace())
    module: ModuleType = fastmri if dataset == "fastmri" else m4raw
    with pytest.raises(FileExistsError):
        module.prepare(inputs, output)
    assert list(output.iterdir()) == [existing]
    assert existing.read_bytes() == b"preserve"


def test_cli_defaults_do_not_select_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(fastmri, "prepare", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["fastmri.py", "--input-root", str(tmp_path), "--output-dir", str(tmp_path / "new"), "--limit", "1"])
    fastmri.main()
    assert calls[0][5] == "cpu"
    assert torch.get_num_threads() == 1
