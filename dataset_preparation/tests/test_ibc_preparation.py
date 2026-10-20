"""Check fitted physical parameters and historical IBC masks/crops."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("nilearn")

from dataset_preparation import ibc_t2star
from dataset_preparation.ibc_t1 import fit_t1_linear
from dataset_preparation.ibc_t2 import fit_t2
from dataset_preparation.ibc_t2star import fit_t2star


def test_vfa_fit_recovers_known_t1_with_b1_correction() -> None:
    angles = np.array([5, 9, 13, 17], dtype=np.float32)
    b1: np.ndarray = np.full((4, 8, 8), 0.8, dtype=np.float32)
    actual_angles = b1[..., None] * np.deg2rad(angles)
    e1 = np.float32(np.exp(-0.01 / 0.8))
    signal = 4000 * np.sin(actual_angles) * (1 - e1) / (1 - e1 * np.cos(actual_angles))
    mask: np.ndarray = np.ones(b1.shape, dtype=bool)
    mask[0] = False
    t1, m0 = fit_t1_linear(signal, angles, b1, mask, 0.01)
    np.testing.assert_allclose(t1[mask], 0.8, atol=2e-5)
    np.testing.assert_allclose(m0[mask], 4000, atol=0.2)
    assert not t1[0].any()


def test_echo_fits_recover_known_decay_and_normalized_signal() -> None:
    te = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    magnitude = np.broadcast_to(1000 * np.exp(-te / 0.07), (70, 16, 16, 4)).copy()
    mask = np.ones(magnitude.shape[:-1], dtype=bool)
    t2, s0 = fit_t2(magnitude, te, mask)
    star, s0_star = fit_t2star(magnitude, te, mask)
    np.testing.assert_allclose(t2[t2 != 0], 0.07, atol=1e-6)
    np.testing.assert_allclose(star[star != 0], 0.07, atol=1e-6)
    np.testing.assert_allclose(s0[s0 != 0], 1, atol=1e-6)
    np.testing.assert_allclose(s0_star[s0_star != 0], 1, atol=1e-6)
    assert star.shape == (30, 16, 16)


def test_t2star_preparation_skips_excluded_subject_before_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for subject in ("12", "13", "14"):
        (tmp_path / f"sub-{subject}_ses-17_part-mag_T2starmap.nii.gz").touch()
    arrays = (torch.full((2, 256, 160), 0.07), torch.ones(2, 256, 160))
    monkeypatch.setattr(ibc_t2star, "compute_t2star_map", lambda *_: arrays)
    output = tmp_path / "output"
    ibc_t2star.prepare(tmp_path, output, limit=1)
    for kind, expected in zip(("t2starmap", "s0map"), arrays, strict=True):
        with h5py.File(str(output / f"ibc_relaxo_{kind}.h5"), "r") as handle:
            assert list(handle) == ["sub-13_ses-17"]
            np.testing.assert_array_equal(handle["sub-13_ses-17"][:], expected)
    with pytest.raises(FileExistsError):
        ibc_t2star.prepare(tmp_path, output, limit=1)
