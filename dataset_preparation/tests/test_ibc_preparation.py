"""Check fitted physical parameters and historical IBC masks/crops."""

import h5py
import numpy as np
import pytest
import torch

pytest.importorskip("nilearn")

from dataset_preparation.ibc_relaxo import fit_t1_linear, fit_t2, fit_t2star
from dataset_preparation.normalize_h5 import normalize_file, pad_or_crop_centered


def test_vfa_fit_recovers_known_t1_with_b1_correction():
    angles = np.array([5, 9, 13, 17], dtype=np.float32)
    b1 = np.full((4, 8, 8), 0.8, dtype=np.float32)
    actual_angles = b1[..., None] * np.deg2rad(angles)
    e1 = np.float32(np.exp(-0.01 / 0.8))
    signal = 4000 * np.sin(actual_angles) * (1 - e1) / (1 - e1 * np.cos(actual_angles))
    mask = np.ones(b1.shape, dtype=bool)
    mask[0] = False
    t1, m0 = fit_t1_linear(signal, angles, b1, mask, 0.01)
    np.testing.assert_allclose(t1[mask], 0.8, atol=2e-5)
    np.testing.assert_allclose(m0[mask], 4000, atol=0.2)
    assert not t1[0].any()


def test_echo_fits_recover_known_decay_and_normalized_signal():
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


def test_shape_normalization_uses_original_odd_crop_and_refuses_overwrite(tmp_path):
    source, output = tmp_path / "source.h5", tmp_path / "out.h5"
    data = np.arange(2 * 7 * 9, dtype=np.float32).reshape(2, 7, 9)
    with h5py.File(source, "x") as handle:
        handle.create_dataset("subject", data=data)
    normalize_file(source, output, (4, 12))
    expected = torch.nn.functional.pad(torch.from_numpy(data), (1, 2, -2, -1)).numpy()
    np.testing.assert_array_equal(pad_or_crop_centered(data, 4, 12), expected)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["subject"][:], expected)
    with pytest.raises(FileExistsError):
        normalize_file(source, output, (4, 12))


def test_historical_normalization_skips_bad_ibc_subject(tmp_path):
    source, output = tmp_path / "ibc_relaxo_s0map.h5", tmp_path / "out.h5"
    with h5py.File(source, "x") as handle:
        handle.create_dataset("sub-12_ses-17", data=np.ones((2, 8, 8), np.float32))
        handle.create_dataset("sub-14_ses-20", data=np.ones((2, 8, 8), np.float32))
    normalize_file(source, output)
    with h5py.File(output, "r") as handle:
        assert list(handle) == ["sub-14_ses-20"]
        assert handle["sub-14_ses-20"].shape == (2, 256, 160)
