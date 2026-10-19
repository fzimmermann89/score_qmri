"""Verify uncertainty and error use matching estimators and sample axes."""

import runpy
from pathlib import Path

import numpy as np
import pytest

analysis = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/analyze_posterior_ensembles.py"))
posterior_maps = analysis["posterior_maps"]


def test_uncertainty_distinguishes_phase_spread_from_magnitude_spread():
    samples = np.array([[1 + 0j, 1 + 1j], [-1 + 0j, 3 + 3j]])
    truth = np.array([1 + 0j, 2 + 2j])
    maps = posterior_maps(samples, truth)
    np.testing.assert_allclose(maps["mean_complex"], [0, 2 + 2j])
    np.testing.assert_allclose(maps["mean_magnitude"], [1, 2 * np.sqrt(2)])
    np.testing.assert_allclose(maps["std_magnitude"], [0, 2])
    np.testing.assert_allclose(maps["std_complex"], [np.sqrt(2), 2])
    np.testing.assert_allclose(maps["error_magnitude"], [0, 0], atol=1e-15)
    np.testing.assert_allclose(maps["error_complex"], [1, 0])


@pytest.mark.parametrize(
    "samples,truth", [(np.ones((1, 2)), np.ones(2)), (np.ones((3, 4)), np.ones(2)), (np.full((3, 2), np.nan), np.ones(2))]
)
def test_uncertainty_rejects_insufficient_or_invalid_ensembles(samples, truth):
    with pytest.raises(ValueError):
        posterior_maps(samples, truth)


def test_spatial_diagnostic_keeps_error_and_spread_units_and_excludes_background():
    std = np.arange(1, 18, dtype=float)
    error = 2 * std
    error[-1] = 1e6
    roi = np.ones(17, dtype=bool)
    roi[-1] = False
    result = analysis["diagnostic"](error, std, roi)
    assert result["rmse_over_rms_std"] == pytest.approx(2)
    assert result["spearman_error_vs_std"] == pytest.approx(1)
    assert sum(b["count"] for b in result["uncertainty_bins"]) == 16
    for bin_values in result["uncertainty_bins"]:
        assert bin_values["rmse"] == pytest.approx(2 * bin_values["rms_std"])
