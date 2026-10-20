"""Independent linear-field controls for local score diagnostics."""

import runpy
from pathlib import Path

import pytest
import torch

diagnostics = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/check_denoiser.py"))["score_diagnostics"]


class LinearDenoiser:
    sigma_min = 0.001
    sigma_max = 20.0
    sigma_data = 0.2

    def __init__(self, matrix):
        self.matrix = matrix

    def denoise(self, x, sigma, kind):
        return torch.einsum("ij,bjhw->bihw", self.matrix, x)


def evaluate(matrix, kind=1, roi=None):
    clean = torch.full((1, 2, 1, 2), 0.2, dtype=torch.float64)
    noise = torch.ones_like(clean)
    probe = torch.zeros_like(clean)
    probe[:, 0] = 1
    return diagnostics(LinearDenoiser(matrix), clean, torch.tensor([kind]), 0.1, noise, probe, roi=roi)


def test_exact_gaussian_has_symmetric_positive_denoiser_and_known_mse():
    gain = 0.04 / (0.04 + 0.1**2)
    result = evaluate(torch.eye(2, dtype=torch.float64) * gain)
    assert result["score_jacobian_relative_skew"] == pytest.approx(0, abs=1e-12)
    assert result["denoiser_rayleigh_min"] == pytest.approx(gain)
    expected_mse = (gain * 0.3 - 0.2) ** 2
    assert result["mse_over_noise_variance"] == pytest.approx(expected_mse / 0.1**2)
    assert result["main_edm_mse"] == pytest.approx(result["zero_network_edm_mse"])


def test_rotational_field_is_detected_and_real_kind_excludes_imaginary_block():
    matrix = torch.tensor([[0.8, -0.3], [0.3, 0.8]], dtype=torch.float64)
    assert evaluate(matrix)["score_jacobian_relative_skew"] > 1
    assert evaluate(matrix, kind=6)["score_jacobian_relative_skew"] == pytest.approx(0, abs=1e-12)


def test_negative_jacobian_direction_is_detected():
    result = evaluate(torch.diag(torch.tensor([-0.2, 0.8], dtype=torch.float64)))
    assert result["denoiser_rayleigh_min"] == pytest.approx(-0.2)
