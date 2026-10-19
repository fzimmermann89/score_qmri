"""Distribution and failure checks against analytic linear-Gaussian targets."""

import pytest
import torch

from score_qmri.sampling.gaussian import complex_normal_like, sample_linear_gaussian


class Identity:
    def __call__(self, x):
        return (x,)

    @property
    def H(self):
        return self


class MaskedFourier:
    def __init__(self, mask, adjoint=False):
        self.mask = mask
        self.adjoint = adjoint

    def __call__(self, x):
        if self.adjoint:
            return (torch.fft.ifft2(x * self.mask, norm="ortho"),)
        return (torch.fft.fft2(x, norm="ortho") * self.mask,)

    @property
    def H(self):
        return MaskedFourier(self.mask, not self.adjoint)


def test_identity_posterior_mean_and_real_imag_variance():
    center = torch.zeros(3000, 1, 1, 2, 2, dtype=torch.complex128)
    measurement = torch.ones_like(center)
    draw, info = sample_linear_gaussian(Identity(), measurement, center, 1.0, 1.0, generator=torch.Generator().manual_seed(21), rtol=1e-12)
    assert draw.real.mean().item() == pytest.approx(0.5, abs=0.025)
    assert draw.imag.mean().item() == pytest.approx(0, abs=0.025)
    assert draw.real.var().item() == pytest.approx(0.5, abs=0.03)
    assert draw.imag.var().item() == pytest.approx(0.5, abs=0.03)
    assert info.relative_residual.max() < 1e-12


def test_masked_fourier_observed_and_nullspace_distributions():
    mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.float64)
    center = torch.zeros(3000, 1, 1, 2, 2, dtype=torch.complex128)
    draw, _ = sample_linear_gaussian(
        MaskedFourier(mask),
        torch.ones_like(center) * mask,
        center,
        1.0,
        0.5,
        generator=torch.Generator().manual_seed(12),
        rtol=1e-12,
    )
    freq = torch.fft.fft2(draw, norm="ortho")
    observed, unobserved = freq[..., 0, 0], freq[..., 1, 1]
    assert observed.real.mean().item() == pytest.approx(0.8, abs=0.03)
    assert observed.real.var().item() == pytest.approx(0.2, abs=0.03)
    assert observed.imag.var().item() == pytest.approx(0.2, abs=0.03)
    assert unobserved.real.mean().item() == pytest.approx(0, abs=0.06)
    assert unobserved.real.var().item() == pytest.approx(1, abs=0.07)
    assert unobserved.imag.var().item() == pytest.approx(1, abs=0.07)


def test_seeded_reproducibility_and_single_pixel_layout():
    center = torch.zeros(4, 1, 1, 1, 1, dtype=torch.complex64)
    args = (Identity(), torch.ones_like(center), center, 0.8, 0.3)
    first, _ = sample_linear_gaussian(*args, generator=torch.Generator().manual_seed(9))
    second, _ = sample_linear_gaussian(*args, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    noise = complex_normal_like(torch.zeros(10000, 1, dtype=torch.complex64), torch.Generator().manual_seed(2))
    assert noise.real.var().item() == pytest.approx(1, abs=0.05)
    assert noise.imag.var().item() == pytest.approx(1, abs=0.05)


def test_batch_mean_solve_does_not_mix_examples():
    center = torch.zeros(3, 1, 1, 2, 2, dtype=torch.complex128)
    measurement = torch.arange(3, dtype=torch.float64).to(torch.complex128).reshape(3, 1, 1, 1, 1).expand_as(center)
    draw, _ = sample_linear_gaussian(Identity(), measurement, center, 1, 1, rtol=1e-12, mean_only=True)
    torch.testing.assert_close(draw, measurement / 2)


def test_nonconvergence_is_reported():
    center = torch.zeros(2, 1, 1, 2, 2, dtype=torch.complex128)
    op = MaskedFourier(torch.tensor([[1, 0], [0, 0]]))
    with pytest.raises(RuntimeError, match="CG failed"):
        sample_linear_gaussian(op, torch.ones_like(center), center, 1, 0.1, max_iter=1, rtol=1e-12)


@pytest.mark.parametrize("prior,noise", [(0, 1), (1, 0), (float("nan"), 1), (1, float("inf"))])
def test_invalid_noise_is_rejected(prior, noise):
    x = torch.zeros(1, 1, 1, 2, 2, dtype=torch.complex64)
    with pytest.raises(ValueError, match="finite and positive"):
        sample_linear_gaussian(Identity(), x, x, prior, noise)
