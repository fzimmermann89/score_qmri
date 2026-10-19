"""Check EDM conversion against VP coefficients and paper DAPS equations."""

from dataclasses import replace
from math import sqrt

import pytest
import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.sampling.ancestral_dps import AncestralDPSConfig, AncestralDPSSampler
from score_qmri.sampling.daps_sampler import to_2ch, to_complex
from score_qmri.sampling.paper_daps import PaperDAPSConfig, PaperDAPSSampler, reference_sigma_grid


class GaussianDenoiser:
    device = torch.device("cpu")
    sigma_min = 0.001
    sigma_max = 20.0
    variance = 0.04

    def denoise(self, x, sigma, kind):
        return x * self.variance / (self.variance + sigma[:, None, None, None] ** 2)

    def score(self, x, sigma, kind):
        return -x / (self.variance + sigma[:, None, None, None] ** 2)

    def clamp_x0(self, x, kind):
        raise AssertionError("No implicit clipping should occur in paper-form modes")


class Identity:
    def __call__(self, x):
        return (x,)

    @property
    def H(self):
        return self


def test_score_prior_conditional_recovers_analytic_gaussian_moments():
    """Check the conditional sampling distribution, beyond reference parity."""
    model = GaussianDenoiser()
    cfg = PaperDAPSConfig(tau=0.2, prior="score", langevin_steps=300)
    sampler = PaperDAPSSampler(model, cfg)
    xt = torch.full((8192, 2, 1, 1), 0.3, dtype=torch.float64)
    center = torch.zeros_like(xt)
    measurement = to_complex(torch.full_like(xt, 0.1))
    sigma, eta = 0.5, 0.001
    draw = sampler._conditional(
        xt,
        center,
        sigma,
        eta,
        torch.ones(8192, dtype=torch.long),
        Identity(),
        measurement,
        torch.Generator().manual_seed(42),
    )
    precision = sigma**-2 + cfg.tau**-2 + 1 / (model.variance + model.sigma_min**2)
    mean = (0.3 / sigma**2 + 0.1 / cfg.tau**2) / precision
    # ULA's stationary variance has a known finite-step correction.
    ula_variance = 1 / (precision * (1 - eta * precision / 2))
    assert abs(float(draw.mean()) - mean) < 4 * (ula_variance / draw.numel()) ** 0.5
    assert abs(float(draw.var()) / ula_variance - 1) < 0.04


@pytest.mark.parametrize("sigma,sigma_next", [(20.0, 10.0), (1.0, 0.5), (0.01, 0.0)])
@pytest.mark.parametrize("conditioning", ["residual_norm", "gaussian_likelihood"])
def test_ancestral_step_matches_independent_vp_ddpm_coefficients(sigma, sigma_next, conditioning):
    generator = torch.Generator().manual_seed(22)
    z = torch.randn(3, 2, 2, 2, dtype=torch.float64, generator=generator)
    epsilon = torch.randn(z.shape, dtype=z.dtype, generator=generator)
    y = torch.randn(z.shape, dtype=z.dtype, generator=generator) * 0.1
    cfg = AncestralDPSConfig(conditioning=conditioning, tau=0.2, guidance_scale=0.03)
    model = GaussianDenoiser()
    sampler = AncestralDPSSampler(model, cfg)
    actual = sampler._step(z, torch.ones(3, dtype=torch.long), Identity(), to_complex(y), sigma, sigma_next, noise=epsilon)
    alpha_bar = 1 / (1 + sigma**2)
    alpha_bar_prev = 1 / (1 + sigma_next**2)
    alpha = alpha_bar / alpha_bar_prev
    beta = 1 - alpha
    clean = model.variance / (model.variance + sigma**2) * z / sqrt(alpha_bar)
    residual = clean - y
    jacobian = model.variance / (model.variance + sigma**2) / sqrt(alpha_bar)
    if conditioning == "residual_norm":
        gradient = jacobian * residual / residual.flatten(1).norm(dim=1)[:, None, None, None]
    else:
        gradient = jacobian * residual / cfg.tau**2
    coefficient1 = beta * sqrt(alpha_bar_prev) / (1 - alpha_bar)
    coefficient2 = sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
    variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    expected = coefficient1 * clean + coefficient2 * z + sqrt(variance) * epsilon - cfg.guidance_scale * gradient
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)


def test_ancestral_zero_residual_is_finite_and_no_noise_at_zero_endpoint():
    model = GaussianDenoiser()
    sampler = AncestralDPSSampler(model)
    z = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    actual = sampler._step(z, torch.ones(1, dtype=torch.long), Identity(), to_complex(z), 0.1, 0.0, noise=torch.ones_like(z))
    torch.testing.assert_close(actual, z)


@pytest.mark.parametrize("family", ["dps", "daps"])
def test_paper_modes_reproduce_seed_but_retain_sampling_noise(family):
    model = GaussianDenoiser()
    noise = torch.full((2, 1, 1, 2, 2), 0.02 + 0.03j, dtype=torch.complex128)
    y = torch.zeros_like(noise)
    sampler = (
        AncestralDPSSampler(model, AncestralDPSConfig(num_steps=8, guidance_scale=0.01))
        if family == "dps"
        else PaperDAPSSampler(model, PaperDAPSConfig(tau=0.2, anneal_steps=4, reverse_steps=2, langevin_steps=5))
    )
    args = (Identity(), y, DatasetKind.COMPLEX_T1W_COIL_COMBINED)
    draw1 = sampler.sample(*args, noise=noise, generator=torch.Generator().manual_seed(5))
    draw2 = sampler.sample(*args, noise=noise, generator=torch.Generator().manual_seed(5))
    draw3 = sampler.sample(*args, noise=noise, generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(draw1, draw2)
    assert not torch.allclose(draw1, draw3)
    assert draw1.shape == noise.shape
    assert draw1.isfinite().all()
    assert sampler.last_sigmas[-1] == 0


@pytest.mark.parametrize("prior", ["gaussian", "score"])
def test_daps_conditional_is_full_temperature_paper_equation(prior):
    cfg = PaperDAPSConfig(tau=0.2, prior=prior, langevin_steps=1)
    model = GaussianDenoiser()
    sampler = PaperDAPSSampler(model, cfg)
    center = torch.full((1, 2, 1, 1), 0.3, dtype=torch.float64)
    xt = torch.full_like(center, 0.7)
    target = torch.tensor([0.1, 0.2], dtype=center.dtype).unflatten(0, (1, 2, 1, 1))
    eta = 1e-4
    actual = sampler._conditional(xt, center, 0.5, eta, torch.tensor([1]), Identity(), to_complex(target), torch.Generator().manual_seed(7))
    prior_score = (
        torch.zeros_like(center) if prior == "gaussian" else (xt - center) / 0.5**2 - center / (model.variance + model.sigma_min**2)
    )
    noise = torch.randn(center.shape, dtype=center.dtype, generator=torch.Generator().manual_seed(7))
    expected = center + eta * (prior_score - (center - target) / cfg.tau**2) + sqrt(2 * eta) * noise
    torch.testing.assert_close(actual, expected)


def test_daps_pf_estimator_stops_at_positive_sigma_without_extra_denoising():
    cfg = PaperDAPSConfig(tau=0.1, reverse_steps=1, sigma_x0=0.01)
    model = GaussianDenoiser()
    sampler = PaperDAPSSampler(model, cfg)
    x = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    sigma = 0.5
    d = model.variance / (model.variance + sigma**2) * x
    expected = x + (cfg.sigma_x0 - sigma) * (x - d) / sigma
    torch.testing.assert_close(sampler._estimate(x, sigma, torch.tensor([1])), expected)


def test_reference_outer_grid_and_lr_use_reference_step_counts():
    grid = reference_sigma_grid(4, 0.1, 20.0, 7.0, torch.device("cpu"))
    expected = torch.tensor([(20 ** (1 / 7) + r * (0.1 ** (1 / 7) - 20 ** (1 / 7))) ** 7 for r in [0.0, 0.25, 0.5, 0.75]] + [0.0])
    torch.testing.assert_close(grid, expected)
    cfg = PaperDAPSConfig(tau=0.1, anneal_steps=4, reverse_steps=1, langevin_steps=1)
    sampler = PaperDAPSSampler(GaussianDenoiser(), cfg)
    sampler.sample(Identity(), torch.zeros(1, 1, 1, 1, 1, dtype=torch.complex128), 1, generator=torch.Generator().manual_seed(8))
    assert sampler.last_step_sizes == pytest.approx([cfg.langevin_lr * (1 + i / 5 * (cfg.lr_min_ratio - 1)) for i in range(4)])


@pytest.mark.parametrize("family", ["dps", "daps"])
def test_sampler_rejects_extrapolating_beyond_checkpoint_sigma_range(family):
    model = GaussianDenoiser()
    y = torch.zeros(1, 1, 1, 1, 1, dtype=torch.complex64)
    sampler = (
        AncestralDPSSampler(model, AncestralDPSConfig(sigma_start=100.0))
        if family == "dps"
        else PaperDAPSSampler(model, PaperDAPSConfig(tau=0.1, sigma_start=100.0))
    )
    with pytest.raises(ValueError, match="sigma"):
        sampler.sample(Identity(), y, 1)


def test_gaussian_conditioning_requires_an_explicit_noise_standard_deviation():
    with pytest.raises(ValueError, match="tau"):
        AncestralDPSConfig(conditioning="gaussian_likelihood")
    with pytest.raises(ValueError):
        replace(PaperDAPSConfig(tau=0.1), tau=float("nan"))


def test_complex_layout_is_unchanged_by_new_modes():
    x = torch.randn(3, 2, 1, 2, dtype=torch.float64)
    torch.testing.assert_close(to_2ch(to_complex(x)), x)
