"""Independent controls for the discrete target and prior-conditional variance."""

import itertools
from dataclasses import replace

import pytest
import torch
from mrpro.data import KTrajectory, SpatialDimension
from mrpro.operators import FastFourierOp, SensitivityOp
from mrpro.operators.CartesianSamplingOp import CartesianMaskingOp, CartesianSamplingOp
from mrpro.operators.models.EPG import EPGSequence, FispBlock, InversionBlock, Parameters
from mrpro.operators.models.SaturationRecovery import SaturationRecovery

from score_qmri.datasets.kind import DatasetKind
from score_qmri.sampling.split_gibbs_sampler import (
    CartesianAcquisition,
    Coupling,
    DiffusionPriorDraw,
    PriorDraw,
    SignalDictionary,
    SplitGibbsConfig,
    SplitGibbsSampler,
    _SplitGibbsProblem,
)

KINDS = (DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.M0_COMPLEX)
MASK = torch.tensor([[[True, False]], [[False, True]], [[True, False]]])
COILS = torch.tensor([[[1.0, 0.8 + 0.3j]], [[0.2 + 0.4j, -0.3 + 0.6j]]], dtype=torch.complex128)


def dictionary() -> SignalDictionary:
    return SignalDictionary(
        torch.tensor([[0.7, 0.06], [1.3, 0.14]], dtype=torch.float64),
        torch.tensor([[0.8 + 0.2j, 0.3 - 0.5j, -0.2 + 0.4j], [0.2 - 0.4j, 0.9 + 0.1j, 0.4 + 0.3j]], dtype=torch.complex128),
        torch.tensor([0.3, 0.7], dtype=torch.float64),
    )


def test_epg_sequence_dictionary_preserves_amplitude_and_phase() -> None:
    parameters = dictionary().parameters
    sequence = EPGSequence([InversionBlock(0.02), FispBlock(torch.tensor([0.2, 0.6, 0.4]), torch.tensor([0.0, 0.4, 1.2]), 0.003, 0.012)])

    def signal_model(m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor]:
        _, signals = sequence(Parameters(m0=m0, t1=t1, t2=t2), 12)
        return (torch.stack(signals, 0),)

    result = SignalDictionary.from_model(signal_model, parameters, dictionary().weights)
    amplitude = torch.tensor([0.3 + 0.8j, -0.4 + 0.2j], dtype=torch.complex128)
    _, signals = sequence(Parameters(m0=amplitude, t1=parameters[:, 0], t2=parameters[:, 1]), 12)
    torch.testing.assert_close(torch.stack(signals, -1), amplitude[:, None] * result.fingerprints)
    assert result.fingerprints.shape == (2, 3)
    assert result.fingerprints.imag.abs().max() > 0


def make_sampler(batch: int = 1, prior: PriorDraw | None = None, chunk: int = 1, support: torch.Tensor | None = None) -> _SplitGibbsProblem:
    measurement = torch.zeros(batch, 3, 2, 1, 1, 2, dtype=torch.complex128)
    measurement[:, :, :, 0, 0, :] = torch.tensor(
        [[[0.25 + 0.1j, 0], [0.1 - 0.2j, 0]], [[0, 0.2 + 0.4j], [0, -0.1 + 0.1j]], [[-0.1 + 0.2j, 0], [0.3 + 0.2j, 0]]]
    )
    operator = CartesianAcquisition(
        CartesianMaskingOp(MASK[None, :, None, None]), FastFourierOp(dim=(-2, -1)), SensitivityOp(COILS[:, None])
    )
    return _SplitGibbsProblem(
        operator,
        measurement,
        dictionary(),
        torch.ones(1, 2, dtype=torch.bool) if support is None else support,
        torch.zeros(batch, 1, 2, dtype=torch.complex128),
        0.45,
        SplitGibbsConfig((0.4, 0.2, 0.5), dictionary_chunk=chunk),
        prior or (lambda x, rho, g: x),
    )


def run(
    sampler: _SplitGibbsProblem,
    initial_indices: torch.Tensor,
    initial_m0: torch.Tensor,
    *,
    num_samples: int,
    burn_in: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    cfg = replace(sampler.cfg, num_samples=num_samples, burn_in=burn_in)
    (draws,) = SplitGibbsSampler(None, cfg).sample(
        sampler.operator,
        sampler.measurement,
        noise_std=sampler.noise_std,
        template=sampler.template,
        dictionary=sampler.dictionary,
        support=sampler.support,
        initial_indices=initial_indices,
        initial_m0=initial_m0,
        prior_draw=sampler.prior_draw,
        generator=generator,
    )
    return draws


def test_imputation_preserves_observations_and_real_noise_variance() -> None:
    sampler = make_sampler(batch=16000)
    indices = torch.zeros(16000, 1, 2, dtype=torch.long)
    amplitude = torch.full((16000, 1, 2), 0.3 + 0.2j, dtype=torch.complex128)
    generator = torch.Generator().manual_seed(4)
    completed = sampler.impute(indices, amplitude, generator)
    sampled = MASK[:, None, None].expand(3, 2, 1, 1, 2)
    torch.testing.assert_close(completed[:, sampled], sampler.measurement[:, sampled], rtol=0, atol=0)
    signal = sampler.dictionary.fingerprints[indices].moveaxis(-1, 1) * amplitude[:, None]
    (prediction,) = sampler.operator.unitary(sampler.operator.sensitivity(signal[:, :, None, None])[0])
    noise = (completed - prediction)[:, ~sampled]
    assert abs(noise.real.mean()) < 0.006
    assert abs(noise.imag.mean()) < 0.006
    torch.testing.assert_close(noise.real.var(), torch.tensor(0.45**2, dtype=torch.float64), atol=0.004, rtol=0)
    torch.testing.assert_close(noise.imag.var(), torch.tensor(0.45**2, dtype=torch.float64), atol=0.004, rtol=0)


def test_collapse_against_real_gaussian_evidence() -> None:
    sampler = make_sampler()
    images = sampler.operator.unitary.H(sampler.measurement)[0]
    auxiliary = torch.tensor([[[[1.1, 0.8]], [[0.09, 0.12]], [[0.4, -0.2]], [[-0.3, 0.5]]]], dtype=torch.float64)
    logits, means, variances = sampler.collapsed_parameters(sampler.measurement, auxiliary, 0, 2)
    expected = torch.empty_like(logits)
    for voxel in range(2):
        z = images[0, :, :, 0, 0, voxel].flatten()
        real_z = torch.cat((z.real, z.imag))
        mu = auxiliary[0, 2:, 0, voxel]
        for j in range(2):
            atom = (sampler.dictionary.fingerprints[j, :, None] * COILS[:, 0, voxel]).flatten()
            design = torch.cat((torch.stack((atom.real, -atom.imag), -1), torch.stack((atom.imag, atom.real), -1)))
            covariance = sampler.noise_std**2 * torch.eye(len(real_z)) + sampler.cfg.coupling[2] ** 2 * design @ design.T
            evidence = torch.distributions.MultivariateNormal(design @ mu, covariance).log_prob(real_z)
            penalty = (
                (sampler.dictionary.parameters[j] - auxiliary[0, :2, 0, voxel]) / torch.tensor(sampler.cfg.coupling[:2])
            ).square().sum() / 2
            expected[0, 0, voxel, j] = evidence + sampler.dictionary.weights[j].log() - penalty
            precision = design.T @ design / sampler.noise_std**2 + torch.eye(2) / sampler.cfg.coupling[2] ** 2
            conditional_mean = torch.linalg.solve(precision, design.T @ real_z / sampler.noise_std**2 + mu / sampler.cfg.coupling[2] ** 2)
            torch.testing.assert_close(means[0, 0, voxel, j], torch.complex(*conditional_mean))
            torch.testing.assert_close(variances[0, 0, voxel, j], torch.linalg.inv(precision)[0, 0])
    torch.testing.assert_close(logits.softmax(-1), expected.softmax(-1))


@pytest.mark.parametrize("use_sde", [False, True], ids=["exact-prior", "reverse-sde-prior"])
def test_full_chain_matches_enumerated_discrete_smoothed_target(use_sde: bool) -> None:
    """Two voxels coupled by undersampled multicoil data; no continuous reference."""
    mu = torch.tensor([1.0, 0.1, 0.25, -0.15], dtype=torch.float64)
    prior_variance = torch.tensor([0.16, 0.025, 0.12, 0.12], dtype=torch.float64)

    def exact_prior(observation: torch.Tensor, coupling: Coupling, generator: torch.Generator | None) -> torch.Tensor:
        r2 = observation.new_tensor([*coupling[:2], coupling[2], coupling[2]]).square()
        variance = 1 / (1 / prior_variance + 1 / r2)
        mean = variance[None, :, None, None] * (observation / r2[None, :, None, None] + (mu / prior_variance)[None, :, None, None])
        return mean + variance.sqrt()[None, :, None, None] * torch.randn(observation.shape, dtype=observation.dtype, generator=generator)

    class OracleScore(GaussianModel):
        def normalize(self, x: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
            return x

        def denoise(self, x: torch.Tensor, sigma: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
            index = torch.where(kind == DatasetKind.T1MAP, 0, torch.where(kind == DatasetKind.T2MAP, 1, 2))
            means = torch.stack((torch.stack((mu[0], mu[0] * 0)), torch.stack((mu[1], mu[1] * 0)), mu[2:])).to(x)
            variances = prior_variance[:3].to(x)[index, None, None, None]
            s2 = sigma[:, None, None, None].square()
            return (variances * x + s2 * means[index, :, None, None]) / (variances + s2)

    batch = 4096
    prior = DiffusionPriorDraw(OracleScore(), KINDS, (1.0, 1.0, 1.0), steps=128) if use_sde else exact_prior
    sampler = make_sampler(batch, prior)
    configurations = torch.tensor(list(itertools.product(range(2), repeat=2)))
    mask = MASK[:, None, None].expand(3, 2, 1, 1, 2)
    y = sampler.measurement[0][mask]
    m_mean = torch.complex(mu[2], mu[3]).expand(2)
    amplitude_variance = 2 * (prior_variance[2] + sampler.cfg.coupling[2] ** 2)
    log_weights, conditional_means, second_moments = [], [], []
    for config in configurations:
        # Build the observed linear amplitude operator independently, one basis vector at a time.
        columns = []
        for voxel in range(2):
            signal = sampler.dictionary.fingerprints[config].T[:, None, :] * torch.eye(2, dtype=torch.float64)[voxel]
            (kspace,) = FastFourierOp(dim=(-2, -1))(signal[:, None, None] * COILS[:, None])
            columns.append(kspace[mask])
        operator = torch.stack(columns, -1)
        cov = 2 * sampler.noise_std**2 * torch.eye(len(y)) + amplitude_variance * operator @ operator.H
        residual = y - operator @ m_mean
        log_evidence = -(residual.conj() @ torch.linalg.solve(cov, residual)).real - torch.linalg.slogdet(cov)[1]
        t = sampler.dictionary.parameters[config]
        t_variance = prior_variance[:2] + torch.tensor(sampler.cfg.coupling[:2]).square()
        log_prior = -((t - mu[:2]).square() / (2 * t_variance)).sum() + sampler.dictionary.weights[config].log().sum()
        log_weights.append(log_evidence + log_prior)
        mean = m_mean + amplitude_variance * operator.H @ torch.linalg.solve(cov, residual)
        covariance = amplitude_variance * torch.eye(2) - amplitude_variance**2 * operator.H @ torch.linalg.solve(cov, operator)
        conditional_means.append(mean)
        second_moments.append(covariance + mean[:, None] * mean.conj()[None, :])
    probabilities = torch.stack(log_weights).softmax(0)
    expected_mean = (probabilities[:, None] * torch.stack(conditional_means)).sum(0)
    expected_second = (probabilities[:, None, None] * torch.stack(second_moments)).sum(0)
    samples = run(
        sampler,
        torch.zeros(batch, 1, 2, dtype=torch.long),
        torch.zeros(batch, 1, 2, dtype=torch.complex128),
        num_samples=1,
        burn_in=160,
        generator=torch.Generator().manual_seed(71),
    )[0]
    code = (samples[:, 0, 0] > 1).long() @ torch.tensor([2, 1])
    observed = torch.bincount(code, minlength=4).double() / batch
    torch.testing.assert_close(observed, probabilities, atol=0.025, rtol=0)
    amplitude = torch.complex(samples[:, 2, 0], samples[:, 3, 0])
    torch.testing.assert_close(amplitude.mean(0), expected_mean, atol=0.04, rtol=0)
    torch.testing.assert_close(amplitude.T @ amplitude.conj() / batch, expected_second, atol=0.045, rtol=0)


class GaussianModel(torch.nn.Module):
    sigma_min = 0.001
    sigma_max = 20.0

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()

    def normalize(self, x: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        scales = torch.where(kind == DatasetKind.T1MAP, 0.25, torch.where(kind == DatasetKind.T2MAP, 2.0, 1.0))
        return x * scales[:, None, None, None]

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        assert x.dtype == sigma.dtype == self.anchor.dtype
        result = (0.09 * x + sigma[:, None, None, None].square() * 0.2) / (0.09 + sigma[:, None, None, None].square())
        result[kind != DatasetKind.M0_COMPLEX, 1] = 0
        return result


def test_sde_conditional_moments_scaling_and_reproducibility() -> None:
    model = GaussianModel()
    prior = DiffusionPriorDraw(model, KINDS, (2.0, 0.5, 0.4), steps=256)
    obs = torch.full((12000, 4, 1, 1), 0.6, dtype=torch.float64)
    rho = (0.8, 0.4, 1.0)
    scale = torch.tensor([0.5, 1.0, 0.4, 0.4], dtype=torch.float64)
    variance = 0.09 / scale.square()
    r2 = torch.tensor([rho[0], rho[1], rho[2], rho[2]]).square()
    expected_variance = variance * r2 / (variance + r2)
    expected_mean = (variance * 0.6 + r2 * 0.2 / scale) / (variance + r2)
    draws = prior(obs, rho, torch.Generator().manual_seed(83))[:, :, 0, 0]
    torch.testing.assert_close(draws.mean(0), expected_mean, atol=0.014, rtol=0)
    torch.testing.assert_close(draws.var(0), expected_variance, atol=0.018, rtol=0.04)
    small = obs[:2]
    torch.testing.assert_close(prior(small, rho, torch.Generator().manual_seed(1)), prior(small, rho, torch.Generator().manual_seed(1)))
    # At a substantial endpoint the exact reverse process retains extra variance.
    endpoint = DiffusionPriorDraw(model, KINDS, (2.0, 0.5, 0.4), steps=256, sigma_min=0.2)
    noisy = endpoint(obs, rho, torch.Generator().manual_seed(83))[:, :, 0, 0]
    start2 = (scale * torch.tensor([rho[0], rho[1], rho[2], rho[2]])).square()
    endpoint_variance = (0.09 + 0.2**2) * (start2 - 0.2**2) / (0.09 + start2) / scale.square()
    endpoint_mean = ((0.09 + 0.2**2) * scale * 0.6 + (start2 - 0.2**2) * 0.2) / (0.09 + start2) / scale
    torch.testing.assert_close(noisy.mean(0), endpoint_mean, atol=0.014, rtol=0)
    torch.testing.assert_close(noisy.var(0), endpoint_variance, atol=0.018, rtol=0.04)
    with pytest.raises(ValueError, match="endpoint"):
        prior(small, (0.0001, 0.4, 1.0), torch.Generator())


class MixtureModel(GaussianModel):
    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        assert x.dtype == sigma.dtype == self.anchor.dtype
        centers = x.new_tensor([-0.8, 0.8])
        variance = 0.05 + sigma[:, None, None, None, None].square()
        logits = x.new_tensor([0.45, 0.55]).log() - (x[..., None] - centers).square() / (2 * variance)
        component_means = (0.05 * x[..., None] + sigma[:, None, None, None, None].square() * centers) / variance
        result = (logits.softmax(-1) * component_means).sum(-1)
        result[kind != DatasetKind.M0_COMPLEX, 1] = 0
        return result


def test_sde_preserves_mixture_modes_and_converges_with_step_refinement() -> None:
    model = MixtureModel()
    observation = torch.tensor([0.12, -0.1, 0.2, -0.2], dtype=torch.float64)
    coupling = (0.9, 0.6, 0.7)
    r2 = torch.tensor([0.9, 0.6, 0.7, 0.7], dtype=torch.float64).square()
    centers = torch.tensor([-0.8, 0.8], dtype=torch.float64)
    weights = (torch.tensor([0.45, 0.55]).log() - (observation[:, None] - centers).square() / (2 * (0.05 + r2[:, None]))).softmax(-1)
    means = (0.05 * observation[:, None] + r2[:, None] * centers) / (0.05 + r2[:, None])
    variance = 0.05 * r2 / (0.05 + r2)
    expected_mean = (weights * means).sum(-1)
    expected_second = (weights * (means.square() + variance[:, None])).sum(-1)
    expected_positive = (weights * torch.distributions.Normal(means, variance.sqrt()[:, None]).cdf(torch.zeros_like(means))).sum(-1)
    expected_positive = 1 - expected_positive
    obs = observation[None, :, None, None].expand(12000, -1, -1, -1)
    errors = []
    for steps in (8, 512):
        prior = DiffusionPriorDraw(model, KINDS, (4.0, 0.5, 1.0), steps=steps)
        draws = prior(obs, coupling, torch.Generator().manual_seed(19))[:, :, 0, 0]
        errors.append((draws.square().mean(0) - expected_second).abs().max())
        if steps == 512:
            torch.testing.assert_close(draws.mean(0), expected_mean, atol=0.024, rtol=0)
            torch.testing.assert_close(draws.square().mean(0), expected_second, atol=0.02, rtol=0)
            torch.testing.assert_close((draws > 0).double().mean(0), expected_positive, atol=0.018, rtol=0)
    assert errors[1] < errors[0] / 2


def test_background_physical_only_and_input_unchanged() -> None:
    observed_aux = []

    def prior(x: torch.Tensor, rho: Coupling, g: torch.Generator | None) -> torch.Tensor:
        observed_aux.append(x.clone())
        return torch.ones_like(x)

    sampler = make_sampler(prior=prior, support=torch.tensor([[True, False]]))
    indices = torch.zeros(1, 1, 2, dtype=torch.long)
    amplitude = torch.ones(1, 1, 2, dtype=torch.complex128)
    before = sampler.measurement.clone()
    draws = run(sampler, indices, amplitude, num_samples=2, burn_in=2, generator=torch.Generator().manual_seed(2))
    assert not draws[..., 1].any()
    assert not observed_aux[0][..., 1].any()
    torch.testing.assert_close(amplitude, torch.ones_like(amplitude))
    torch.testing.assert_close(sampler.measurement, before)
    with pytest.raises(ValueError, match="indices"):
        run(sampler, indices + 10, amplitude, num_samples=1, burn_in=0, generator=torch.Generator())


def test_native_signal_model_and_compressed_cartesian_acquisition() -> None:
    model = SaturationRecovery(torch.tensor([0.1, 0.5, 1.5], dtype=torch.float64))
    grid = torch.tensor([[0.7], [1.3]], dtype=torch.float64)
    dictionary = SignalDictionary.from_model(model, grid, torch.tensor([0.3, 0.7], dtype=torch.float64))
    amplitude = torch.tensor([0.4 + 0.2j, 0.2 - 0.3j], dtype=torch.complex128)
    torch.testing.assert_close(model(amplitude, grid[:, 0])[0].T, amplitude[:, None] * dictionary.fingerprints)
    trajectory = KTrajectory(
        kz=torch.zeros(1, 1, 1, 1, 1),
        ky=torch.tensor([-2.0, 0.0]).reshape(1, 1, 1, 2, 1),
        kx=torch.arange(-2.0, 2.0).reshape(1, 1, 1, 1, 4),
    )
    selection = CartesianSamplingOp(SpatialDimension(1, 4, 4), trajectory)
    coils = torch.polar(torch.ones(2, 1, 4, 4, dtype=torch.float64), torch.arange(32, dtype=torch.float64).reshape(2, 1, 4, 4) / 7)
    operator = CartesianAcquisition(selection, FastFourierOp(dim=(-2, -1)), SensitivityOp(coils))
    signal = torch.ones(1, 3, 1, 1, 4, 4, dtype=torch.complex128) * (0.3 + 0.4j)
    power = operator.validate(signal)
    torch.testing.assert_close(power, torch.full((1, 4, 4, 3), 2.0, dtype=torch.float64))
    (data,) = operator(signal)
    assert data.shape == (1, 3, 2, 1, 2, 4)
    full = operator.complete(signal.expand(5000, -1, -1, -1, -1, -1), data, 0.2, torch.Generator().manual_seed(41))
    torch.testing.assert_close(selection(full)[0], data.expand(5000, -1, -1, -1, -1, -1), rtol=0, atol=0)
    prediction = operator.unitary(operator.sensitivity(signal)[0])[0]
    noise = (full - prediction)[..., 1::2, :].flatten()
    torch.testing.assert_close(noise.real.var(), torch.tensor(0.04, dtype=torch.float64), atol=0.001, rtol=0)
    torch.testing.assert_close(noise.imag.var(), torch.tensor(0.04, dtype=torch.float64), atol=0.001, rtol=0)
    # General one-coordinate dictionary reaches the same public sampler boundary.
    (draws,) = SplitGibbsSampler(GaussianModel(), SplitGibbsConfig((0.3, 0.4), num_samples=2, burn_in=1, prior_steps=4)).sample(
        operator,
        data,
        noise_std=0.2,
        template=torch.zeros(1, 4, 4, dtype=torch.complex128),
        dictionary=dictionary,
        support=torch.ones(4, 4, dtype=torch.bool),
        kind=(DatasetKind.T1MAP, DatasetKind.M0_COMPLEX),
        dataset_scales=(1.0, 1.0),
        generator=torch.Generator().manual_seed(7),
    )
    assert draws.shape == (2, 1, 3, 4, 4)
    assert torch.isfinite(draws).all()


@pytest.mark.parametrize("bad_factor", ["sampling", "unitary", "sensitivity"])
def test_rejects_factors_that_break_exact_completion(bad_factor: str) -> None:
    template = torch.ones(1, 3, 1, 1, 2, 4, dtype=torch.complex128)
    sampling = CartesianMaskingOp(torch.ones(2, 4))
    unitary = FastFourierOp(dim=(-2, -1))
    sensitivity = SensitivityOp(torch.ones(2, 1, 2, 4, dtype=torch.complex128))
    if bad_factor == "sampling":
        sampling = CartesianMaskingOp(torch.full((2, 4), 0.5))
    elif bad_factor == "unitary":
        unitary = unitary * 2
    else:
        weights = torch.arange(1.0, 9.0, dtype=torch.float64).reshape(1, 1, 2, 4).to(torch.complex128)
        sensitivity = SensitivityOp(weights) @ unitary
    with pytest.raises(ValueError, match="Exact Gibbs"):
        CartesianAcquisition(sampling, unitary, sensitivity).validate(template)


def test_dictionary_allows_general_coordinates_and_prior_kind_order() -> None:
    grid = torch.tensor([[0.7, -0.2, 0.01], [1.3, 0.4, 0.03]], dtype=torch.float64)

    def signal(m0: torch.Tensor, t1: torch.Tensor, offset: torch.Tensor, rate: torch.Tensor) -> tuple[torch.Tensor]:
        return (m0[None] * torch.exp(torch.complex(-t1[None], offset[None])) * (1 + rate[None]),)

    result = SignalDictionary.from_model(signal, grid, torch.ones(2, dtype=torch.float64))
    assert result.fingerprints.shape == (2, 1)
    kinds = (DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.T2STARMAP, DatasetKind.M0_COMPLEX)
    prior = DiffusionPriorDraw(GaussianModel(), kinds, (1.0, 1.0, 1.0, 1.0), steps=4)
    obs = torch.zeros(2, 5, 2, 2, dtype=torch.float64)
    draws = prior(obs, (0.4, 0.2, 0.3, 0.5), torch.Generator().manual_seed(10))
    assert draws.shape == obs.shape and torch.isfinite(draws).all()
    with pytest.raises(ValueError, match="kinds"):
        DiffusionPriorDraw(GaussianModel(), kinds[::-1], (1.0, 1.0, 1.0, 1.0))
