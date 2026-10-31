"""Gaussian target controls and nonlinear finite-Langevin checks for PnP-DM."""

from dataclasses import replace

import pytest
import torch
from mrpro.operators import FastFourierOp
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling.pnp_dm_sampler import PnPDMConfig, PnPDMSampler


class GaussianModel(torch.nn.Module):
    """Exact anisotropic Gaussian score in model coordinates."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    sigma_min = 0.001
    sigma_max = 2.0

    def __init__(self, covariance: Tensor | None = None, scale: float = 1.0) -> None:
        super().__init__()
        covariance = torch.tensor([[0.04, 0.018], [0.018, 0.07]], dtype=torch.float64) if covariance is None else covariance
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=covariance.dtype), requires_grad=False)
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.scale = scale
        self.covariance = covariance
        self.calls: list[float] = []

    @property
    def device(self) -> torch.device:
        return self.eigenvalues.device

    def normalize(self, x: Tensor, kind: Tensor) -> Tensor:
        return x * self.scale

    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor:
        return x / self.scale

    def denoise(self, x: Tensor, sigma: Tensor, kind: Tensor) -> Tensor:
        self.calls.append(float(sigma[0]))
        gain = self.eigenvalues / (self.eigenvalues + sigma[:, None, None, None] ** 2)
        return ((x @ self.eigenvectors) * gain) @ self.eigenvectors.mT


class MatrixOperator:
    def __init__(self, matrix: Tensor) -> None:
        self.matrix = matrix

    def __call__(self, x: Tensor) -> tuple[Tensor]:
        return (x @ self.matrix.to(x).mT,)

    @property
    def H(self) -> "MatrixOperator":
        return MatrixOperator(self.matrix.mH)


def assert_moments(draws: Tensor, mean: Tensor, covariance: Tensor, limit: float = 5.0) -> None:
    """Use independent chains, with Monte Carlo standard errors per moment."""
    n = len(draws)
    mean_se = (covariance.diag() / n).sqrt()
    covariance_se = ((covariance.diag()[:, None] * covariance.diag()[None, :] + covariance.square()) / (n - 1)).sqrt()
    assert ((draws.mean(0) - mean).abs() / mean_se).max() < limit
    assert ((torch.cov(draws.mT) - covariance).abs() / covariance_se).max() < limit


@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("rectangular", [False, True])
def test_linear_conditional_is_covariance_correct_complex_gaussian(scale: float, rectangular: bool) -> None:
    model = GaussianModel(scale=scale)
    matrix = torch.tensor([[1.0 + 0.2j, 0.4 - 0.3j], [0.0, 0.05j]], dtype=torch.complex128)
    matrix = matrix[:1] if rectangular else matrix
    operator = MatrixOperator(matrix)
    template = torch.zeros(8192, 1, 2, dtype=torch.complex128)
    y = torch.full((1, 1, len(matrix)), 0.15 + 0.07j, dtype=torch.complex128)
    problem = SamplingProblem(model, operator, y, template, DatasetKind.M0_COMPLEX, 0.08, 0)
    physical = torch.full_like(template, 0.12 - 0.09j)
    x, rho = problem.encode(physical), 0.2
    sampler = PnPDMSampler(model, PnPDMConfig(cg_rtol=1e-11))
    draws = sampler._likelihood(x, problem, rho, torch.Generator().manual_seed(32)).flatten(1)

    b = matrix / scale
    real_matrix = torch.cat([torch.cat([b.real, -b.imag], 1), torch.cat([b.imag, b.real], 1)], 0)
    measured = torch.cat([y.flatten().real, y.flatten().imag])
    precision = real_matrix.mT @ real_matrix / 0.08**2 + torch.eye(4, dtype=torch.float64) / rho**2
    covariance = precision.inverse()
    mean = covariance @ (real_matrix.mT @ measured / 0.08**2 + x[0].flatten() / rho**2)
    assert_moments(draws, mean, covariance)


def test_reverse_sde_draws_denoising_conditional_with_nonzero_variance() -> None:
    model = GaussianModel()
    operator = MatrixOperator(torch.eye(2, dtype=torch.float64))
    template = torch.zeros(8192, 1, 2, dtype=torch.complex128)
    problem = SamplingProblem(model, operator, template, template, DatasetKind.M0_COMPLEX, 0.1, 0)
    observation = problem.encode(torch.full_like(template, 0.25 - 0.14j))
    rho = 0.3
    draws = PnPDMSampler(model, PnPDMConfig(reverse_steps=512))._prior(observation, problem, rho, torch.Generator().manual_seed(47))
    covariance = model.covariance
    gain = torch.linalg.solve(covariance + rho**2 * torch.eye(2), covariance)
    conditional_cov = covariance - gain @ covariance
    for part in range(2):
        assert_moments(draws[:, part, 0], gain @ observation[0, part, 0], conditional_cov)
    assert model.calls[0] == rho and model.calls[-1] == model.sigma_min
    assert all(model.sigma_min <= sigma <= rho for sigma in model.calls)


def test_final_reverse_interval_remains_stochastic() -> None:
    model = GaussianModel(covariance=torch.zeros(2, 2, dtype=torch.float64))
    operator = MatrixOperator(torch.eye(2, dtype=torch.float64))
    template = torch.zeros(8192, 1, 2, dtype=torch.float64)
    problem = SamplingProblem(model, operator, template, template, DatasetKind.T1MAP, 0.1, 0)
    sampler = PnPDMSampler(model, PnPDMConfig(reverse_steps=2))
    draws = sampler._prior(problem.encode(template), problem, 0.2, torch.Generator().manual_seed(9))
    assert_moments(draws[:, 0, 0], torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64) * model.sigma_min**2)
    assert torch.count_nonzero(draws[:, 1]) == 0


def test_fixed_rho_chain_matches_declared_smoothed_likelihood_target() -> None:
    model = GaussianModel(scale=0.25)
    operator = MatrixOperator(torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)))
    template = torch.zeros(4096, 1, 2, dtype=torch.float64)
    y = torch.tensor([[[0.8, 0.0]]], dtype=torch.float64)
    tau, rho = 0.1, 0.22
    cfg = PnPDMConfig(rho=rho, rho_start=0.6, anneal_steps=3, burn_in=35, reverse_steps=192, cg_rtol=1e-11)
    (draws,) = PnPDMSampler(model, cfg).sample(
        operator, y, noise_std=tau, template=template, kind=DatasetKind.T1MAP, generator=torch.Generator().manual_seed(93)
    )
    b = operator.matrix / model.scale
    effective_noise = tau**2 * torch.eye(2) + rho**2 * b @ b.mT
    covariance = (model.covariance.inverse() + b.mT @ torch.linalg.solve(effective_noise, b)).inverse()
    mean = covariance @ b.mT @ torch.linalg.solve(effective_noise, y.flatten())
    assert_moments(draws[0, :, 0] * model.scale, mean, covariance)
    # Finite splitting must be detectable: this is not the unsplit posterior.
    unsplit_cov = (model.covariance.inverse() + b.mT @ b / tau**2).inverse()
    assert covariance[0, 0] > 20 * unsplit_cov[0, 0]
    assert draws.shape == (1, *template.shape) and not draws.requires_grad


def test_reverse_discretization_refines_toward_gaussian_conditional() -> None:
    model = GaussianModel()
    rho, variance = 0.4, 0.04
    exact_mean = variance / (variance + rho**2)
    exact_cov = variance * rho**2 / (variance + rho**2)
    errors = []
    for steps in (16, 64, 256):
        grid = PnPDMSampler(model, PnPDMConfig(reverse_steps=steps))._reverse_grid(rho, model.device)
        mean, covariance = 1.0, 0.0
        for current, following in zip(grid[:-1], grid[1:], strict=True):
            dt = float(current**2 - following**2)
            transition = 1 - dt / (variance + float(current**2))
            mean, covariance = transition * mean, transition**2 * covariance + dt
        errors.append(abs(mean - exact_mean) + abs(covariance - exact_cov))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.001


def test_finite_langevin_has_its_declared_finite_gaussian_moments() -> None:
    model = GaussianModel(scale=0.4)
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)

    # No H: a differentiable caller composition takes the generic finite path.
    def operator(v: Tensor) -> tuple[Tensor]:
        return (v @ matrix.mT,)

    template = torch.zeros(8192, 1, 2, dtype=torch.float64)
    y = torch.full((1, 1, 2), 0.2, dtype=torch.float64)
    problem = SamplingProblem(model, operator, y, template, DatasetKind.T2MAP, 0.3, 0)
    cfg = PnPDMConfig(langevin_steps=7, langevin_step=0.001)
    x = problem.encode(torch.full_like(template, 0.1))
    rho = 0.25
    result = PnPDMSampler(model, cfg)._likelihood(x, problem, rho, torch.Generator().manual_seed(16))[:, 0, 0]
    b = matrix / model.scale
    precision = b.mT @ b / 0.3**2 + torch.eye(2) / rho**2
    transition = torch.eye(2) - cfg.langevin_step * precision
    mean = x[0, 0, 0].clone()
    covariance = torch.zeros_like(precision)
    force = b.mT @ y.flatten() / 0.3**2 + mean / rho**2
    for _ in range(cfg.langevin_steps):
        mean = transition @ mean + cfg.langevin_step * force
        covariance = transition @ covariance @ transition.mT + 2 * cfg.langevin_step * torch.eye(2)
    assert_moments(result, mean, covariance)
    assert not torch.allclose(covariance, precision.inverse(), atol=0.001)


class EchoSignalFourier:
    """Caller composition with a contrast axis before the chain axis."""

    def __init__(self) -> None:
        self.acquisition = FastFourierOp(dim=(-2, -1))
        self.times = torch.tensor([0.4, 1.2], dtype=torch.float64)[:, None, None, None]

    def __call__(self, physical: Tensor) -> tuple[Tensor]:
        return (self.acquisition(torch.exp(-self.times * physical.unsqueeze(0)))[0],)


def test_nonlinear_finite_langevin_uses_physical_signal_gradient_and_real_noise() -> None:
    model, operator = GaussianModel(scale=0.25), EchoSignalFourier()
    template = torch.tensor([[[0.7, 1.3]], [[1.1, 0.8]]], dtype=torch.float64)
    measurement = operator(template)[0] + 0.02 + 0.03j
    problem = SamplingProblem(model, operator, measurement, template, DatasetKind.T2MAP, 0.1, 1)
    cfg = PnPDMConfig(langevin_steps=1, langevin_step=1e-5)
    x = problem.encode(template * 0.8)
    signal = torch.exp(-operator.times * (template * 0.8).unsqueeze(0))
    residual = operator.acquisition(signal)[0] - measurement
    pullback = operator.acquisition.H(residual)[0].real
    physical_gradient = (-operator.times * signal * pullback).sum(0) / problem.noise_std**2
    noise = torch.randn(template.shape, generator=torch.Generator().manual_seed(17), dtype=template.dtype)
    expected = x[:, 0] - cfg.langevin_step * physical_gradient / model.scale + (2 * cfg.langevin_step) ** 0.5 * noise
    result = PnPDMSampler(model, cfg)._likelihood(x, problem, 0.2, torch.Generator().manual_seed(17))
    torch.testing.assert_close(result[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(result[:, 1]) == 0
    with pytest.raises(ValueError, match="requires a linear operator"):
        PnPDMSampler(model, replace(cfg, likelihood="linear_gaussian")).sample(
            operator,
            measurement,
            noise_std=0.1,
            template=template,
            kind=DatasetKind.T2MAP,
            measurement_batch_dim=1,
            generator=torch.Generator().manual_seed(17),
        )


def test_centered_orthonormal_mri_and_rng_and_measurement_units() -> None:
    model = GaussianModel()
    operator = FastFourierOp(dim=(-2, -1))
    template = torch.zeros(3, 1, 2, dtype=torch.complex128)
    measurement = operator(torch.full_like(template, 0.2 + 0.1j))[0]
    cfg = PnPDMConfig(rho=0.2, burn_in=2, num_samples=3, thin=2, reverse_steps=8, cg_rtol=1e-10)
    sampler = PnPDMSampler(model, cfg)

    def run(op: FastFourierOp, y: Tensor, tau: float, seed: int) -> Tensor:
        return sampler.sample(
            op, y, noise_std=tau, template=template, kind=DatasetKind.M0_COMPLEX, generator=torch.Generator().manual_seed(seed)
        )[0]

    first = run(operator, measurement, 0.1, 8)
    torch.testing.assert_close(first, run(operator, measurement, 0.1, 8), rtol=0, atol=0)
    torch.testing.assert_close(first, run(operator * 10, measurement * 10, 1.0, 8), rtol=1e-10, atol=1e-12)
    assert not torch.equal(first, run(operator, measurement, 0.1, 9))
    assert first.shape == (3, *template.shape) and first.dtype == template.dtype


def test_masked_fourier_real_map_conditional_matches_dense_real_coordinates() -> None:
    model = GaussianModel(scale=0.25)
    fourier = FastFourierOp(dim=(-2, -1))
    operator = torch.tensor([0.0, 1.0], dtype=torch.float64) * fourier
    template = torch.zeros(8192, 1, 2, dtype=torch.float64)
    y = operator(torch.tensor([[[0.7, 1.0]]], dtype=torch.float64))[0] + 0.01j
    problem = SamplingProblem(model, operator, y, template, DatasetKind.T1MAP, 0.1, 0)
    rho = 0.15
    x = problem.encode(torch.full_like(template, 0.3))
    draws = PnPDMSampler(model, PnPDMConfig(cg_rtol=1e-11))._likelihood(x, problem, rho, torch.Generator().manual_seed(36))
    basis = torch.eye(2, dtype=torch.float64).reshape(2, 1, 2) / model.scale
    b = operator(basis)[0][:, 0].mT
    real_matrix = torch.cat([b.real, b.imag])
    measured = torch.cat([y.flatten().real, y.flatten().imag])
    covariance = (real_matrix.mT @ real_matrix / 0.1**2 + torch.eye(2) / rho**2).inverse()
    mean = covariance @ (real_matrix.mT @ measured / 0.1**2 + x[0, 0, 0] / rho**2)
    assert_moments(draws[:, 0, 0], mean, covariance)
    assert torch.count_nonzero(draws[:, 1]) == 0


def test_nonlinear_sample_accepts_composed_physics_with_contrast_axis() -> None:
    model, operator = GaussianModel(scale=0.25), EchoSignalFourier()
    template = torch.tensor([[[0.7, 1.3]], [[1.1, 0.8]]], dtype=torch.float64)
    measurement = operator(template)[0] + 0.02j
    sampler = PnPDMSampler(model, PnPDMConfig(rho=0.1, burn_in=2, num_samples=2, reverse_steps=8, langevin_steps=3))
    args = {"noise_std": 0.1, "template": template, "kind": DatasetKind.T2MAP, "measurement_batch_dim": 1, "initial": template}
    (first,) = sampler.sample(operator, measurement, generator=torch.Generator().manual_seed(7), **args)
    (repeat,) = sampler.sample(operator, measurement, generator=torch.Generator().manual_seed(7), **args)
    torch.testing.assert_close(first, repeat, rtol=0, atol=0)
    assert first.isfinite().all() and first.shape == (2, *template.shape)
    assert not first.is_complex() and not first.requires_grad


def test_failed_cg_and_nonfinite_trajectories_raise() -> None:
    model = GaussianModel()
    operator = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.02]], dtype=torch.float64))
    template = torch.ones(2, 1, 2, dtype=torch.float64)
    cfg = PnPDMConfig(cg_max_iter=1, cg_rtol=1e-12)
    with pytest.raises(RuntimeError, match="CG did not converge"):
        PnPDMSampler(model, cfg).sample(
            operator, template, noise_std=0.01, template=template, kind=DatasetKind.T1MAP, generator=torch.Generator().manual_seed(1)
        )
    problem = SamplingProblem(model, operator, template, template, DatasetKind.T1MAP, 0.1, 0)
    with pytest.raises(RuntimeError, match="reverse SDE"):
        PnPDMSampler(model, cfg)._prior(problem.encode(template * float("nan")), problem, 0.2, torch.Generator().manual_seed(1))

    def nonfinite_forward(x: Tensor) -> tuple[Tensor]:
        return (x * float("nan"),)

    problem = SamplingProblem(model, nonfinite_forward, template, template, DatasetKind.T1MAP, 0.1, 0)
    with pytest.raises(RuntimeError, match="finite-Langevin"):
        PnPDMSampler(model, cfg)._likelihood(problem.encode(template), problem, 0.2, torch.Generator().manual_seed(1))


def test_collection_discards_annealing_and_equilibrates_at_fixed_rho(monkeypatch: pytest.MonkeyPatch) -> None:
    model = GaussianModel()
    cfg = PnPDMConfig(rho=0.1, rho_start=0.8, anneal_steps=3, burn_in=4, num_samples=2, thin=3)
    sampler = PnPDMSampler(model, cfg)
    visited: list[float] = []

    def likelihood(x: Tensor, problem: SamplingProblem, rho: float, generator: torch.Generator) -> Tensor:
        visited.append(rho)
        return x + 1

    monkeypatch.setattr(sampler, "_likelihood", likelihood)
    monkeypatch.setattr(sampler, "_prior", lambda z, problem, rho, generator: z)
    operator = MatrixOperator(torch.eye(2, dtype=torch.float64))
    template = torch.zeros(1, 1, 2, dtype=torch.float64)
    (draws,) = sampler.sample(
        operator, template, noise_std=0.1, template=template, kind=DatasetKind.T1MAP, generator=torch.Generator().manual_seed(1)
    )
    assert len(visited) == cfg.anneal_steps + cfg.burn_in + cfg.num_samples * cfg.thin
    assert visited[0] == cfg.rho_start and visited[cfg.anneal_steps - 1] == cfg.rho
    assert all(rho == cfg.rho for rho in visited[cfg.anneal_steps :])
    torch.testing.assert_close(draws[:, 0, 0, 0], torch.tensor([10.0, 13.0], dtype=torch.float64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": 0},
        {"rho": float("nan")},
        {"reverse_steps": 1},
        {"num_samples": 0},
        {"thin": 0},
        {"burn_in": -1},
        {"rho_start": 0.01},
        {"rho_start": 0.2},
        {"likelihood": "pfode"},
        {"langevin_step": float("inf")},
        {"cg_rtol": -1},
        {"cg_rtol": 1},
    ],
)
def test_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PnPDMConfig(**kwargs)


def test_unsupported_sigmas_and_invalid_initial_raise() -> None:
    model = GaussianModel()
    template = torch.zeros(1, 1, 2, dtype=torch.float64)
    operator = MatrixOperator(torch.eye(2, dtype=torch.float64))
    for cfg in (PnPDMConfig(rho=3), PnPDMConfig(reverse_sigma_min=0.0001), PnPDMConfig(reverse_sigma_min=0.1)):
        with pytest.raises(ValueError, match="model.sigma_min"):
            PnPDMSampler(model, cfg).sample(
                operator, template, noise_std=0.1, template=template, kind=DatasetKind.T1MAP, generator=torch.Generator().manual_seed(1)
            )
    with pytest.raises(ValueError, match="initial must match"):
        PnPDMSampler(model, PnPDMConfig()).sample(
            operator,
            template,
            noise_std=0.1,
            template=template,
            kind=DatasetKind.T1MAP,
            initial=template.to(torch.complex128),
            generator=torch.Generator().manual_seed(1),
        )
