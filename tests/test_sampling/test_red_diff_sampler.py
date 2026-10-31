"""Deterministic RED-Diff gradient controls and tiny reconstruction problems."""

from math import sqrt

import pytest
import torch
from mrpro.operators import FastFourierOp, Operator, models
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel, NormalizationConfig
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling.red_diff_sampler import REDDiffConfig, REDDiffSampler


class GaussianEDMPrior(DiffusionModel):
    """Exact correlated Gaussian denoiser with the real model's normalization API."""

    covariance: Tensor

    def __init__(self, scale: float = 0.25) -> None:
        super().__init__(
            unet=torch.nn.Identity(),
            diffusion=DiffusionConfig(normalization=NormalizationConfig(kind_scales={"T1MAP": scale})),
            compile_model=False,
        )
        self.double().eval()
        self.register_buffer("covariance", torch.tensor([[0.13, 0.04], [0.04, 0.08]], dtype=torch.float64))
        self.factor = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.calls: list[tuple[Tensor, Tensor, bool]] = []

    def _denoise_both_2ch(self, x_2ch: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self.calls.append((x_2ch.detach().clone(), sigma.detach().clone(), torch.is_grad_enabled()))
        gain = torch.linalg.solve(
            self.covariance + sigma[:, None, None].square() * torch.eye(2, dtype=x_2ch.dtype),
            self.covariance.expand(len(x_2ch), -1, -1),
        )
        clean = self.factor * torch.einsum("bij,bchj->bchi", gain, x_2ch)
        return clean, clean


class MatrixOperator:
    def __init__(self, matrix: Tensor) -> None:
        self.matrix = matrix

    def __call__(self, x: Tensor) -> tuple[Tensor]:
        return (x @ self.matrix.to(x).mT,)

    @property
    def H(self) -> "MatrixOperator":
        return MatrixOperator(self.matrix.mH)


def make_problem(model: GaussianEDMPrior, *, complex_state: bool = False, noise_std: float = 0.3, batch: int = 1) -> SamplingProblem:
    dtype = torch.complex128 if complex_state else torch.float64
    kind = DatasetKind.COMPLEX_T1W_COIL_COMBINED if complex_state else DatasetKind.T1MAP
    template = torch.ones(batch, 1, 2, dtype=dtype)
    matrix = torch.tensor([[1.0, 0.3], [-0.2, 0.5], [0.1, 0.4]], dtype=torch.float64)
    y = torch.full((batch, 1, 3), 0.2 + 0.1j if complex_state else 0.2, dtype=dtype)
    return SamplingProblem(model, MatrixOperator(matrix), y, template, kind, noise_std, 0)


@pytest.mark.parametrize("complex_state", [False, True])
@pytest.mark.parametrize("sigma", [0.03, 0.4, 1.2])
def test_objective_and_gradient_match_official_epsilon_surrogate(complex_state: bool, sigma: float) -> None:
    model = GaussianEDMPrior()
    cfg = REDDiffConfig(prior_weight=0.7)
    problem = make_problem(model, complex_state=complex_state, batch=2)
    mu = problem.encode(problem.template * 0.6).detach().requires_grad_(True)
    epsilon = problem.noise(torch.Generator().manual_seed(6))
    sampler = REDDiffSampler(model, cfg)
    objective = sampler._objective(mu, problem, sigma, epsilon)
    (gradient,) = torch.autograd.grad(objective, mu)

    # Independent VP-coordinate construction: alpha is amplitude, not alpha_bar.
    alpha, sigma_vp = 1 / sqrt(1 + sigma**2), sigma / sqrt(1 + sigma**2)
    xt = alpha * mu.detach() + sigma_vp * epsilon
    identity = torch.eye(2, dtype=mu.dtype)
    gain = torch.linalg.solve(model.covariance + sigma**2 * identity, model.covariance)
    clean = (xt / alpha) @ gain.mT
    eps_hat = (xt - alpha * clean) / sigma_vp
    stopped = problem.constrain((eps_hat - epsilon).detach())
    expected_prior = cfg.prior_weight * (sigma_vp / alpha) * stopped
    likelihood = problem.residual(mu).abs().square().sum() / (2 * problem.noise_std**2)
    expected_loss = likelihood + (expected_prior * mu).sum()
    torch.testing.assert_close(objective, expected_loss)
    torch.testing.assert_close(gradient, problem.gradient(mu) + expected_prior)
    assert all(not grad_enabled for _, _, grad_enabled in model.calls)
    assert model.factor.grad is None

    # Differentiating through the residual is a different (incorrect) update.
    bad_clean = mu @ gain.mT + sigma * epsilon @ gain.mT
    bad_loss = cfg.prior_weight * (problem.constrain(mu - bad_clean) * mu).sum()
    (bad_gradient,) = torch.autograd.grad(bad_loss, mu)
    assert not torch.allclose(bad_gradient, expected_prior)


def test_expected_gradient_matches_smoothed_gaussian_energy_and_minimizer() -> None:
    model, sigma, weight = GaussianEDMPrior(), 0.4, 1.3
    problem = make_problem(model)
    sampler = REDDiffSampler(model, REDDiffConfig(prior_weight=weight))
    mu = problem.encode(problem.template * 0.5).detach().requires_grad_(True)
    epsilon = problem.noise(torch.Generator().manual_seed(1))
    # Antithetic noise integrates this affine gradient exactly; no Monte Carlo tolerance.
    objective = sum(sampler._objective(mu, problem, sigma, sign * epsilon) for sign in (-1, 1)) / 2
    (gradient,) = torch.autograd.grad(objective, mu)
    precision = sigma**2 * torch.linalg.inv(model.covariance + sigma**2 * torch.eye(2, dtype=mu.dtype))
    energy = problem.residual(mu).abs().square().sum() / (2 * problem.noise_std**2)
    energy = energy + weight / 2 * (mu * (mu @ precision.mT)).sum()
    (expected,) = torch.autograd.grad(energy, mu)
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)

    assert isinstance(problem.operator, MatrixOperator)
    matrix = problem.operator.matrix / problem.scale.item()
    hessian = matrix.mT @ matrix / problem.noise_std**2 + weight * precision
    center = torch.linalg.solve(hessian, matrix.mT @ problem.measurement.flatten() / problem.noise_std**2)
    at_minimum = problem.pack(center.reshape(problem.shape)).requires_grad_(True)
    loss = sum(sampler._objective(at_minimum, problem, sigma, sign * epsilon) for sign in (-1, 1)) / 2
    torch.testing.assert_close(torch.autograd.grad(loss, at_minimum)[0], torch.zeros_like(at_minimum), atol=1e-13, rtol=0)
    # The exact finite-sigma RED-Diff control is a regularized reconstruction,
    # not the Gaussian posterior mean for the unsmoothed prior.
    posterior = torch.linalg.solve(
        matrix.mT @ matrix / problem.noise_std**2 + torch.linalg.inv(model.covariance),
        matrix.mT @ problem.measurement.flatten() / problem.noise_std**2,
    )
    assert not torch.allclose(center, posterior)


def test_likelihood_gradient_has_per_real_noise_and_summed_reductions() -> None:
    model = GaussianEDMPrior()
    problem = make_problem(model, complex_state=True, batch=2)
    mu = problem.encode(problem.template * (0.5 + 0.1j)).detach().requires_grad_(True)
    sampler = REDDiffSampler(model, REDDiffConfig(prior_weight=0))
    loss = sampler._objective(mu, problem, 0.3, torch.zeros_like(mu))
    residual = problem.residual(mu)
    torch.testing.assert_close(loss, (residual.real.square() + residual.imag.square()).sum() / (2 * problem.noise_std**2))
    (gradient,) = torch.autograd.grad(loss, mu)
    assert isinstance(problem.operator, MatrixOperator)
    expected = problem.pack(problem.operator.H(residual)[0]) / problem.noise_std**2
    torch.testing.assert_close(gradient, expected)
    one = make_problem(model, complex_state=True)
    single = mu[:1].detach().requires_grad_(True)
    single_loss = sampler._objective(single, one, 0.3, torch.zeros_like(single))
    torch.testing.assert_close(gradient[:1], torch.autograd.grad(single_loss, single)[0])
    torch.testing.assert_close(loss, 2 * single_loss)
    assert model.calls == []


def test_public_model_normalization_and_noise_are_mapped_together() -> None:
    model = GaussianEDMPrior(scale=0.125)
    problem = make_problem(model)
    mu = problem.encode(problem.template)
    sigma, epsilon = 0.2, problem.noise(torch.Generator().manual_seed(4))
    REDDiffSampler(model, REDDiffConfig())._objective(mu, problem, sigma, epsilon)
    z, levels, _ = model.calls[-1]
    torch.testing.assert_close(z, model.normalize(problem.pack(problem.template), problem.kind) + sigma * epsilon)
    physical_noise = model.denormalize(sigma * epsilon, problem.kind)
    torch.testing.assert_close(problem.pack(problem.decode(z)), problem.pack(problem.template) + physical_noise)
    torch.testing.assert_close(levels, levels.new_full((len(z),), sigma))
    assert torch.count_nonzero(z[:, 1]) == 0


def test_tiny_linear_inverse_problem_matches_least_squares() -> None:
    model = GaussianEDMPrior()
    matrix = torch.tensor([[1.0, 0.3], [-0.2, 0.7], [0.1, 0.4]], dtype=torch.float64)
    operator = MatrixOperator(matrix)
    template = torch.zeros(1, 1, 2, dtype=torch.complex128)
    measurement = torch.tensor([[[0.4 + 0.2j, -0.1 + 0.3j, 0.05 - 0.02j]]], dtype=torch.complex128)
    expected = torch.linalg.lstsq(matrix.to(measurement), measurement.flatten()).solution
    result = REDDiffSampler(model, REDDiffConfig(num_steps=250, learning_rate=0.04, prior_weight=0)).sample(
        operator,
        measurement,
        template=template,
        kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
        noise_std=0.3,
        generator=torch.Generator().manual_seed(7),
    )[0]
    torch.testing.assert_close(result.flatten(), expected, atol=2e-5, rtol=0)
    assert not result.requires_grad


def test_finite_schedule_matches_hand_computed_adam_updates() -> None:
    model = GaussianEDMPrior()
    problem = make_problem(model)
    assert isinstance(problem.operator, MatrixOperator)
    cfg = REDDiffConfig(num_steps=3, learning_rate=0.02, prior_weight=0.8, sigma_start=1.0, sigma_end=0.25, rho=1)
    initial = problem.template * 0.9
    sampler = REDDiffSampler(model, cfg)
    rng = torch.Generator().manual_seed(3)
    expected = problem.encode(initial).detach()
    first, second = torch.zeros_like(expected), torch.zeros_like(expected)
    identity = torch.eye(2, dtype=expected.dtype)
    for iteration, sigma in enumerate([1.0, 0.625, 0.25], start=1):
        epsilon = problem.noise(rng)
        gain = torch.linalg.solve(model.covariance + sigma**2 * identity, model.covariance)
        clean = (expected + sigma * epsilon) @ gain.mT
        prior_gradient = cfg.prior_weight * problem.constrain(expected - clean)
        residual = problem.operator(problem.decode(expected))[0] - problem.measurement
        data_gradient = problem.pack(problem.operator.H(residual)[0]) / (problem.scale * problem.noise_std**2)
        gradient = data_gradient + prior_gradient
        first = 0.9 * first + 0.1 * gradient
        second = 0.99 * second + 0.01 * gradient.square()
        expected -= cfg.learning_rate * (first / (1 - 0.9**iteration)) / ((second / (1 - 0.99**iteration)).sqrt() + 1e-8)
    with torch.no_grad():
        result = sampler.sample(
            problem.operator,
            problem.measurement,
            noise_std=problem.noise_std,
            template=problem.template,
            initial=initial,
            kind=DatasetKind.T1MAP,
            generator=torch.Generator().manual_seed(3),
        )[0]
    torch.testing.assert_close(result, problem.decode(expected), rtol=1e-12, atol=1e-12)
    assert [float(levels[0]) for _, levels, _ in model.calls] == [1.0, 0.625, 0.25]
    assert not result.requires_grad


class ZeroNetwork(torch.nn.Module):
    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(x), torch.zeros_like(x)


@pytest.mark.parametrize("kind", [DatasetKind.T1MAP, DatasetKind.COMPLEX_T1W_COIL_COMBINED])
def test_actual_edm_preconditioned_main_head(kind: DatasetKind) -> None:
    model = DiffusionModel(unet=ZeroNetwork(), compile_model=False).eval()
    dtype = torch.complex64 if kind.is_complex else torch.float32
    template = torch.ones(1, 1, 2, dtype=dtype)
    operator = FastFourierOp(dim=(-2, -1))
    result = REDDiffSampler(model, REDDiffConfig(num_steps=3)).sample(
        operator,
        operator(template)[0],
        template=template,
        kind=kind,
        noise_std=0.2,
        generator=torch.Generator().manual_seed(4),
    )[0]
    assert result.isfinite().all() and result.dtype == template.dtype
    assert not result.requires_grad and all(parameter.grad is None for parameter in model.parameters())


class KnownM0IR(Operator):
    """Caller composes fixed nuisance parameters with a single unknown T1 map."""

    def __init__(self) -> None:
        super().__init__()
        self.signal = models.InversionRecovery(torch.tensor([0.1, 0.5, 1.5], dtype=torch.float64))

    def forward(self, t1: Tensor) -> tuple[Tensor]:
        return (self.signal(torch.ones_like(t1), t1)[0],)


def test_nonlinear_signal_composition_uses_exact_likelihood_gradient() -> None:
    model = GaussianEDMPrior()
    # Pinned mrpro signal and centered orthonormal Fourier, composed as operators.
    operator = FastFourierOp(dim=(-2, -1)) @ KnownM0IR()
    physical = torch.tensor([[[1.0, 1.3]], [[0.7, 1.7]]], dtype=torch.float64)
    y = operator(physical)[0] + 0.02 + 0.03j
    problem = SamplingProblem(model, operator, y, physical, DatasetKind.T1MAP, 0.1, 1)
    mu = problem.encode(physical * 0.9).requires_grad_(True)
    sampler = REDDiffSampler(model, REDDiffConfig(num_steps=4, learning_rate=0.001, prior_weight=0))
    loss = sampler._objective(mu, problem, 0.4, torch.zeros_like(mu))
    gradient = torch.autograd.grad(loss, mu)[0]
    direction = problem.pack(torch.full_like(physical, 0.3))

    def energy(value: Tensor) -> Tensor:
        residual: Tensor = operator(problem.decode(value))[0] - y
        return residual.abs().square().sum() / (2 * problem.noise_std**2)

    delta = 1e-6
    finite_difference = (energy(mu + delta * direction) - energy(mu - delta * direction)) / (2 * delta)
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, rtol=1e-8, atol=1e-8)
    result = sampler.sample(
        operator,
        y,
        noise_std=0.1,
        template=physical,
        initial=physical * 0.9,
        kind=DatasetKind.T1MAP,
        measurement_batch_dim=1,
        generator=torch.Generator().manual_seed(4),
    )[0]
    assert energy(problem.encode(result)) < energy(mu)
    assert result.dtype == physical.dtype and not result.requires_grad
    assert model.calls == []


def test_centered_fourier_reconstruction_rng_units_and_batch_independence() -> None:
    model = GaussianEDMPrior()
    operator = FastFourierOp(dim=(-2, -1))
    template = torch.ones(2, 1, 2, dtype=torch.complex128)
    y = operator(template * (0.4 + 0.2j))[0]
    sampler = REDDiffSampler(model, REDDiffConfig(num_steps=5, sigma_start=0.5))

    def reconstruct(seed: int, *, factor: float = 1.0, single: bool = False) -> Tensor:
        batch = slice(0, 1) if single else slice(None)
        return sampler.sample(
            factor * operator,
            factor * y[batch],
            noise_std=factor * 0.2,
            template=template[batch],
            kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
            generator=torch.Generator().manual_seed(seed),
        )[0]

    global_state = torch.random.get_rng_state().clone()
    first = reconstruct(5)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)
    torch.testing.assert_close(first, reconstruct(5), rtol=0, atol=0)
    torch.testing.assert_close(first, reconstruct(5, factor=10), rtol=1e-12, atol=1e-12)
    assert not torch.equal(first, reconstruct(6))
    # Match supplied per-step epsilon exactly for a separate first-batch control.
    # A scalar objective sum must not reduce each item's gradient by batch size.
    problem = SamplingProblem(model, operator, y, template, DatasetKind.COMPLEX_T1W_COIL_COMBINED, 0.2, 0)
    one = SamplingProblem(model, operator, y[:1], template[:1], DatasetKind.COMPLEX_T1W_COIL_COMBINED, 0.2, 0)
    mu = problem.encode(template).requires_grad_(True)
    epsilon = problem.noise(torch.Generator().manual_seed(2))
    batch_loss = sampler._objective(mu, problem, 0.4, epsilon)
    batch_gradient = torch.autograd.grad(batch_loss, mu)[0]
    single = mu[:1].detach().requires_grad_(True)
    single_loss = sampler._objective(single, one, 0.4, epsilon[:1])
    torch.testing.assert_close(batch_gradient[:1], torch.autograd.grad(single_loss, single)[0])
    assert model.factor.grad is None
    assert all(not grad_enabled for _, _, grad_enabled in model.calls)


@pytest.mark.parametrize(
    "settings",
    [{"num_steps": 1}, {"learning_rate": 0}, {"prior_weight": -1}, {"rho": float("nan")}, {"sigma_end": 0}],
)
def test_invalid_config(settings: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        REDDiffConfig(**settings)


def test_rejects_unsupported_schedule_invalid_initial_and_nonfinite_physics() -> None:
    model = GaussianEDMPrior()
    problem = make_problem(model)
    kwargs = {
        "template": problem.template,
        "kind": DatasetKind.T1MAP,
        "noise_std": 0.2,
        "generator": torch.Generator().manual_seed(1),
    }
    with pytest.raises(ValueError, match="supported sigma"):
        REDDiffSampler(model, REDDiffConfig(sigma_end=1e-5)).sample(problem.operator, problem.measurement, **kwargs)
    sampler = REDDiffSampler(model, REDDiffConfig(num_steps=2))
    with pytest.raises(ValueError, match="initial must match"):
        sampler.sample(problem.operator, problem.measurement, initial=problem.template.to(torch.complex128), **kwargs)
    with pytest.raises(RuntimeError, match="Nonfinite RED-Diff objective"):
        sampler.sample(lambda x: (x * float("nan"),), problem.template, **kwargs)
