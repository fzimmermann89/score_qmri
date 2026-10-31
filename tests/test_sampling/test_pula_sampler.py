"""Exact Gaussian controls for the fixed-metric pULA reference."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from mrpro.operators import FastFourierOp, Operator

from score_qmri.datasets.kind import DatasetKind
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling.pula_sampler import PULAConfig, PULASampler


class GaussianPrior:
    device = torch.device("cpu")
    sigma_min = 0.001
    sigma_max = 5.0

    def __init__(self, covariance: torch.Tensor, scale: float = 1.0) -> None:
        self.covariance, self.scale = covariance, scale

    def normalize(self, x: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def denormalize(self, x: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        return x / self.scale

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        c = self.covariance.to(x)
        gain = torch.linalg.solve(c + sigma[0] ** 2 * torch.eye(c.shape[0], dtype=x.dtype), c)
        denoised: torch.Tensor = x @ gain.mT
        return denoised


class MatrixOperator:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x @ self.matrix.to(x).mT,)

    @property
    def H(self) -> "MatrixOperator":
        return MatrixOperator(self.matrix.mH)


def rotation(angle: float) -> torch.Tensor:
    t = torch.tensor(angle, dtype=torch.float64)
    return torch.stack([torch.stack([t.cos(), -t.sin()]), torch.stack([t.sin(), t.cos()])])


def setup_gaussian(scale: float = 1.0) -> tuple[GaussianPrior, MatrixOperator, torch.Tensor]:
    r, v = rotation(0.6), rotation(-0.4)
    c = r @ torch.diag(torch.tensor([0.04, 0.4], dtype=torch.float64)) @ r.mT
    a = v @ torch.diag(torch.tensor([1.5, 0.35], dtype=torch.float64)) @ v.mT
    return GaussianPrior(c, scale), MatrixOperator(a), torch.tensor([[[0.2, -0.3]]], dtype=torch.float64)


def gaussian_targets(
    model: GaussianPrior, operator: MatrixOperator, y: torch.Tensor, cfg: PULAConfig, tau: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = operator.matrix / model.scale
    c = model.covariance + cfg.sigma**2 * torch.eye(2, dtype=torch.float64)
    precision = torch.linalg.inv(c) + a.mT @ a / tau**2
    target = torch.linalg.inv(precision)
    mean = target @ a.mT @ y.flatten() / tau**2
    metric = torch.linalg.inv(a.mT @ a / tau**2 + torch.eye(2, dtype=torch.float64) / cfg.sigma**2)
    transition = torch.eye(2, dtype=torch.float64) - cfg.step_size * metric @ precision
    # Discrete Lyapunov equation, including a noncommuting rotated metric.
    stationary = torch.linalg.solve(
        torch.eye(4, dtype=torch.float64) - torch.kron(transition, transition), (2 * cfg.step_size * metric).flatten()
    ).unflatten(0, (2, 2))
    return mean / model.scale, target / model.scale**2, stationary / model.scale**2


@pytest.mark.parametrize("step", [0.1, 0.5])
def test_rotated_gaussian_mean_covariance_and_finite_step_bias(step: float) -> None:
    model, operator, y = setup_gaussian()
    cfg = PULAConfig(sigma=0.2, step_size=step, burn_in=400, cg_rtol=1e-10)
    mean, continuous, discrete = gaussian_targets(model, operator, y, cfg, 0.3)
    (draws,) = PULASampler(model, cfg).sample(
        operator,
        y,
        noise_std=0.3,
        template=torch.empty(8000, 1, 2, dtype=torch.float64),
        kind=DatasetKind.T1MAP,
        generator=torch.Generator().manual_seed(420),
    )
    values = draws[:, 0]
    assert ((values.mean(0) - mean).abs() < 5 * (discrete.diag() / len(values)).sqrt()).all()
    se = ((discrete.diag()[:, None] * discrete.diag()[None, :] + discrete.square()) / (len(values) - 1)).sqrt()
    assert ((torch.cov(values.mT) - discrete).abs() < 5 * se).all()
    _, _, smaller_step = gaussian_targets(model, operator, y, PULAConfig(sigma=0.2, step_size=step / 2), 0.3)
    assert (smaller_step - continuous).norm() < (discrete - continuous).norm()
    if step == 0.5:
        assert (torch.cov(values.mT) - discrete).norm() < (torch.cov(values.mT) - continuous).norm() / 2


def test_random_rhs_covariance_is_metric_in_both_complex_components() -> None:
    model, operator, _ = setup_gaussian()
    cfg = PULAConfig(sigma=0.2, burn_in=0, cg_rtol=1e-10)
    template = torch.zeros(40000, 1, 2, dtype=torch.complex128)
    problem = SamplingProblem(model, operator, template[:1], template, DatasetKind.M0_COMPLEX, 0.3, 0)
    sampler = PULASampler(model, cfg)
    apply, perturbation = sampler._metric(problem, None, torch.Generator().manual_seed(92))
    noise = apply(perturbation())
    q = operator.matrix.mT @ operator.matrix / 0.3**2 + torch.eye(2, dtype=torch.float64) / cfg.sigma**2
    expected = torch.linalg.inv(q)
    for component in noise.unbind(1):
        torch.testing.assert_close(torch.cov(component[:, 0].mT), expected, rtol=0.04, atol=0.0002)
    assert (noise[:, 0] * noise[:, 1]).mean().abs() < 0.0005
    assert (torch.cov(noise[:, 0, 0].mT) - expected).norm() < (expected - expected @ expected).norm() / 10


def test_physical_normalization_scales_denoiser_and_likelihood_together() -> None:
    model, operator, y = setup_gaussian(scale=0.25)
    cfg = PULAConfig(sigma=0.2, step_size=0.1, burn_in=250, cg_rtol=1e-10)
    mean, _, covariance = gaussian_targets(model, operator, y, cfg, 0.3)
    (draws,) = PULASampler(model, cfg).sample(
        operator,
        y,
        noise_std=0.3,
        template=torch.empty(6000, 1, 2, dtype=torch.float64),
        kind=DatasetKind.T1MAP,
        generator=torch.Generator().manual_seed(94),
    )
    values = draws[:, 0]
    torch.testing.assert_close(values.mean(0), mean, rtol=0, atol=5 * (covariance.diag().max() / len(values)).sqrt().item())
    torch.testing.assert_close(torch.cov(values.mT), covariance, rtol=0.08, atol=0.001)


def test_centered_orthonormal_masked_fourier_complex_gaussian() -> None:
    fourier = FastFourierOp(dim=(-2, -1))
    mask = torch.tensor([[1.0, 0], [0, 1]], dtype=torch.float64)
    operator = mask * fourier
    cfg = PULAConfig(sigma=0.2, step_size=0.25, burn_in=120, cg_rtol=1e-10)
    model = GaussianPrior(0.05 * torch.eye(2, dtype=torch.float64))
    y = torch.tensor([[[0.2 + 0.1j, 0], [0, -0.1 + 0.3j]]], dtype=torch.complex128)
    (draws,) = PULASampler(model, cfg).sample(
        operator,
        y,
        noise_std=0.3,
        template=torch.empty(3000, 2, 2, dtype=torch.complex128),
        kind=DatasetKind.M0_COMPLEX,
        generator=torch.Generator().manual_seed(8),
    )
    coefficients = fourier(draws)[0]
    target_var = 1 / (1 / (0.05 + cfg.sigma**2) + mask / 0.3**2)
    metric = 1 / (1 / cfg.sigma**2 + mask / 0.3**2)
    discrete_var = target_var / (1 - cfg.step_size * metric / (2 * target_var))
    mean = target_var * y / 0.3**2
    for part, center in [(coefficients.real, mean.real), (coefficients.imag, mean.imag)]:
        torch.testing.assert_close(part.mean(0), center[0], rtol=0, atol=0.025)
        torch.testing.assert_close(part.var(0), discrete_var, rtol=0.09, atol=0)


class CubicSignal(Operator):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.stack([x + 0.1 * x**3, 0.5 * x], dim=0),)


def test_nonlinear_transfer_composed_operator_actual_gradient_and_reproducibility() -> None:
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * 0.1, scale=0.25)
    operator = FastFourierOp(dim=(-2, -1)) @ CubicSignal()
    initial = torch.tensor([[[0.2, -0.3]]], dtype=torch.float64)
    measurement = operator(initial)[0] + (0.03 + 0.02j)
    cfg = PULAConfig(sigma=0.2, step_size=0.03, burn_in=0)
    sampler = PULASampler(model, cfg)
    diagonal = torch.tensor([[[[0.01, 0.04]]]], dtype=torch.float64)
    kwargs = {"noise_std": 0.1, "template": initial, "kind": DatasetKind.T1MAP, "initial": initial, "measurement_batch_dim": 1}
    with pytest.raises(ValueError, match="Nonlinear"):
        sampler.sample(operator, measurement, generator=torch.Generator().manual_seed(12), **kwargs)
    problem = SamplingProblem(model, operator, measurement, initial, DatasetKind.T1MAP, 0.1, 1)
    u = problem.encode(initial).requires_grad_(True)
    (gradient,) = torch.autograd.grad(problem.residual(u).abs().square().sum() / (2 * 0.1**2), u)
    rng = torch.Generator().manual_seed(12)
    prior = (model.denoise(u, u.new_tensor([cfg.sigma]), problem.kind) - u) / cfg.sigma**2
    expected = problem.decode(
        problem.constrain(u + cfg.step_size * diagonal * (prior - gradient) + (2 * cfg.step_size * diagonal).sqrt() * problem.noise(rng))
    )
    (actual,) = sampler.sample(operator, measurement, diagonal=diagonal, generator=torch.Generator().manual_seed(12), **kwargs)
    torch.testing.assert_close(actual, expected)
    (again,) = sampler.sample(operator, measurement, diagonal=diagonal, generator=torch.Generator().manual_seed(12), **kwargs)
    torch.testing.assert_close(actual, again, rtol=0, atol=0)
    (different,) = sampler.sample(operator, measurement, diagonal=diagonal, generator=torch.Generator().manual_seed(13), **kwargs)
    assert not torch.equal(actual, different)


def test_diagonal_is_copied_and_noise_matches_frozen_metric() -> None:
    model, operator, _ = setup_gaussian()
    template = torch.zeros(30000, 1, 2, dtype=torch.float64)
    problem = SamplingProblem(model, operator, template[:1], template, DatasetKind.T1MAP, 0.3, 0)
    diagonal = torch.tensor([0.03, 0.2], dtype=torch.float64, requires_grad=True)
    apply, perturbation = PULASampler(model, PULAConfig())._metric(problem, diagonal, torch.Generator().manual_seed(2))
    expected = diagonal.detach().clone()
    with torch.no_grad():
        diagonal.fill_(42)
    noise = apply(perturbation())
    torch.testing.assert_close(apply(torch.ones_like(noise))[:, 0, 0], expected.expand(30000, -1))
    torch.testing.assert_close(noise[:, 0, 0].var(0), expected, rtol=0.04, atol=0)
    assert not noise.requires_grad
    assert torch.count_nonzero(noise[:, 1]) == 0


def test_burn_in_thinning_and_last_output_agree() -> None:
    model, operator, y = setup_gaussian()
    cfg = PULAConfig(sigma=0.2, burn_in=5, num_samples=3, thinning=2)
    sampler = PULASampler(model, cfg)
    kwargs = {"noise_std": 0.3, "template": torch.zeros(2, 1, 2, dtype=torch.float64), "kind": DatasetKind.T1MAP}
    (chain,) = sampler.sample(operator, y, return_chain=True, generator=torch.Generator().manual_seed(31), **kwargs)
    (last,) = sampler.sample(operator, y, generator=torch.Generator().manual_seed(31), **kwargs)
    assert chain.shape == (3, 2, 1, 2)
    torch.testing.assert_close(chain[-1], last, rtol=0, atol=0)
    manual = PULASampler(model, PULAConfig(sigma=0.2, burn_in=0, num_samples=11))
    (all_states,) = manual.sample(operator, y, return_chain=True, generator=torch.Generator().manual_seed(31), **kwargs)
    torch.testing.assert_close(chain, all_states[[6, 8, 10]], rtol=0, atol=0)
    with pytest.raises(FrozenInstanceError):
        cfg.step_size = 2  # type: ignore[misc]


@pytest.mark.parametrize("diagonal", [torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(float("nan")), torch.ones(3), torch.tensor(1j)])
def test_invalid_metrics_fail(diagonal: torch.Tensor) -> None:
    model, operator, y = setup_gaussian()
    with pytest.raises(ValueError, match="diagonal"):
        PULASampler(model, PULAConfig()).sample(
            operator,
            y,
            noise_std=0.3,
            template=y,
            kind=DatasetKind.T1MAP,
            diagonal=diagonal,
            generator=torch.Generator(),
        )


@pytest.mark.parametrize(
    "kwargs", [{"sigma": 0}, {"step_size": float("nan")}, {"burn_in": -1}, {"thinning": 0}, {"num_samples": 1.5}, {"cg_rtol": 1}]
)
def test_invalid_config_fails(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PULAConfig(**kwargs)


def test_solver_failure_and_nonfinite_trajectory_are_reported() -> None:
    model, operator, y = setup_gaussian()
    kwargs = {"noise_std": 0.3, "template": y, "kind": DatasetKind.T1MAP}
    with pytest.raises(RuntimeError, match="CG did not converge"):
        PULASampler(model, PULAConfig(sigma=0.2, cg_max_iter=1, cg_rtol=1e-12)).sample(
            operator,
            y,
            generator=torch.Generator().manual_seed(21),
            **kwargs,
        )
    with pytest.raises(RuntimeError, match="Nonfinite pULA"):
        PULASampler(model, PULAConfig(sigma=0.2, step_size=1e308, burn_in=1)).sample(
            operator,
            y,
            diagonal=torch.tensor(100.0),
            initial=y,
            generator=torch.Generator().manual_seed(21),
            **kwargs,
        )
