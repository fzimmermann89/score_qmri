"""Analytic controls for ΠGDM's score, paper update and labeled extension."""

from math import sqrt
from typing import Any

import pytest
import torch
from mrpro.operators import FastFourierOp, LinearOperator, SensitivityOp, models
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling.edm_sampler import karras_sigmas
from score_qmri.sampling.pigdm_sampler import PiGDMConfig, PiGDMSampler

COMPLEX = DatasetKind.COMPLEX_T1W_COIL_COMBINED


class GaussianPrior:
    """Exact model-coordinate Gaussian denoiser, independent real channels."""

    device = torch.device("cpu")
    sigma_min = 0.001
    sigma_max = 5.0
    sigma_data = 0.2

    def __init__(self, covariance: Tensor, scale: float = 1.0) -> None:
        self.covariance, self.scale = covariance, scale

    def normalize(self, x: Tensor, kind: Tensor) -> Tensor:
        return x * self.scale

    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor:
        return x / self.scale

    def denoise(self, x: Tensor, sigma: Tensor, kind: Tensor) -> Tensor:
        covariance = self.covariance.to(x)
        gain = torch.linalg.solve(
            covariance + sigma[:, None, None] ** 2 * torch.eye(len(covariance)).to(x), covariance.expand(len(x), -1, -1)
        )
        return torch.einsum("bij,bcj->bci", gain, x.flatten(2)).reshape_as(x)


class MatrixOperator:
    def __init__(self, matrix: Tensor) -> None:
        self.matrix = matrix

    def __call__(self, x: Tensor) -> tuple[Tensor]:
        return (x @ self.matrix.to(x).mT,)

    @property
    def H(self) -> "MatrixOperator":
        return MatrixOperator(self.matrix.mH)


class Mask(LinearOperator):
    def __init__(self, mask: Tensor) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, x: Tensor) -> tuple[Tensor]:
        return (self.mask * x,)

    def adjoint(self, x: Tensor) -> tuple[Tensor]:
        return self.forward(x)


def real_vector(value: Tensor) -> Tensor:
    return torch.view_as_real(value).flatten() if value.is_complex() else value.flatten()


def dense_forward(problem: SamplingProblem, center: Tensor) -> Tensor:
    jacobian: Tensor = torch.autograd.functional.jacobian(lambda z: real_vector(problem.forward(z)), center)
    return jacobian.flatten(1)


def test_linear_score_is_exact_for_isotropic_gaussian_and_noisy_rank_deficient_operator() -> None:
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2], [2.0, 0.8]], dtype=torch.float64)
    op = MatrixOperator(matrix)
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * 0.04)
    x = torch.tensor([[[[0.2, -0.1]], [[0.1, 0.3]]]], dtype=torch.float64)
    y = torch.tensor([[[0.1 + 0.05j, -0.03 + 0.1j, 0.22 - 0.12j]]], dtype=torch.complex128)
    problem = SamplingProblem(model, op, y, op.H(y)[0], COMPLEX, 0.07, 0)
    sigma = 0.4
    sampler = PiGDMSampler(model, PiGDMConfig(cg_rtol=1e-12))
    denoised, guidance = sampler._denoise_and_guidance(x, problem, sigma)
    gain = 0.04 / (0.04 + sigma**2)
    conditional_covariance = sigma**2 * gain * torch.eye(2, dtype=x.dtype)
    observation_covariance = 0.07**2 * torch.eye(3, dtype=x.dtype) + matrix @ conditional_covariance @ matrix.T

    def likelihood(z: Tensor) -> Tensor:
        residual = problem.residual(gain * z)
        value: Tensor = -0.5 * (residual.conj() * torch.linalg.solve(observation_covariance.to(y), residual.mT).mT).real.sum()
        return value

    expected = torch.autograd.functional.jacobian(likelihood, x)
    torch.testing.assert_close(denoised, gain * x)
    torch.testing.assert_close(guidance, expected, atol=1e-11, rtol=1e-11)
    # The null direction of observation covariance is regularized by physical noise.
    assert torch.linalg.matrix_rank(matrix @ matrix.T) < 3


def test_covariance_is_declared_scalar_not_anisotropic_tweedie_or_rayleigh_estimate() -> None:
    covariance = torch.tensor([[0.01, 0.003], [0.003, 0.09]], dtype=torch.float64)
    model = GaussianPrior(covariance)
    op = MatrixOperator(torch.diag(torch.tensor([1.0, 0.2], dtype=torch.float64)))
    y = torch.tensor([[[0.1 + 0.05j, -0.03 + 0.1j]]], dtype=torch.complex128)
    problem = SamplingProblem(model, op, y, op.H(y)[0], COMPLEX, 0.07, 0)
    x = torch.full((1, 2, 1, 2), 0.2, dtype=torch.float64)
    sigma = 0.4
    for std in (None, 0.1):
        sampler = PiGDMSampler(model, PiGDMConfig(covariance_std=std, cg_rtol=1e-12))
        denoised, guidance = sampler._denoise_and_guidance(x, problem, sigma)
        jacobian = dense_forward(problem, denoised)
        residual = -real_vector(problem.residual(denoised))
        r2 = sigma**2 * (std or model.sigma_data) ** 2 / (sigma**2 + (std or model.sigma_data) ** 2)
        c = 0.07**2 * torch.eye(len(residual), dtype=x.dtype) + r2 * jacobian @ jacobian.T
        pullback = (jacobian.T @ torch.linalg.solve(c, residual)).reshape_as(x)
        expected = model.denoise(pullback, x.new_tensor([sigma]), problem.kind)
        torch.testing.assert_close(guidance, expected, atol=1e-12, rtol=1e-12)
        gain = torch.linalg.solve(covariance + sigma**2 * torch.eye(2), covariance)
        true_c = 0.07**2 * torch.eye(2) + op.matrix @ (sigma**2 * gain) @ op.matrix.T
        exact = (gain @ op.matrix.T).to(y) @ torch.linalg.solve(true_c.to(y), (y - op(problem.decode(denoised))[0]).flatten())
        actual = problem.unpack(guidance).flatten()
        assert not torch.allclose(actual, exact, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("eta", [0.0, 0.2, 1.0])
def test_step_uses_paper_ddim_coefficients_and_r_squared_guidance(eta: float) -> None:
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * 0.04)
    op = MatrixOperator(torch.eye(2, dtype=torch.float64))
    y = torch.full((1, 1, 2), 0.1 + 0.05j, dtype=torch.complex128)
    problem = SamplingProblem(model, op, y, op.H(y)[0], COMPLEX, 0.07, 0)
    sampler = PiGDMSampler(model, PiGDMConfig(eta=eta, cg_rtol=1e-12))
    x = torch.full((1, 2, 1, 2), 0.2, dtype=torch.float64)
    epsilon = torch.full_like(x, 0.3)
    sigma, following = 0.4, 0.3
    denoised, guidance = sampler._denoise_and_guidance(x, problem, sigma)
    c = eta * sqrt((sigma**2 - following**2) * following**2 / sigma**2)
    ddim = denoised + sqrt(following**2 - c**2) / sigma * (x - denoised) + c * epsilon
    actual = sampler._step(x, problem, sigma, following, epsilon)
    torch.testing.assert_close(actual, ddim + sampler._variance(sigma) * guidance)
    assert not torch.allclose(actual, ddim + (sigma**2 - following**2) * guidance)
    terminal = sampler._step(x, problem, sigma, 0.0, epsilon)
    torch.testing.assert_close(terminal, denoised + sampler._variance(sigma) * guidance)


def test_noisy_complex_multicoil_guidance_matches_dense_real_gaussian_score() -> None:
    rng = torch.Generator().manual_seed(42)
    csm = torch.view_as_complex(torch.randn(1, 3, 1, 2, 3, 2, dtype=torch.float64, generator=rng))
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float64)
    op = Mask(mask) @ FastFourierOp(dim=(-2, -1)) @ SensitivityOp(csm)
    model = GaussianPrior(torch.eye(6, dtype=torch.float64) * 0.04)
    truth = torch.view_as_complex(torch.randn(1, 1, 1, 2, 3, 2, dtype=torch.float64, generator=rng)) * 0.2
    # Explicit N(0, tau²) for EACH component, rather than complex randn's tau²/2.
    tau = 0.13
    clean = op(truth)[0]
    measurement = clean + tau * torch.view_as_complex(torch.randn(*clean.shape, 2, dtype=torch.float64, generator=rng))
    problem = SamplingProblem(model, op, measurement, truth, COMPLEX, tau, 0)
    x = problem.pack(truth) + 0.1
    sigma = 0.3
    sampler = PiGDMSampler(model, PiGDMConfig(cg_max_iter=100, cg_rtol=1e-11))
    denoised, guidance = sampler._denoise_and_guidance(x, problem, sigma)
    b = dense_forward(problem, denoised)
    gain = 0.04 / (sigma**2 + 0.04)
    c = tau**2 * torch.eye(b.shape[0], dtype=x.dtype) + sigma**2 * gain * b @ b.T
    expected = gain * b.T @ torch.linalg.solve(c, -real_vector(problem.residual(denoised)))
    torch.testing.assert_close(guidance.flatten(), expected, rtol=1e-10, atol=1e-10)
    direct_fourier = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(csm * truth, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    torch.testing.assert_close(clean, mask * direct_fourier)
    # Overcomplete coils and masked entries make BB* singular; tau² I matters.
    assert torch.linalg.matrix_rank(b) < b.shape[0]
    epsilon = problem.noise(torch.Generator().manual_seed(8))
    assert epsilon[:, 1].count_nonzero() > 0


class NonlinearHeads(torch.nn.Module):
    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        main = 0.3 * x.roll(1, dims=-1) + 0.1 * x.square()
        return main, 100 * x


@pytest.mark.parametrize("cg_dtype", [None, torch.float64])
def test_full_main_denoiser_vjp_includes_edm_skip_and_input_derivative(cg_dtype: torch.dtype | None) -> None:
    model = DiffusionModel(unet=NonlinearHeads(), diffusion=DiffusionConfig(cond_dim=8), compile_model=False).eval()
    model.requires_grad_(False)
    op = MatrixOperator(torch.tensor([[1.0, 0.4], [0.0, 0.2]]))
    y = torch.tensor([[[0.1 + 0.2j, -0.1 + 0.3j]]])
    problem = SamplingProblem(model, op, y, op.H(y)[0], COMPLEX, 0.15, 0)
    x = torch.tensor([[[[0.2, -0.1]], [[0.1, 0.3]]]])
    sigma = 0.4
    sampler = PiGDMSampler(model, PiGDMConfig(cg_rtol=1e-6, cg_dtype=cg_dtype))
    denoised, actual = sampler._denoise_and_guidance(x, problem, sigma)
    b = dense_forward(problem, denoised)
    c = 0.15**2 * torch.eye(b.shape[0]) + sampler._variance(sigma) * b @ b.T

    def frozen_log_likelihood(z: Tensor) -> Tensor:
        d = model.denoise(z, z.new_tensor([sigma]), problem.kind)
        r = real_vector(problem.residual(d))
        value: Tensor = -0.5 * r @ torch.linalg.solve(c, r)
        return value

    direction = torch.tensor([[[[0.1, 0.3]], [[-0.2, 0.4]]]])
    difference = (frozen_log_likelihood(x + 1e-3 * direction) - frozen_log_likelihood(x - 1e-3 * direction)) / 2e-3
    torch.testing.assert_close((actual * direction).sum(), difference, atol=2e-5, rtol=2e-3)
    expected = torch.autograd.functional.jacobian(frozen_log_likelihood, x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    detached_denoiser = (b.T @ torch.linalg.solve(c, -real_vector(problem.residual(denoised)))).reshape_as(x)
    assert not torch.allclose(actual, detached_denoiser, rtol=0.1, atol=0.01)
    assert all(p.grad is None for p in model.parameters())


class KnownAmplitudeDecay:
    """Caller composition: one unknown T2 map, fixed M0, linear acquisition."""

    def __init__(self, acquisition: LinearOperator) -> None:
        self.signal = models.MonoExponentialDecay(torch.tensor([0.1, 0.4, 1.0], dtype=torch.float64))
        self.acquisition = acquisition

    def __call__(self, t2: Tensor) -> tuple[Tensor]:
        signal = self.signal(torch.ones_like(t2) * 0.8, t2)[0]
        result: Tensor = self.acquisition(signal)[0]
        return (result,)


def test_nonlinear_extension_is_opt_in_and_uses_frozen_local_covariance() -> None:
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * 0.04, scale=2.0)
    op = KnownAmplitudeDecay(FastFourierOp(dim=(-2, -1)))
    template = torch.tensor([[[0.3, 0.5]], [[0.2, 0.4]]], dtype=torch.float64)
    y = op(template)[0] + 0.01 + 0.02j
    problem = SamplingProblem(model, op, y, template, DatasetKind.T2MAP, 0.05, 1)
    x = problem.encode(template * 0.8)
    with pytest.raises(ValueError, match="local_linearization"):
        PiGDMSampler(model, PiGDMConfig()).sample(
            op,
            y,
            noise_std=0.05,
            template=template,
            kind=DatasetKind.T2MAP,
            measurement_batch_dim=1,
            generator=torch.Generator().manual_seed(5),
        )
    cfg = PiGDMConfig(num_steps=3, sigma_start=0.4, nonlinear_mode="local_linearization", cg_rtol=1e-11)
    sampler = PiGDMSampler(model, cfg)
    sigma = 0.3
    denoised, actual = sampler._denoise_and_guidance(x, problem, sigma)
    # Build both chains' dense system together: batch independence makes it block diagonal.
    b = dense_forward(problem, denoised)
    r = -real_vector(problem.residual(denoised))
    c = 0.05**2 * torch.eye(len(r), dtype=x.dtype) + sampler._variance(sigma) * b @ b.T
    expected = (0.04 / (0.04 + sigma**2)) * b.T @ torch.linalg.solve(c, r)
    torch.testing.assert_close(actual.flatten(), expected, atol=1e-10, rtol=1e-10)
    assert actual[:, 1].count_nonzero() == 0

    def local_log_likelihood(z: Tensor) -> Tensor:
        d = model.denoise(z, z.new_full((len(z),), sigma), problem.kind)
        jacobian = torch.autograd.functional.jacobian(lambda value: real_vector(problem.forward(value)), d, create_graph=True).flatten(1)
        covariance = 0.05**2 * torch.eye(len(r), dtype=z.dtype) + sampler._variance(sigma) * jacobian @ jacobian.T
        residual = real_vector(problem.residual(d))
        value: Tensor = -0.5 * (residual @ torch.linalg.solve(covariance, residual) + torch.linalg.slogdet(covariance).logabsdet)
        return value

    full_derivative = torch.autograd.functional.jacobian(local_log_likelihood, x)
    assert not torch.allclose(actual, full_derivative, atol=1e-3, rtol=1e-3)
    single = SamplingProblem(model, op, y[:, 1:], template[1:], DatasetKind.T2MAP, 0.05, 1)
    _, individual = sampler._denoise_and_guidance(x[1:], single, sigma)
    torch.testing.assert_close(actual[1:], individual, atol=1e-11, rtol=1e-11)
    (draw,) = sampler.sample(
        op,
        y,
        noise_std=0.05,
        template=template,
        kind=DatasetKind.T2MAP,
        measurement_batch_dim=1,
        # Choose a positive start for this decay model's physical domain. The
        # sampler deliberately does not impose a positivity transform or clip.
        noise=torch.ones_like(template),
        generator=torch.Generator().manual_seed(5),
    )
    assert draw.shape == template.shape and draw.dtype == template.dtype
    assert draw.isfinite().all() and not draw.requires_grad


def test_model_coordinate_normalization_measurement_units_and_rng() -> None:
    covariance = torch.eye(2, dtype=torch.float64) * 0.04
    scaled_prior = GaussianPrior(covariance, scale=0.25)
    normalized_prior = GaussianPrior(covariance)
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    y = torch.full((3, 1, 2), 0.15 + 0.05j, dtype=torch.complex128)
    template = torch.empty(3, 1, 2, dtype=torch.float64)
    cfg = PiGDMConfig(num_steps=12, cg_rtol=1e-11)

    def sample(model: GaussianPrior, matrix: Tensor, y: Tensor, tau: float, seed: int = 8) -> Tensor:
        result: Tensor = PiGDMSampler(model, cfg).sample(
            MatrixOperator(matrix),
            y,
            noise_std=tau,
            template=template,
            kind=DatasetKind.T1MAP,
            generator=torch.Generator().manual_seed(seed),
        )[0]
        return result

    state = torch.random.get_rng_state()
    result = sample(scaled_prior, matrix, y, 0.05)
    torch.testing.assert_close(result * 0.25, sample(normalized_prior, matrix / 0.25, y, 0.05), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result, sample(scaled_prior, 10 * matrix, 10 * y, 0.5), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(result, sample(scaled_prior, matrix, y, 0.05), atol=0, rtol=0)
    assert not torch.equal(result, sample(scaled_prior, matrix, y, 0.05, seed=9))
    assert torch.equal(state, torch.random.get_rng_state())


def test_finite_paper_sampler_matches_exact_gaussian_transition_moments() -> None:
    # Analytic finite-method control. Paper Eq. 10's adaptive weight is not a
    # discretization of the exact conditional reverse SDE, even for this prior.
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * 0.04)
    matrix = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    op = MatrixOperator(matrix)
    y = torch.tensor([[[0.12 + 0.04j, 0j]]], dtype=torch.complex128)
    tau = 0.1
    count = 8192
    template = torch.empty(count, 1, 2, dtype=torch.complex128)
    cfg = PiGDMConfig(num_steps=32, eta=0.2, cg_rtol=1e-11)
    (draws,) = PiGDMSampler(model, cfg).sample(
        op, y, noise_std=tau, template=template, kind=COMPLEX, generator=torch.Generator().manual_seed(41)
    )
    mean = torch.zeros(2, dtype=y.dtype)
    variance = torch.full((2,), model.sigma_max**2, dtype=torch.float64)
    grid = karras_sigmas(cfg.num_steps, sigma_min=model.sigma_min, sigma_max=model.sigma_max, rho=cfg.rho, device=model.device).tolist() + [
        0.0
    ]
    for sigma, following in zip(grid[:-1], grid[1:], strict=True):
        gain = 0.04 / (0.04 + sigma**2)
        r2 = sigma**2 * gain
        stochastic = cfg.eta * following * sqrt(1 - (following / sigma) ** 2)
        carry = sqrt(max(0, following**2 - stochastic**2)) / sigma
        precision = matrix.diag() / (tau**2 + r2 * matrix.diag().square())
        transition = gain + carry * (1 - gain) - r2 * gain**2 * precision * matrix.diag()
        mean = transition * mean + r2 * gain * precision * y.flatten()
        variance = transition.square() * variance + stochastic**2
    samples = draws[:, 0]
    for part, target in ((samples.real, mean.real), (samples.imag, mean.imag)):
        assert ((part.mean(0) - target).abs() / (variance / count).sqrt()).max() < 5
        assert ((part.var(0) - variance).abs() / (variance * sqrt(2 / (count - 1)))).max() < 5
    exact_posterior_variance = 1 / (1 / 0.04 + matrix.diag().square() / tau**2)
    # Preserve the distinction between a verified score/transition and posterior calibration.
    assert (variance[0] - exact_posterior_variance[0]).abs() > 0.1 * exact_posterior_variance[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_steps": 1},
        {"cg_max_iter": 0},
        {"rho": 0.0},
        {"cg_rtol": float("nan")},
        {"eta": -0.1},
        {"eta": 1.1},
        {"covariance_std": 0.0},
        {"covariance_std": float("inf")},
        {"cg_dtype": torch.complex64},
        {"nonlinear_mode": "pseudoinverse"},
    ],
)
def test_invalid_approximation_controls(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PiGDMConfig(**kwargs)


def test_inverse_failure_is_not_silently_used() -> None:
    model = GaussianPrior(torch.eye(2, dtype=torch.float64) * 0.04)
    op = MatrixOperator(torch.diag(torch.tensor([1.0, 0.2], dtype=torch.float64)))
    y = torch.full((1, 1, 2), 0.1 + 0.2j, dtype=torch.complex128)
    template = torch.empty_like(y)
    sampler = PiGDMSampler(model, PiGDMConfig(num_steps=3, cg_max_iter=1, cg_rtol=1e-12))
    with pytest.raises(RuntimeError, match="CG did not converge"):
        sampler.sample(op, y, noise_std=0.05, template=template, kind=COMPLEX, generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError, match="per real coordinate"):
        sampler.sample(op, y, noise_std=0.0, template=template, kind=COMPLEX)
