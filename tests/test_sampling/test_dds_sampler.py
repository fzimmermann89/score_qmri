"""Exact quadratic and finite Gaussian controls for noisy linear DDS."""

from dataclasses import replace

import pytest
import torch
from mrpro.operators import EinsumOp, FastFourierOp, SensitivityOp
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.models.unet import UNet
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling.dds_sampler import DDSConfig, DDSSampler, DDSStepDiagnostics, _truncated_cg
from score_qmri.sampling.edm_sampler import karras_sigmas

COMPLEX_KIND = DatasetKind.COMPLEX_T1W_COIL_COMBINED


class GaussianPrior:
    """Exact denoiser for independent real/imag Gaussian vectors, in model units."""

    device = torch.device("cpu")
    sigma_min = 0.05
    sigma_max = 1.0

    def __init__(self, covariance: Tensor | None = None, scale: float = 1.0) -> None:
        self.covariance = torch.tensor([[0.08, 0.02], [0.02, 0.04]], dtype=torch.float64) if covariance is None else covariance
        self.scale = scale
        self.calls: list[Tensor] = []

    def normalize(self, x: Tensor, kind: Tensor) -> Tensor:
        return x * self.scale

    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor:
        return x / self.scale

    def denoise(self, x: Tensor, sigma: Tensor, kind: Tensor) -> Tensor:
        assert not torch.is_grad_enabled()
        assert (sigma >= self.sigma_min).all() and (sigma <= self.sigma_max).all()
        self.calls.append(sigma.clone())
        covariance = self.covariance.to(x)
        gain = torch.linalg.solve(covariance + sigma[:, None, None] ** 2 * torch.eye(2, dtype=x.dtype), covariance.expand(len(x), -1, -1))
        return torch.einsum("bij,bchj->bchi", gain, x)


def test_cg_truncates_from_warm_start_with_independent_batch_diagnostics() -> None:
    matrix = torch.diag(torch.tensor([1.0, 3.0, 12.0], dtype=torch.float64))
    initial = torch.tensor([[0.2, 0.8, -0.1], [0.0, 0.0, 0.0]], dtype=torch.float64)
    rhs = torch.tensor([[1.0, -0.5, 0.2], [0.0, 0.0, 0.0]], dtype=torch.float64)
    actual, report = _truncated_cg(lambda x: x @ matrix, initial, rhs, 1, 1e-12)
    r = rhs[0] - initial[0] @ matrix
    expected = initial[0] + r * (r @ r) / (r @ matrix @ r)
    torch.testing.assert_close(actual[0], expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual[1], initial[1])
    assert report.iterations.tolist() == [1, 0]
    assert report.converged.tolist() == [False, True]
    torch.testing.assert_close(report.residual_norm, (rhs - actual @ matrix).norm(dim=1))
    single, _ = _truncated_cg(lambda x: x @ matrix, initial[:1], rhs[:1], 1, 1e-12)
    torch.testing.assert_close(actual[:1], single, rtol=0, atol=0)
    solved, converged = _truncated_cg(lambda x: x @ matrix, initial, rhs, 5, 1e-11)
    torch.testing.assert_close(solved, torch.linalg.solve(matrix, rhs.mT).mT)
    assert converged.converged.all()
    assert converged.iterations.tolist() == [3, 0]
    # The local helper must neither alter inputs nor build an implicit backward.
    assert initial[0, 0] == 0.2 and not solved.requires_grad


@pytest.mark.parametrize("bad", ["negative", "nonfinite"])
def test_cg_rejects_invalid_physics(bad: str) -> None:
    def matvec(x: Tensor) -> Tensor:
        return -x if bad == "negative" else x * float("nan")

    with pytest.raises(RuntimeError, match="curvature|Nonfinite"):
        _truncated_cg(matvec, torch.zeros(2, 3), torch.ones(2, 3), 2, 1e-6)


@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_noisy_complex_proximal_update_matches_dense_solution(scale: float) -> None:
    model = GaussianPrior(scale=scale)
    matrix = torch.tensor([[1.0 + 0.2j, 0.4j], [0.0, 0.3 - 0.1j]], dtype=torch.complex128)
    op = EinsumOp(matrix, "i j, ... j -> ... i")
    template = torch.zeros(2, 1, 2, dtype=torch.complex128)
    y = torch.tensor([[[0.2 + 0.1j, -0.1 + 0.3j]], [[0.7j, 0.2 - 0.1j]]], dtype=template.dtype)
    problem = SamplingProblem(model, op, y, template, COMPLEX_KIND, 0.3, 0)
    center = torch.tensor([[[[0.1, 0.3]], [[-0.2, 0.4]]]], dtype=torch.float64).expand(2, -1, -1, -1).clone()
    cfg = DDSConfig(gamma=0.17, cg_max_iter=5, cg_rtol=1e-11)
    consistent, report = DDSSampler(model, cfg)._data_consistency(center, problem, 0.5)
    linear = matrix / scale
    weight = cfg.gamma / problem.noise_std**2
    precision = torch.eye(2, dtype=matrix.dtype) + weight * linear.mH @ linear
    d = torch.complex(center[:, 0, 0], center[:, 1, 0])
    rhs = d + weight * y[:, 0] @ linear.conj()
    expected = torch.linalg.solve(precision, rhs.mT).mT
    torch.testing.assert_close(torch.complex(consistent[:, 0, 0], consistent[:, 1, 0]), expected, rtol=1e-11, atol=1e-12)
    assert report.cg[0].converged.all()
    assert (report.objective_after < report.objective_before).all()
    # Check stationarity of the actual physical Gaussian likelihood plus tether.
    point = consistent.detach().requires_grad_(True)
    objective = 0.5 * (point - center).square().sum() + 0.5 * cfg.gamma * problem.residual(point).abs().square().sum() / 0.3**2
    gradient = torch.autograd.grad(objective, point)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(point), atol=1e-11, rtol=0)
    # No hard replacement of noisy measured coordinates.
    assert problem.residual(consistent).norm() > 0.01


def test_rank_deficient_fourier_update_preserves_nullspace_and_softens_noisy_data() -> None:
    model = GaussianPrior()
    fft = FastFourierOp(dim=(-2, -1))
    mask = torch.tensor([[[[[1.0, 0.0]]]]], dtype=torch.float64)
    op = mask * fft
    center_physical = torch.tensor([[[[[0.1 + 0.2j, 0.3 - 0.1j]]]]], dtype=torch.complex128)
    y = torch.tensor([[[[[0.6 - 0.2j, 0.0]]]]], dtype=center_physical.dtype)
    problem = SamplingProblem(model, op, y, center_physical, COMPLEX_KIND, 0.4, 0)
    cfg = DDSConfig(gamma=0.3)
    actual, _ = DDSSampler(model, cfg)._data_consistency(problem.encode(center_physical), problem, 0.5)
    observed = fft(problem.decode(actual))[0]
    denoised = fft(center_physical)[0]
    w = cfg.gamma / 0.4**2
    expected = (denoised + w * y) / (1 + w * mask)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(observed[..., 1], denoised[..., 1])
    assert (observed[..., 0] - y[..., 0]).abs().min() > 0.01
    centered = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(center_physical, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))
    torch.testing.assert_close(denoised, centered)


def test_pinned_multicoil_composition_matches_dense_normal_equations() -> None:
    fft = FastFourierOp(dim=(-2, -1))
    csm = torch.tensor([1.0, 0.5j, 0.4 - 0.2j, 0.8], dtype=torch.complex128).reshape(1, 2, 1, 1, 2)
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64).reshape(1, 2, 1, 1, 2)
    op = mask * (fft @ SensitivityOp(csm))
    basis = torch.eye(2, dtype=torch.complex128).reshape(2, 1, 1, 1, 2)
    matrix = op(basis)[0].flatten(1).mT
    model = GaussianPrior()
    template = torch.zeros(1, 1, 1, 1, 2, dtype=torch.complex128)
    y = torch.tensor([0.2 + 0.1j, 0.0, -0.1j, 0.15], dtype=template.dtype).reshape(1, 2, 1, 1, 2)
    center = torch.tensor([[[[0.2, -0.1]], [[0.3, 0.4]]]], dtype=torch.float64)
    cfg = DDSConfig(gamma=0.2, cg_max_iter=4, cg_rtol=1e-11)
    problem = SamplingProblem(model, op, y, template, COMPLEX_KIND, 0.1, 0)
    actual, _ = DDSSampler(model, cfg)._data_consistency(center, problem, 0.5)
    expected = torch.linalg.solve(
        torch.eye(2, dtype=matrix.dtype) + 20 * matrix.mH @ matrix,
        torch.complex(center[0, 0, 0], center[0, 1, 0]) + 20 * matrix.mH @ y.flatten(),
    )
    torch.testing.assert_close(problem.decode(actual).flatten(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_transition_is_released_vp_update_in_ve_coordinates(eta: float) -> None:
    x = torch.tensor([0.5, -0.2], dtype=torch.float64)
    center, consistent, noise = x * 0.4, x * 0.3 + 0.1, x * 2
    sigma, following = 0.8, 0.3
    alpha = 1 / (1 + sigma**2)
    alpha_next = 1 / (1 + following**2)
    x_vp = alpha**0.5 * x
    epsilon = (x_vp - alpha**0.5 * center) / (1 - alpha) ** 0.5
    # Released solver_2d.py c1/c2 coefficients; no standard DDIM ratio law.
    vp_next = alpha_next**0.5 * consistent + (1 - alpha_next) ** 0.5 * (eta * noise + (1 - eta**2) ** 0.5 * epsilon)
    actual = DDSSampler._transition(x, center, consistent, sigma, following, eta, noise)
    torch.testing.assert_close(actual, vp_next / alpha_next**0.5, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(DDSSampler._transition(x, center, consistent, sigma, 0.0, eta, noise), consistent)
    if eta == 0:
        # Without DC this is Euler probability flow; it is not a posterior draw.
        torch.testing.assert_close(
            DDSSampler._transition(x, center, center, sigma, following, eta, noise), x + (following - sigma) * (x - center) / sigma
        )


def test_gaussian_finite_trajectory_mean_covariance_and_endpoint() -> None:
    model = GaussianPrior()
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    op = EinsumOp(matrix.to(torch.complex128), "i j, ... j -> ... i")
    y = torch.tensor([0.15 + 0.05j, 0.02 - 0.01j], dtype=torch.complex128)
    cfg = DDSConfig(num_steps=6, eta=0.65, gamma=0.007, cg_rtol=1e-11)
    tau = 0.2
    eye = torch.eye(2, dtype=torch.float64)
    precision = eye + cfg.gamma / tau**2 * matrix.mT @ matrix
    proximal_gain = torch.linalg.inv(precision)
    offset = (proximal_gain @ matrix.mT).to(y) @ y * cfg.gamma / tau**2
    grid = karras_sigmas(cfg.num_steps, sigma_min=model.sigma_min, sigma_max=model.sigma_max, rho=cfg.rho, device=model.device).double()
    grid[0], grid[-1] = model.sigma_max, model.sigma_min
    mean = torch.zeros_like(y)
    covariance = model.sigma_max**2 * eye
    for sigma, following in zip(grid[:-1], grid[1:], strict=True):
        denoiser = torch.linalg.solve(model.covariance + sigma**2 * eye, model.covariance)
        transition = proximal_gain @ denoiser + following / sigma * (1 - cfg.eta**2) ** 0.5 * (eye - denoiser)
        mean = transition.to(mean) @ mean + offset
        covariance = transition @ covariance @ transition.mT + (following * cfg.eta) ** 2 * eye
    final_gain = torch.linalg.solve(model.covariance + model.sigma_min**2 * eye, model.covariance)
    mean = final_gain.to(mean) @ mean
    covariance = final_gain @ covariance @ final_gain.mT
    reports: list[DDSStepDiagnostics] = []
    (recon,) = DDSSampler(model, cfg).sample(
        op,
        y.reshape(1, 1, 2),
        noise_std=tau,
        template=torch.zeros(8192, 1, 2, dtype=y.dtype),
        kind=COMPLEX_KIND,
        generator=torch.Generator().manual_seed(71),
        callback=reports.append,
    )
    assert len(model.calls) == cfg.num_steps and len(reports) == cfg.num_steps - 1
    assert model.calls[-1][0] == model.sigma_min
    values = recon[:, 0]
    for part, target in [(values.real, mean.real), (values.imag, mean.imag)]:
        mean_se = (covariance.diag() / len(values)).sqrt()
        covariance_se = ((covariance.diag()[:, None] * covariance.diag()[None, :] + covariance.square()) / (len(values) - 1)).sqrt()
        assert ((part.mean(0) - target).abs() / mean_se).max() < 5
        assert ((torch.cov(part.mT) - covariance).abs() / covariance_se).max() < 5
    # This finite DDS distribution is not the known Gaussian posterior.
    posterior_covariance = torch.linalg.inv(torch.linalg.inv(model.covariance) + matrix.mT @ matrix / tau**2)
    assert (posterior_covariance - covariance).norm() > 0.01
    assert not recon.requires_grad


def test_likelihood_units_rng_and_batch_independence() -> None:
    matrix = torch.tensor([[1.0, 0.4], [0.0, 0.2]], dtype=torch.float64)
    y = torch.tensor([[[0.2 + 0.1j, -0.1j]], [[0.1 - 0.1j, 0.05j]]], dtype=torch.complex128)
    initial_noise = torch.tensor([[[0.4j, -0.1]], [[0.1, -0.2j]]], dtype=y.dtype)
    cfg = DDSConfig(num_steps=4, cg_max_iter=1, cg_rtol=1e-11)

    def run(factor: float = 1.0, seed: int = 7) -> Tensor:
        return DDSSampler(GaussianPrior(), cfg).sample(
            EinsumOp((matrix * factor).to(y), "i j, ... j -> ... i"),
            y * factor,
            noise_std=0.2 * factor,
            template=y,
            kind=COMPLEX_KIND,
            noise=initial_noise,
            generator=torch.Generator().manual_seed(seed),
        )[0]

    first = run()
    torch.testing.assert_close(first, run(), rtol=0, atol=0)
    torch.testing.assert_close(first, run(factor=10), rtol=1e-12, atol=1e-12)
    assert not torch.equal(first, run(seed=8))
    deterministic = DDSSampler(GaussianPrior(), replace(cfg, eta=0))
    global_state = torch.random.get_rng_state().clone()
    batch = deterministic.sample(
        EinsumOp(matrix.to(y), "i j, ... j -> ... i"), y, noise_std=0.2, template=y, kind=COMPLEX_KIND, noise=initial_noise
    )[0]
    single = deterministic.sample(
        EinsumOp(matrix.to(y), "i j, ... j -> ... i"), y[1:], noise_std=0.2, template=y[1:], kind=COMPLEX_KIND, noise=initial_noise[1:]
    )[0]
    torch.testing.assert_close(batch[1:], single, rtol=1e-12, atol=1e-12)
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_real_diffusion_model_public_scaling_roundtrip() -> None:
    # Exercise the actual public model API rather than a case-specific map scale.
    model = DiffusionModel(
        unet=UNet(base_channels=8, num_res_blocks=1, channel_mult=(1,), attention_levels=(), cond_dim=128),
        compile_model=False,
    ).eval()
    template = torch.ones(1, 1, 2, dtype=torch.float32) * 700
    op = FastFourierOp(dim=(-2, -1))
    problem = SamplingProblem(model, op, op(template)[0], template, DatasetKind.T1MAP, 0.1, 0)
    physical_2ch = torch.stack([template[:, 0], torch.zeros_like(template[:, 0])], 1).unsqueeze(2)
    torch.testing.assert_close(problem.encode(template), model.normalize(physical_2ch, problem.kind))
    torch.testing.assert_close(problem.decode(problem.encode(template)), template)
    (answer,) = DDSSampler(model, DDSConfig(num_steps=2, sigma_start=0.1)).sample(
        op,
        op(template)[0],
        noise_std=100,
        template=template,
        kind=DatasetKind.T1MAP,
        generator=torch.Generator().manual_seed(5),
    )
    assert answer.isfinite().all() and answer.dtype == template.dtype


@pytest.mark.parametrize(
    "field,value",
    [
        ("eta", -0.1),
        ("eta", 1.1),
        ("eta", float("nan")),
        ("gamma", -1.0),
        ("gamma", float("inf")),
        ("cg_max_iter", 0),
        ("cg_rtol", 1.0),
        ("num_steps", 1),
        ("sigma_start", float("nan")),
        ("gauss_newton_steps", 0),
        ("data_consistency", "automatic"),
    ],
)
def test_invalid_config(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        DDSConfig(**{field: value})


@pytest.mark.parametrize("tau", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_noise_sd(tau: float) -> None:
    y = torch.zeros(1, 1, 2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="noise_std"):
        DDSSampler(GaussianPrior(), DDSConfig(num_steps=2)).sample(
            FastFourierOp(dim=(-2, -1)),
            y,
            noise_std=tau,
            template=y,
            kind=COMPLEX_KIND,
        )


def test_invalid_schedule_and_nonfinite_measurements() -> None:
    y = torch.zeros(1, 1, 2, dtype=torch.complex128)
    for start in (0.01, 2.0):
        with pytest.raises(ValueError, match="schedule"):
            DDSSampler(GaussianPrior(), DDSConfig(sigma_start=start)).sample(
                FastFourierOp(dim=(-2, -1)),
                y,
                noise_std=0.1,
                template=y,
                kind=COMPLEX_KIND,
            )
    with pytest.raises(ValueError, match="measurement must be finite"):
        DDSSampler(GaussianPrior(), DDSConfig()).sample(
            FastFourierOp(dim=(-2, -1)),
            y + float("nan"),
            noise_std=0.1,
            template=y,
            kind=COMPLEX_KIND,
        )
