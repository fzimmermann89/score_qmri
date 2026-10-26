"""Real coordinates, physical scaling and nonlinear measurement derivatives."""

import pytest
import torch
from mrpro.operators import FastFourierOp, models

from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler, FullScoreDAPSConfig, FullScoreDAPSSampler
from score_qmri.sampling._problem import SamplingProblem

from .test_dps_sampler import GaussianPrior


class MapPrior(GaussianPrior):
    sigma_max = 1.0

    def normalize(self, x, kind):
        return x * 0.25

    def denormalize(self, x, kind):
        return x * 4


class KnownM0IR:
    def __init__(self):
        self.ir = models.InversionRecovery(torch.tensor([0.1, 0.5, 1.5], dtype=torch.float64))

    def __call__(self, t1):
        return self.ir(torch.ones_like(t1), t1)


def test_ir_gradient_and_jacobian_include_scaling_and_time_axis():
    model, operator = MapPrior(), KnownM0IR()
    template = torch.tensor([[[1.0, 1.3]], [[0.7, 1.7]]], dtype=torch.float64)
    y = operator(template)[0].to(torch.complex128) + 0.03 + 0.02j
    problem = SamplingProblem(model, operator, y, template, 6, 0.1, 1)
    point = problem.encode(template).requires_grad_(True)
    assert problem.forward(point.float()).dtype == torch.complex64
    loss = problem.residual(point).abs().square().sum() / (2 * 0.1**2)
    expected = torch.autograd.grad(loss, point)[0]
    torch.testing.assert_close(problem.gradient(point), expected)
    jacobian, adjoint = problem.linearize(point)
    tangent = torch.ones_like(point)
    cotangent = torch.full_like(problem.measurement, 0.2 + 0.1j)
    torch.testing.assert_close((jacobian(tangent).conj() * cotangent).real.sum(), (tangent * adjoint(cotangent)).sum())
    finite_difference = (problem.forward(point + 1e-5 * tangent) - problem.forward(point - 1e-5 * tangent)) / 2e-5
    torch.testing.assert_close(jacobian(tangent), finite_difference, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(problem.decode(point), template)
    assert torch.count_nonzero(problem.noise(torch.Generator().manual_seed(4))[:, 1]) == 0


def test_real_linear_pullback_matches_autograd():
    model = MapPrior()
    operator = FastFourierOp(dim=(-2, -1))
    template = torch.ones(2, 1, 2, dtype=torch.float64)
    y = operator(template)[0] + 0.02j
    problem = SamplingProblem(model, operator, y, template, 6, 0.1, 0)
    x = problem.encode(template * 0.8).requires_grad_(True)
    objective = problem.residual(x).abs().square().sum() / (2 * 0.1**2)
    gradient = torch.autograd.grad(objective, x)[0]
    torch.testing.assert_close(problem.gradient(x), gradient)


class NonlinearEncoding:
    def __call__(self, x):
        return (torch.fft.fft2(x.square(), norm="ortho").unsqueeze(0),)


@pytest.mark.parametrize(
    "sampler_type,cfg",
    [
        (DPSSampler, DPSConfig(num_steps=3, guidance="dps", guidance_scale=0.01)),
        (DPSSampler, DPSConfig(num_steps=3, cg_rtol=1e-8)),
        (DAPSSampler, DAPSConfig(num_steps=3, reverse_steps=2, langevin_steps=2)),
        (FullScoreDAPSSampler, FullScoreDAPSConfig(num_steps=3, reverse_steps=2, langevin_steps=2, cg_rtol=1e-8)),
    ],
)
def test_real_nonlinear_sampling_returns_real_maps(sampler_type, cfg):
    model, operator = MapPrior(), NonlinearEncoding()
    template = torch.ones(2, 1, 2, dtype=torch.float64)
    measurement = operator(template)[0]
    sampler = sampler_type(model, cfg)
    (batched,) = sampler.sample(
        operator,
        measurement,
        noise_std=0.1,
        template=template,
        kind=6,
        measurement_batch_dim=1,
        generator=torch.Generator().manual_seed(4),
    )
    assert batched.shape == template.shape and batched.dtype == template.dtype
    assert batched.isfinite().all() and not batched.requires_grad


def test_nonlinear_dps_step_keeps_independent_batches():
    model, operator = MapPrior(), NonlinearEncoding()
    template = torch.tensor([[[1.0, 1.3]], [[0.7, 1.7]]], dtype=torch.float64)
    measurement = operator(template)[0]
    problem = SamplingProblem(model, operator, measurement, template, 6, 0.1, 1)
    single = SamplingProblem(model, operator, measurement[:, 1:], template[1:], 6, 0.1, 1)
    x = problem.encode(template * 0.8)
    sampler = DPSSampler(model, DPSConfig(cg_rtol=1e-12))
    batched = sampler._step(x, problem, 0.4, 0.3, torch.zeros_like(x))
    individual = sampler._step(x[1:], single, 0.4, 0.3, torch.zeros_like(x[1:]))
    torch.testing.assert_close(batched[1:], individual, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype,cg_dtype", [(torch.float64, None), (torch.float32, torch.float64)])
def test_nonlinear_dps_correction_matches_dense_covariance(dtype, cg_dtype):
    model, operator = MapPrior(), NonlinearEncoding()
    template = torch.tensor([[[1.0, 1.3]]], dtype=dtype)
    problem = SamplingProblem(model, operator, operator(template)[0], template, 6, 0.1, 1)
    x = problem.encode(template * 0.8)
    sigma, following = 0.4, 0.3
    denoised = model.denoise(x, x.new_tensor([sigma]), problem.kind)
    residual = torch.view_as_real(problem.residual(denoised)).flatten()
    jacobian = torch.func.jacrev(lambda z: torch.view_as_real(problem.forward(z)).flatten())(denoised).flatten(1)
    gain = 0.04 / (0.04 + sigma**2)
    variance = sigma**2 * gain
    covariance = 0.1**2 * torch.eye(len(residual), dtype=x.dtype) + variance * jacobian @ jacobian.T
    gradient = gain * (jacobian.T @ torch.linalg.solve(covariance, residual)).reshape_as(x)
    ratio = (following / sigma) ** 2
    expected = ratio * x + (1 - ratio) * denoised - (sigma**2 - following**2) * gradient
    actual = DPSSampler(model, DPSConfig(cg_rtol=1e-12, cg_dtype=cg_dtype))._step(x, problem, sigma, following, torch.zeros_like(x))
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
