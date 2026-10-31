"""Controls for the explicitly labeled Gauss–Newton proximal DDS adaptation."""

from dataclasses import replace

import pytest
import torch
from mrpro.operators import FastFourierOp, Operator, models
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling.dds_sampler import DDSConfig, DDSSampler, DDSStepDiagnostics

from .test_dds_sampler import COMPLEX_KIND, GaussianPrior


class SquaredSignal(Operator):
    def forward(self, x: Tensor) -> tuple[Tensor]:
        return (x.square(),)


class KnownAmplitudeIR(Operator):
    """One unknown T1 map; fixed complex amplitude belongs to caller physics."""

    def __init__(self) -> None:
        super().__init__()
        self.signal = models.InversionRecovery(torch.tensor([0.1, 0.5, 1.5], dtype=torch.float64))

    def forward(self, t1: Tensor) -> tuple[Tensor]:
        result: Tensor = self.signal(torch.full_like(t1, 1.0 + 0.2j, dtype=torch.complex128), t1)[0]
        return (result,)


def test_linear_mode_rejects_nonlinear_operator() -> None:
    operator = FastFourierOp(dim=(-2, -1)) @ SquaredSignal()
    template = torch.ones(1, 1, 2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="gauss_newton"):
        DDSSampler(GaussianPrior(), DDSConfig()).sample(
            operator,
            operator(template)[0],
            noise_std=0.1,
            template=template,
            kind=COMPLEX_KIND,
        )


@pytest.mark.parametrize("complex_state", [False, True])
def test_one_gauss_newton_step_matches_real_coordinate_jacobian_system(complex_state: bool) -> None:
    operator = FastFourierOp(dim=(-2, -1)) @ SquaredSignal()
    model = GaussianPrior(scale=0.25)
    template = torch.tensor([[[0.9, 1.2]]], dtype=torch.float64)
    if complex_state:
        template = template.to(torch.complex128) + 0.1j
    kind = COMPLEX_KIND if complex_state else DatasetKind.T1MAP
    y = operator(template)[0] + 0.02 + 0.03j
    problem = SamplingProblem(model, operator, y, template, kind, 0.2, 0)
    center = problem.encode(template * 0.8)
    # A small proximal coefficient makes the full local step acceptable, so
    # we can compare it directly to an independently formed dense GN system.
    cfg = DDSConfig(data_consistency="gauss_newton", gauss_newton_steps=1, gamma=1e-4, cg_max_iter=8, cg_rtol=1e-11)
    residual = torch.view_as_real(problem.residual(center)).flatten()
    jacobian = torch.func.jacrev(lambda u: torch.view_as_real(problem.forward(u)).flatten())(center).flatten(1)
    weight = cfg.gamma / 0.2**2
    hessian = torch.eye(center.numel(), dtype=center.dtype) + weight * jacobian.mT @ jacobian
    delta = torch.linalg.solve(hessian, -weight * jacobian.mT @ residual).reshape_as(center)
    result, report = DDSSampler(model, cfg)._data_consistency(center, problem, 0.5)
    assert report.step_sizes[0].tolist() == [1.0]
    torch.testing.assert_close(result, center + delta, rtol=1e-10, atol=1e-12)
    assert (report.objective_after < report.objective_before).all()
    if not complex_state:
        assert torch.count_nonzero(result[:, 1]) == 0


def test_backtracking_checks_nonlinear_objective_per_sample() -> None:
    operator = FastFourierOp(dim=(-2, -1)) @ SquaredSignal()
    model = GaussianPrior()
    template = torch.ones(2, 1, 2, dtype=torch.float64)
    center_physical = torch.tensor([[[0.05, 0.05]], [[0.9, 0.9]]], dtype=torch.float64)
    problem = SamplingProblem(model, operator, operator(template)[0], template, DatasetKind.T1MAP, 0.1, 0)
    center = problem.encode(center_physical)
    cfg = DDSConfig(data_consistency="gauss_newton", gauss_newton_steps=1, gamma=100, cg_rtol=1e-12)
    sampler = DDSSampler(model, cfg)
    result, report = sampler._data_consistency(center, problem, 0.4)
    assert 0 < report.step_sizes[0][0] < 1
    assert report.step_sizes[0][1] == 1
    assert (report.objective_after < report.objective_before).all()
    one = SamplingProblem(model, operator, operator(template[1:])[0], template[1:], DatasetKind.T1MAP, 0.1, 0)
    single, single_report = sampler._data_consistency(center[1:], one, 0.4)
    torch.testing.assert_close(result[1:], single, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(report.objective_after[1:], single_report.objective_after)
    # A deliberately inadequate line-search budget must retain the old point,
    # rather than silently accepting an increase in the nonlinear objective.
    failed, failed_report = DDSSampler(model, replace(cfg, backtracking_steps=1))._data_consistency(center, problem, 0.4)
    assert failed_report.step_sizes[0][0] == 0
    torch.testing.assert_close(failed[0], center[0])


def test_nonlinear_proximal_stationarity_keeps_the_original_denoised_tether() -> None:
    operator = FastFourierOp(dim=(-2, -1)) @ SquaredSignal()
    model = GaussianPrior(scale=0.25)
    template = torch.ones(1, 1, 2, dtype=torch.float64)
    problem = SamplingProblem(model, operator, operator(template)[0], template, DatasetKind.T1MAP, 0.3, 0)
    center = problem.encode(template * 0.8)
    cfg = DDSConfig(data_consistency="gauss_newton", gauss_newton_steps=15, gamma=0.02, cg_rtol=1e-12)
    sampler = DDSSampler(model, cfg)
    result, report = sampler._data_consistency(center, problem, 0.5)
    point = result.detach().requires_grad_(True)
    objective = 0.5 * (point - center).square().sum() + cfg.gamma * problem.residual(point).abs().square().sum() / (2 * 0.3**2)
    gradient = torch.autograd.grad(objective, point)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-8, rtol=0)
    assert (report.objective_after < report.objective_before).all()
    # The proximal optimum is soft even after iterating the nonlinear solve.
    assert problem.residual(result).norm() > 0.01


def test_pinned_ir_signal_fourier_composition_real_map_and_time_axis() -> None:
    operator = FastFourierOp(dim=(-2, -1)) @ KnownAmplitudeIR()
    model = GaussianPrior(scale=0.25)
    template = torch.tensor([[[0.8, 1.1]], [[1.0, 1.3]]], dtype=torch.float64)
    y = operator(template)[0] + 0.02j
    cfg = DDSConfig(num_steps=3, sigma_start=0.1, data_consistency="gauss_newton", gamma=0.02, cg_rtol=1e-11)
    sampler = DDSSampler(model, cfg)
    problem = SamplingProblem(model, operator, y, template, DatasetKind.T1MAP, 0.1, 1)
    # Check the same signal/acquisition boundary without relying on a random
    # diffusion initialization landing in a meaningful T1 basin.
    center = problem.encode(template * 0.9)
    consistent, local = sampler._data_consistency(center, problem, 0.1)
    assert (local.objective_after < local.objective_before).all()
    assert problem.decode(consistent).dtype == template.dtype
    reports: list[DDSStepDiagnostics] = []
    (reconstruction,) = sampler.sample(
        operator,
        y,
        noise_std=0.1,
        template=template,
        kind=DatasetKind.T1MAP,
        measurement_batch_dim=1,
        initial=template,
        noise=torch.zeros_like(template),
        generator=torch.Generator().manual_seed(1),
        callback=reports.append,
    )
    assert reconstruction.isfinite().all() and not reconstruction.requires_grad
    assert reconstruction.shape == template.shape and reconstruction.dtype == template.dtype
    assert all(report.mode == "gauss_newton" for report in reports)
    assert all((report.objective_after <= report.objective_before).all() for report in reports)
