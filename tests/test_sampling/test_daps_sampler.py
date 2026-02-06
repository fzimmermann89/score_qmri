"""Tests for DAPS sampler."""

import pytest
import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel
from score_qmri.models.unet import UNet
from score_qmri.sampling.daps_sampler import DAPSConfig, DAPSSampler
from score_qmri.sampling.protocols import LinearOperatorProtocol


class IdentityOperator:
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x,)

    @property
    def H(self) -> "IdentityOperator":
        return self


class MaskingOperator:
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        k = torch.fft.fft2(x, norm="ortho")
        return (k * self.mask,)

    @property
    def H(self) -> "MaskingOperatorAdjoint":
        return MaskingOperatorAdjoint(self.mask)


class MaskingOperatorAdjoint:
    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def __call__(self, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (torch.fft.ifft2(y * self.mask.conj(), norm="ortho"),)

    @property
    def H(self) -> MaskingOperator:
        return MaskingOperator(self.mask)


def test_daps_config_defaults():
    cfg = DAPSConfig()
    assert cfg.num_steps == 50
    assert cfg.x0_estimator == "pf_ode"
    assert cfg.ode_steps == 5
    assert cfg.mcmc_steps == 5
    assert cfg.mcmc_algo == "langevin"
    assert cfg.step_size == 1e-4
    assert cfg.delta == 0.01
    assert cfg.beta_y == 1.0


def test_daps_config_custom():
    cfg = DAPSConfig(
        num_steps=25,
        x0_estimator="one_step",
        ode_steps=1,
        mcmc_steps=3,
        mcmc_algo="mh",
        step_size=1e-5,
        delta=0.5,
        beta_y=0.7,
    )
    assert cfg.num_steps == 25
    assert cfg.x0_estimator == "one_step"
    assert cfg.ode_steps == 1
    assert cfg.mcmc_steps == 3
    assert cfg.mcmc_algo == "mh"
    assert cfg.step_size == 1e-5
    assert cfg.delta == 0.5
    assert cfg.beta_y == 0.7


def test_identity_operator_protocol():
    op = IdentityOperator()
    assert isinstance(op, LinearOperatorProtocol)
    assert isinstance(op.H, LinearOperatorProtocol)


def test_masking_operator_has_required_methods():
    mask = torch.ones(1, 1, 1, 32, 32)
    op = MaskingOperator(mask)
    assert callable(op)
    assert hasattr(op, "H")
    assert callable(op.H)


def test_daps_sampler_output_shape():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DAPSConfig(num_steps=3, ode_steps=1, x0_estimator="one_step", mcmc_steps=2, beta_y=1.0)
    sampler = DAPSSampler(model, config)

    B, H, W = 2, 32, 32
    op = IdentityOperator()
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE)

    assert result.shape == (B, 1, 1, H, W)
    assert result.dtype == torch.complex64


def test_daps_sampler_with_masking_operator():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DAPSConfig(num_steps=2, ode_steps=1, x0_estimator="one_step", mcmc_steps=2, beta_y=1.0)
    sampler = DAPSSampler(model, config)

    B, H, W = 1, 32, 32
    mask = (torch.rand(1, 1, 1, H, W) > 0.5).float()
    op = MaskingOperator(mask)
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    result = sampler.sample(op, y, DatasetKind.COMPLEX_T2W_COILWISE)

    assert result.shape == (B, 1, 1, H, W)
    assert result.dtype == torch.complex64


def test_daps_sampler_rejects_real_kinds():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    sampler = DAPSSampler(model)

    op = IdentityOperator()
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    with pytest.raises(ValueError, match="only supports complex kinds"):
        sampler.sample(op, y, DatasetKind.T1MAP)


def test_daps_sampler_with_initial_noise():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DAPSConfig(num_steps=2, ode_steps=1, x0_estimator="one_step", mcmc_steps=2)
    sampler = DAPSSampler(model, config)

    B, H, W = 1, 32, 32
    op = IdentityOperator()
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    initial_noise = torch.randn(B, 2, H, W)
    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COIL_COMBINED, initial_noise=initial_noise)

    assert result.shape == (B, 1, 1, H, W)


def test_daps_gradient_flows():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DAPSConfig(num_steps=2, ode_steps=1, x0_estimator="one_step", mcmc_steps=2, beta_y=1.0)
    sampler = DAPSSampler(model, config)

    op = IdentityOperator()
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE)
    assert result is not None


def test_daps_step_size_decays():
    cfg = DAPSConfig(step_size=1e-3, delta=0.1)

    ratio_early = 0.1
    ratio_late = 0.9

    step_early = cfg.step_size * (cfg.delta + ratio_early * (1.0 - cfg.delta))
    step_late = cfg.step_size * (cfg.delta + ratio_late * (1.0 - cfg.delta))

    assert step_early < step_late
    assert step_late > 0


def test_daps_sampler_runs_mcmc_steps():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config_1 = DAPSConfig(num_steps=2, ode_steps=1, x0_estimator="one_step", mcmc_steps=1, beta_y=1.0)
    config_5 = DAPSConfig(num_steps=2, ode_steps=1, x0_estimator="one_step", mcmc_steps=5, beta_y=1.0)

    sampler_1 = DAPSSampler(model, config_1)
    sampler_5 = DAPSSampler(model, config_5)

    op = IdentityOperator()
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)
    noise = torch.randn(1, 2, 32, 32)

    result_1 = sampler_1.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE, initial_noise=noise.clone())
    result_5 = sampler_5.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE, initial_noise=noise.clone())

    assert result_1.shape == result_5.shape
    assert not torch.allclose(result_1, result_5)
