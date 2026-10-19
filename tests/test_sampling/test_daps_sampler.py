"""Tests for DAPS sampler."""

from typing import cast

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
    config = DAPSConfig(anneal_steps=3, reverse_steps=1, langevin_steps=2, tau=1.0, inner_kernel="gaussian")
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
    config = DAPSConfig(anneal_steps=2, reverse_steps=1, langevin_steps=2, tau=1.0)
    sampler = DAPSSampler(model, config)

    B, H, W = 1, 32, 32
    mask = (torch.rand(1, 1, 1, H, W) > 0.5).float()
    op = cast(LinearOperatorProtocol, MaskingOperator(mask))
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

    with pytest.raises(NotImplementedError, match="only supports complex kinds"):
        sampler.sample(op, y, DatasetKind.T1MAP)


def test_daps_sampler_with_noise():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DAPSConfig(anneal_steps=2, reverse_steps=1, langevin_steps=2)
    sampler = DAPSSampler(model, config)

    B, H, W = 1, 32, 32
    op = IdentityOperator()
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    noise = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)
    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COIL_COMBINED, noise=noise)

    assert result.shape == (B, 1, 1, H, W)


def test_daps_gradient_flows():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DAPSConfig(anneal_steps=2, reverse_steps=1, langevin_steps=2, tau=1.0)
    sampler = DAPSSampler(model, config)

    op = IdentityOperator()
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE)
    assert result is not None


def test_daps_lr_decays():
    """_lr_at decays from langevin_lr to langevin_lr * lr_min_ratio."""
    cfg = DAPSConfig(langevin_lr=1e-3, lr_min_ratio=0.1)
    sampler = DAPSSampler.__new__(DAPSSampler)
    sampler.cfg = cfg

    lr_early = sampler._lr_at(0.0)
    lr_late = sampler._lr_at(1.0)

    assert lr_early == pytest.approx(cfg.langevin_lr)
    assert lr_late == pytest.approx(cfg.langevin_lr * cfg.lr_min_ratio)
    assert lr_early > lr_late > 0


def test_daps_sampler_runs_langevin_steps():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config_1 = DAPSConfig(anneal_steps=2, reverse_steps=1, langevin_steps=1, tau=1.0, inner_kernel="langevin")
    config_5 = DAPSConfig(anneal_steps=2, reverse_steps=1, langevin_steps=5, tau=1.0, inner_kernel="langevin")

    sampler_1 = DAPSSampler(model, config_1)
    sampler_5 = DAPSSampler(model, config_5)

    op = IdentityOperator()
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)
    noise = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    result_1 = sampler_1.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE, noise=noise.clone())
    result_5 = sampler_5.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE, noise=noise.clone())

    assert result_1.shape == result_5.shape
    assert not torch.allclose(result_1, result_5)


def test_langevin_likelihood_gradient_uses_per_real_noise_variance():
    sampler = DAPSSampler.__new__(DAPSSampler)
    sampler.cfg = DAPSConfig(inner_kernel="langevin", langevin_steps=1, langevin_lr=1.0, tau=0.5)
    center = torch.full((1, 2, 1, 1), 0.4)
    measurement = torch.full((1, 1, 1, 1, 1), 0.1 + 0.2j)
    sigma = 0.5
    lr = 0.1 / (sigma**-2 + sampler.cfg.tau**-2)
    kind = torch.tensor([int(DatasetKind.COMPLEX_T1W_COILWISE)])
    result = sampler._langevin(center, sigma, 0, kind, IdentityOperator(), measurement, torch.Generator().manual_seed(43))
    noise = torch.randn(center.shape, generator=torch.Generator().manual_seed(43))
    residual = center - torch.tensor([0.1, 0.2]).reshape(1, 2, 1, 1)
    expected = center - lr * residual / sampler.cfg.tau**2 + (2 * lr) ** 0.5 * noise
    torch.testing.assert_close(result, expected)
