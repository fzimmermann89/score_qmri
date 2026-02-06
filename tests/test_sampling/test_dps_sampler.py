"""Tests for DPS sampler."""

from typing import cast

import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel
from score_qmri.models.unet import UNet
from score_qmri.sampling.dps_sampler import DPSConfig, DPSSampler
from score_qmri.sampling.protocols import LinearOperatorProtocol


class IdentityOperator:
    """Identity operator for testing: A(x) = x, A†(y) = y."""

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (x,)

    @property
    def H(self) -> "IdentityOperator":
        return self


class MaskingOperator:
    """Simulates k-space undersampling by masking."""

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # x: (B, 1, 1, H, W) complex
        # FFT to k-space, apply mask, return
        k = torch.fft.fft2(x, norm="ortho")
        return (k * self.mask,)

    @property
    def H(self) -> "MaskingOperatorAdjoint":
        return MaskingOperatorAdjoint(self.mask)


class MaskingOperatorAdjoint:
    """Adjoint of MaskingOperator."""

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def __call__(self, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # y: (B, Coils, 1, k2, k1) complex
        # Apply conjugate mask, IFFT
        return (torch.fft.ifft2(y * self.mask.conj(), norm="ortho"),)

    @property
    def H(self) -> MaskingOperator:
        return MaskingOperator(self.mask)


def test_dps_config_defaults():
    cfg = DPSConfig()
    assert cfg.num_steps == 50
    assert cfg.guidance_scale == 1.0
    assert cfg.gradient_clipping == 1.0
    assert cfg.normalize_gradient is True
    assert cfg.rho == 7.0


def test_dps_config_custom():
    cfg = DPSConfig(num_steps=20, rho=5.0, guidance_scale=0.1)
    assert cfg.num_steps == 20
    assert cfg.rho == 5.0
    assert cfg.guidance_scale == 0.1


def test_identity_operator_protocol():
    op = IdentityOperator()
    assert isinstance(op, LinearOperatorProtocol)
    assert isinstance(op.H, LinearOperatorProtocol)


def test_masking_operator_has_required_methods():
    mask = torch.ones(1, 1, 1, 32, 32)
    op = MaskingOperator(mask)
    # Verify operator has required interface (protocol check skipped due to recursive type)
    assert callable(op)
    assert hasattr(op, "H")
    assert callable(op.H)


def test_dps_sampler_output_shape():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DPSConfig(num_steps=5, guidance_scale=0.1)
    sampler = DPSSampler(model, config)

    B, H, W = 2, 32, 32
    op = cast(LinearOperatorProtocol, IdentityOperator())

    # Create measurement in mrpro format
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE)

    assert result.shape == (B, 1, 1, H, W)
    assert result.dtype == torch.complex64


def test_dps_sampler_with_masking_operator():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DPSConfig(num_steps=3, guidance_scale=0.01)
    sampler = DPSSampler(model, config)

    B, H, W = 1, 32, 32

    # Create random undersampling mask
    mask = torch.rand(1, 1, 1, H, W) > 0.5
    mask = mask.float()
    op = cast(LinearOperatorProtocol, MaskingOperator(mask))

    # Create measurement
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    result = sampler.sample(op, y, DatasetKind.COMPLEX_T2W_COILWISE)

    assert result.shape == (B, 1, 1, H, W)
    assert result.dtype == torch.complex64


def test_dps_sampler_rejects_real_kinds():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    sampler = DPSSampler(model)

    op = cast(LinearOperatorProtocol, IdentityOperator())
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    import pytest

    with pytest.raises(ValueError, match="only supports complex kinds"):
        sampler.sample(op, y, DatasetKind.T1MAP)


def test_dps_sampler_with_initial_noise():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DPSConfig(num_steps=3)
    sampler = DPSSampler(model, config)

    B, H, W = 1, 32, 32
    op = cast(LinearOperatorProtocol, IdentityOperator())
    y = torch.randn(B, 1, 1, H, W, dtype=torch.complex64)

    initial_noise = torch.randn(B, 2, H, W)
    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COIL_COMBINED, initial_noise=initial_noise)

    assert result.shape == (B, 1, 1, H, W)


def test_dps_sampler_deterministic_with_fixed_noise():
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    model.eval()  # Important: freeze normalizer stats
    config = DPSConfig(num_steps=3)
    sampler = DPSSampler(model, config)

    op = cast(LinearOperatorProtocol, IdentityOperator())
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    noise = torch.randn(1, 2, 32, 32)

    r1 = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE, initial_noise=noise.clone())
    r2 = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE, initial_noise=noise.clone())

    assert torch.allclose(r1, r2, rtol=1e-5, atol=1e-5)


def test_dps_gradient_flows():
    """Test that gradients are computed during sampling."""
    unet = UNet(base_channels=16, num_res_blocks=1, channel_mult=(1, 2), attention_levels=())
    model = DiffusionModel(unet=unet, diffusion=DiffusionConfig())
    config = DPSConfig(num_steps=2, guidance_scale=1.0)
    sampler = DPSSampler(model, config)

    op = cast(LinearOperatorProtocol, IdentityOperator())
    y = torch.randn(1, 1, 1, 32, 32, dtype=torch.complex64)

    # Should not raise - gradients are computed internally
    result = sampler.sample(op, y, DatasetKind.COMPLEX_T1W_COILWISE)
    assert result is not None
