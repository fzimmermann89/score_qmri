"""Tests for DiffusionLightningModule."""

import pytest
import torch

from training.lightning_module import DiffusionLightningModule


@pytest.fixture
def lightning_module():
    """Create Lightning module for testing."""
    try:
        from training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    return DiffusionLightningModule(
        unet_base_channels=16,
        unet_num_blocks=2,
        num_kinds=3,
        num_timesteps=100,
        learning_rate=1e-4,
        validation_image_size=32,
    )


def test_lightning_module_training_step(lightning_module):
    """Test training step computes loss."""
    batch = {
        "image": torch.randn(2, 64, 64),
        "mask": torch.ones(2, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 1]),
    }

    loss = lightning_module.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_lightning_module_configure_optimizers(lightning_module):
    """Test optimizer configuration."""
    opt = lightning_module.configure_optimizers()
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults["lr"] == 1e-4


def test_lightning_module_hparams_saved(lightning_module):
    """Test hyperparameters are saved correctly."""
    assert lightning_module.hparams.unet_base_channels == 16
    assert lightning_module.hparams.unet_num_blocks == 2
    assert lightning_module.hparams.num_kinds == 3
    assert lightning_module.hparams.num_timesteps == 100
    assert lightning_module.hparams.learning_rate == 1e-4


def test_lightning_module_default_params():
    """Test default hyperparameters."""
    try:
        from training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    module = DiffusionLightningModule()
    assert module.hparams.unet_base_channels == 64  # type: ignore[attr-defined]
    assert module.hparams.unet_num_blocks == 4  # type: ignore[attr-defined]
    assert module.hparams.num_kinds == 3  # type: ignore[attr-defined]
    assert module.hparams.num_timesteps == 1000  # type: ignore[attr-defined]


def test_lightning_module_fixed_noise_creation(lightning_module):
    """Test fixed noise is created for each kind."""
    fixed_noise = lightning_module._get_fixed_noise()
    assert len(fixed_noise) == 3
    for kind_idx in range(3):
        assert kind_idx in fixed_noise
        assert fixed_noise[kind_idx].shape == (1, 2, 32, 32)


def test_lightning_module_fixed_noise_reproducible(lightning_module):
    """Test fixed noise is reproducible."""
    noise1 = lightning_module._get_fixed_noise()
    noise2 = lightning_module._get_fixed_noise()
    for kind_idx in range(3):
        assert torch.allclose(noise1[kind_idx], noise2[kind_idx])
