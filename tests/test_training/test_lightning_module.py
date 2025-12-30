"""Tests for DiffusionLightningModule."""

import pytest
import torch

from training.lightning_module import DiffusionLightningModule


@pytest.fixture
def module():
    """Create small module for testing."""
    try:
        from training.losses import ssim_loss

        ssim_loss(torch.randn(1, 2, 64, 64), torch.randn(1, 2, 64, 64))
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    return DiffusionLightningModule(
        base_channels=16,
        num_blocks=2,
        cond_dim=64,
        validation_size=32,
    )


def test_training_step(module):
    """Test training step computes loss."""
    batch = {
        "image": torch.randn(2, 64, 64),
        "mask": torch.ones(2, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 1]),
    }
    loss = module.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_configure_optimizers(module):
    """Test optimizer configuration."""
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)


def test_hparams_saved(module):
    """Test hyperparameters are saved."""
    assert module.hparams.base_channels == 16
    assert module.hparams.num_blocks == 2
    assert module.model.num_kinds > 0


def test_fixed_noise_creation(module):
    """Test fixed noise is created for each kind."""
    module._fixed_noise = None  # Reset
    module._log_samples()  # This won't log (no WandB) but creates noise
    # Manually trigger noise creation
    from sampling.ddim_sampler import create_fixed_noise

    noise = create_fixed_noise((32, 32))
    assert len(noise) == module.model.num_kinds
    for k in range(module.model.num_kinds):
        assert noise[k].shape == (1, 2, 32, 32)
