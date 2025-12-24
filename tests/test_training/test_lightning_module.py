"""Tests for DiffusionLightningModule."""

import pytest
import torch

from models.diffusion import DiffusionModel
from training.lightning_module import DiffusionLightningModule
from training.losses import LossScheduler, MSELoss, SSIML1Loss


class MockConfig:
    learning_rate = 1e-4
    validation_freq = 10


@pytest.fixture
def lightning_module():
    """Create Lightning module for testing."""
    model = DiffusionModel(num_kinds=3, num_timesteps=100)
    mse = MSELoss()
    try:
        ssim_l1 = SSIML1Loss()
    except ImportError:
        pytest.skip("pytorch-msssim not available")

    scheduler = LossScheduler(mse, ssim_l1, lambda s: 0.0)
    return DiffusionLightningModule(model, scheduler, MockConfig())


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
