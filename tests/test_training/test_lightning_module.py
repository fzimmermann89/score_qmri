"""Tests for DiffusionLightningModule."""

import pytest
import torch

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind
from score_qmri.training.lightning_module import DiffusionLightningModule


@pytest.fixture
def module():
    return DiffusionLightningModule(
        base_channels=16,
        num_res_blocks=2,
        channel_mult=(1, 2),
        cond_dim=64,
        validation_size=32,
    )


def test_training_step(module):
    batch = {
        "image": torch.randn(2, 64, 64),
        "mask": torch.ones(2, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 1]),
    }
    loss = module.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_training_step_complex(module):
    batch = {
        "image": torch.complex(torch.randn(2, 64, 64), torch.randn(2, 64, 64)),
        "mask": torch.ones(2, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 1]),
    }
    loss = module.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_configure_optimizers(module):
    config = module.configure_optimizers()
    assert isinstance(config, dict)
    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert isinstance(config["optimizer"], torch.optim.AdamW)
    assert config["lr_scheduler"]["interval"] == "step"


def test_hparams_saved(module):
    assert module.hparams.base_channels == 16
    assert module.hparams.num_res_blocks == 2


def test_create_fixed_noise_for_all_kinds(module):
    from score_qmri.sampling.ddim_sampler import create_fixed_noise

    noise = create_fixed_noise((32, 32))
    assert len(noise) == NUM_DATASET_KINDS

    for kind in DatasetKind:
        n = noise[int(kind)]
        assert n.shape == (1, 32, 32)
        if kind.is_complex:
            assert n.is_complex()
        else:
            assert not n.is_complex()


def test_training_step_partial_valid_samples(module):
    """Test training_step handles mix of valid/invalid samples."""
    batch = {
        "image": torch.randn(4, 64, 64),
        "mask": torch.ones(4, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 1, 0, 1]),
    }
    batch["mask"][:2] = False
    batch["mask"][:2, :10, :10] = True

    loss = module.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss)


def test_training_step_all_invalid_samples(module):
    """Test training_step returns zero loss when all samples have < 1000 valid pixels."""
    batch = {
        "image": torch.randn(4, 64, 64),
        "mask": torch.zeros(4, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 1, 0, 1]),
    }
    batch["mask"][:, :10, :10] = True

    loss = module.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss)


def test_training_step_no_skip_on_partial_invalid(module):
    """Test training_step never returns None even with invalid samples."""
    batch = {
        "image": torch.randn(4, 64, 64),
        "mask": torch.ones(4, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([0, 0, 0, 0]),
    }
    batch["mask"][:2] = False
    batch["mask"][:2, :5, :5] = True

    for _ in range(5):
        loss = module.training_step(batch, 0)
        assert loss is not None
