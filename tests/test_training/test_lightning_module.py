"""Tests for DiffusionLightningModule."""

import pytest
import torch

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind
from score_qmri.training.lightning_module import DiffusionLightningModule


@pytest.fixture
def module():
    return DiffusionLightningModule(
        base_channels=16,
        num_blocks=2,
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
    assert module.hparams.num_blocks == 2


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
