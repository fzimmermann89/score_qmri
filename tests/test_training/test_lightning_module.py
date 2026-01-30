"""Tests for DiffusionLightningModule."""

import pytest
import torch

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind, is_complex_kind
from score_qmri.training.lightning_module import DiffusionLightningModule


@pytest.fixture
def module():
    return DiffusionLightningModule(
        base_channels=16,
        num_res_blocks=2,
        channel_mult=(1, 2),
        cond_dim=64,
        attention_levels=(),
    )


def test_training_step(module):
    background_mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    batch = {
        "image": torch.randn(2, 2, 64, 64),
        "penumbra_mask": torch.zeros(2, 64, 64, dtype=torch.bool),
        "background_mask": background_mask,
        "kind": torch.tensor([0, 1]),
    }
    loss = module.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.item() >= 0


def test_training_step_real_kind(module):
    """Test training with real kinds (imag channel should stay zero)."""
    background_mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    batch = {
        "image": torch.randn(2, 2, 64, 64),
        "penumbra_mask": torch.zeros(2, 64, 64, dtype=torch.bool),
        "background_mask": background_mask,
        "kind": torch.tensor([6, 6]),  # T1MAP is a real kind
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
    from score_qmri.sampling.edm_sampler import create_fixed_noise

    noise = create_fixed_noise((32, 32))
    assert len(noise) == NUM_DATASET_KINDS

    for kind in DatasetKind:
        n = noise[int(kind)]
        assert n.shape == (4, 2, 32, 32)
        if is_complex_kind(int(kind)):
            assert not torch.allclose(n[:, 1], torch.zeros_like(n[:, 1]))
        else:
            assert torch.allclose(n[:, 1], torch.zeros_like(n[:, 1]))


def test_training_step_partial_valid_samples(module):
    background_mask = torch.zeros(4, 64, 64, dtype=torch.bool)
    background_mask[:2] = True
    background_mask[:2, :10, :10] = False
    batch = {
        "image": torch.randn(4, 2, 64, 64),
        "penumbra_mask": torch.zeros(4, 64, 64, dtype=torch.bool),
        "background_mask": background_mask,
        "kind": torch.tensor([0, 1, 0, 1]),
    }

    loss = module.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss)


def test_training_step_all_invalid_samples(module):
    background_mask = torch.ones(4, 64, 64, dtype=torch.bool)
    background_mask[:, :10, :10] = False
    batch = {
        "image": torch.randn(4, 2, 64, 64),
        "penumbra_mask": torch.zeros(4, 64, 64, dtype=torch.bool),
        "background_mask": background_mask,
        "kind": torch.tensor([0, 1, 0, 1]),
    }

    loss = module.training_step(batch, 0)
    assert loss is not None
    assert torch.isfinite(loss)


def test_training_step_no_skip_on_partial_invalid(module):
    background_mask = torch.zeros(4, 64, 64, dtype=torch.bool)
    background_mask[:2] = True
    background_mask[:2, :5, :5] = False
    batch = {
        "image": torch.randn(4, 2, 64, 64),
        "penumbra_mask": torch.zeros(4, 64, 64, dtype=torch.bool),
        "background_mask": background_mask,
        "kind": torch.tensor([0, 0, 0, 0]),
    }

    for _ in range(5):
        loss = module.training_step(batch, 0)
        assert loss is not None
