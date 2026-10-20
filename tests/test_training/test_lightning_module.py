"""Tests for DiffusionLightningModule."""

import pytest
import torch
from lightning import __version__ as lightning_version

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionConfig
from score_qmri.models.unet import UNet
from score_qmri.training.lightning_module import DiffusionLightningModule


@pytest.fixture
def module():
    diffusion = DiffusionConfig(cond_dim=64)
    unet = UNet(
        in_channels=2,
        base_channels=16,
        num_res_blocks=2,
        channel_mult=(1, 2),
        cond_dim=diffusion.cond_dim,
        attention_levels=(),
    )
    return DiffusionLightningModule(unet=unet, diffusion=diffusion)


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
    assert not hasattr(module.hparams, "unet")
    assert not hasattr(module.hparams, "unet_spec")
    assert module.hparams.diffusion["cond_dim"] == 64


def test_load_from_checkpoint_roundtrip(tmp_path, module):
    ckpt_path = tmp_path / "module.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": lightning_version,
            "cli_config": {
                "model": {
                    "unet": {
                        "class_path": "score_qmri.models.unet.UNet",
                        "init_args": {
                            "in_channels": 2,
                            "base_channels": 16,
                            "num_res_blocks": 2,
                            "channel_mult": [1, 2],
                            "attention_levels": [],
                        },
                    },
                    "diffusion": module.hparams.diffusion,
                }
            },
        },
        ckpt_path,
    )

    loaded = DiffusionLightningModule.load_from_checkpoint(ckpt_path)
    assert loaded.hparams.diffusion["cond_dim"] == 64


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
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for parameter in module.model.parameters())


def test_main_score_loss_is_independent_of_foreground_weights(module):
    """Clean-target coordinate weights must not bias the denoising conditional mean."""
    background = torch.zeros(1, 64, 64, dtype=torch.bool)
    background[:, :16] = True
    penumbra = torch.zeros_like(background)
    penumbra[:, 16:32] = True
    batch = {
        "image": 0.1 * torch.randn(1, 2, 64, 64),
        "background_mask": background,
        "penumbra_mask": penumbra,
        "kind": torch.tensor([1]),
    }
    module.hparams.aux_weight = 0.0
    module.hparams.background_weight = 0.05
    module.hparams.penumbra_weight = 0.01
    torch.manual_seed(123)
    weighted = module.training_step(batch, 0)
    module.hparams.background_weight = 1.0
    module.hparams.penumbra_weight = 1.0
    torch.manual_seed(123)
    uniform = module.training_step(batch, 0)
    torch.testing.assert_close(weighted, uniform, rtol=1e-6, atol=1e-8)


def test_ema_updates_once_per_optimizer_step(module, monkeypatch):
    from types import SimpleNamespace

    trainer = SimpleNamespace(global_step=0)
    module._trainer = trainer
    updates = []
    monkeypatch.setattr(module.ema, "update", lambda model: updates.append(trainer.global_step))
    for step in [0, 0, 1, 1, 2, 2]:
        trainer.global_step = step
        module.on_train_batch_end(None, None, 0)
    assert updates == [1, 2]


def test_nonfinite_loss_aborts_instead_of_returning_zero(module, monkeypatch):
    value = float("nan")
    monkeypatch.setattr(
        "score_qmri.training.lightning_module.noise_loss", lambda prediction, *args, **kwargs: prediction.flatten(1).sum(1) * value
    )
    module.hparams.aux_weight = 0
    batch = {
        "image": torch.randn(1, 2, 64, 64),
        "background_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "penumbra_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([1]),
    }
    with pytest.raises(FloatingPointError, match="training loss"):
        module.training_step(batch, 0)


def test_large_finite_loss_is_trained_instead_of_silently_rejected(module, monkeypatch):
    monkeypatch.setattr(
        "score_qmri.training.lightning_module.noise_loss", lambda prediction, *args, **kwargs: prediction.flatten(1).sum(1) * 0 + 1000
    )
    module.hparams.aux_weight = 0
    batch = {
        "image": torch.randn(1, 2, 64, 64),
        "background_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "penumbra_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([1]),
    }
    loss = module.training_step(batch, 0)
    assert loss.requires_grad and loss > 100


@pytest.mark.parametrize("scaled", [False, True])
def test_nonfinite_gradients_do_not_reach_adamw(module, scaled):
    from types import SimpleNamespace

    optimizer = module.configure_optimizers()["optimizer"]
    parameter = next(module.model.parameters())
    before = parameter.detach().clone()
    scaler = torch.amp.GradScaler("cpu") if scaled else None
    module._trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))
    if scaler is not None:
        scaler.scale(parameter.sum()).backward()
    parameter.grad = torch.full_like(parameter, float("nan"))
    if scaler is None:
        with pytest.raises(FloatingPointError, match="training gradients"):
            module.on_before_optimizer_step(optimizer)
    else:
        scale = scaler.get_scale()
        scaler.unscale_(optimizer)
        module.on_before_optimizer_step(optimizer)
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() < scale
    assert torch.equal(parameter, before)
    assert not optimizer.state
