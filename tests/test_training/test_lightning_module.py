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
    weighted_aux = module.training_step(batch, 0)
    module.hparams.background_weight = 1.0
    module.hparams.penumbra_weight = 1.0
    torch.manual_seed(123)
    uniform_aux = module.training_step(batch, 0)
    torch.testing.assert_close(weighted_aux, uniform_aux, rtol=1e-6, atol=1e-8)


def test_disabled_auxiliary_loss_is_not_evaluated(module, monkeypatch):
    def fail_aux(*args, **kwargs):
        raise AssertionError("Disabled auxiliary loss must not be evaluated")

    monkeypatch.setattr("score_qmri.training.lightning_module.ssim_loss", fail_aux)
    monkeypatch.setattr("score_qmri.training.lightning_module.l1_loss", fail_aux)
    module.hparams.aux_weight = 0.0
    batch = {
        "image": 0.1 * torch.randn(1, 2, 64, 64),
        "background_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "penumbra_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([1]),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in module.model.parameters())


@pytest.mark.parametrize("sigma", [0.001, 0.01, 0.2, 0.5])
def test_auxiliary_raw_network_weight_cancels_edm_output_scale(module, monkeypatch, sigma):
    def zero_loss(prediction, *args, **kwargs):
        return prediction.flatten(1).sum(1) * 0

    def unit_loss(prediction, *args, **kwargs):
        return zero_loss(prediction) + 1

    monkeypatch.setattr(module.model, "sample_sigmas", lambda batch_size, device: torch.full((batch_size,), sigma, device=device))
    monkeypatch.setattr("score_qmri.training.lightning_module.noise_loss", zero_loss)
    monkeypatch.setattr("score_qmri.training.lightning_module.ssim_loss", zero_loss)
    monkeypatch.setattr("score_qmri.training.lightning_module.l1_loss", unit_loss)
    module.hparams.aux_weight = 0.02
    batch = {
        "image": 0.1 * torch.randn(1, 2, 64, 64),
        "background_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "penumbra_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
        "kind": torch.tensor([1]),
    }
    loss = module.training_step(batch, 0)
    c_out = sigma * module.model.sigma_data / (sigma**2 + module.model.sigma_data**2) ** 0.5
    expected = 0.02 * module.hparams.l1_weight / c_out
    assert loss.item() == pytest.approx(expected)
    loss.backward()


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
