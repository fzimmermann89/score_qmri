"""Checkpoint recovery must preserve recorded architecture, states and units."""

import pickle
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from runpy import run_path

import pytest
import torch
from jsonargparse import Path as ConfigPath

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel, NormalizationConfig
from score_qmri.models.unet import UNet
from score_qmri.utils.checkpoint import checkpoint_config, model_spec, plain_config, portable_checkpoint


def legacy_checkpoint():
    args = {"base_channels": 8, "num_res_blocks": 1, "channel_mult": (1, 2), "cond_dim": 16, "attention_levels": (), "head_dim": 8}
    model = DiffusionModel(unet=UNet(**args), diffusion=DiffusionConfig(cond_dim=16), compile_model=False)
    # Match the wrapper names written by torch.compile in real training runs.
    state = {key.replace("unet.", "unet._orig_mod."): value for key, value in model.state_dict().items()}
    return {
        "global_step": 17,
        "hyper_parameters": {**args, **asdict(DiffusionConfig(cond_dim=16)), "_instantiator": object()},
        "state_dict": {"model." + key: value + 1 for key, value in state.items()},
        "ema": {"shadow": state},
        "optimizer_states": [{"state": {0: {"moment": torch.tensor([1.0])}}}],
    }


def test_legacy_ema_and_raw_weights_load_strictly_without_compilation(tmp_path):
    checkpoint = legacy_checkpoint()
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)
    ema = DiffusionModel.from_checkpoint(path, map_location="cpu", compile_model=False)
    raw = DiffusionModel.from_checkpoint(path, map_location="cpu", compile_model=False, weights="raw")
    for name, value in ema.state_dict().items():
        torch.testing.assert_close(raw.state_dict()[name], value + 1)
    assert not ema.training
    image = torch.zeros(1, 2, 16, 16)
    denoised = ema.denoise(image, torch.tensor([0.5]), torch.tensor([0]))
    assert denoised.shape == image.shape
    assert denoised.isfinite().all()


def test_portable_metadata_supports_weights_only_load_and_preserves_states(tmp_path):
    checkpoint = legacy_checkpoint()
    checkpoint["cli_config"] = {"checkpoint_dir": ConfigPath(str(tmp_path), mode="dr")}
    checkpoint["datamodule_hyper_parameters"] = {"kind": DatasetKind.T1MAP, "normalization": NormalizationConfig()}
    result = portable_checkpoint(checkpoint)
    assert result["state_dict"] is checkpoint["state_dict"]
    assert result["ema"] is checkpoint["ema"]
    assert result["optimizer_states"] is checkpoint["optimizer_states"]
    assert result["global_step"] == 17
    assert result["hyper_parameters"]["diffusion"]["cond_dim"] == 16
    assert type(result["hyper_parameters"]["diffusion"]["cond_dim"]) is int
    path = tmp_path / "portable.ckpt"
    torch.save(result, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["datamodule_hyper_parameters"]["kind"] == int(DatasetKind.T1MAP)
    assert type(loaded["datamodule_hyper_parameters"]["kind"]) is int
    assert loaded["cli_config"]["checkpoint_dir"] == str(tmp_path)
    assert "_instantiator" not in loaded["hyper_parameters"]
    # Conversion must not mutate metadata in the original checkpoint.
    assert "base_channels" in checkpoint["hyper_parameters"]


def test_current_architecture_and_explicit_raw_choice(tmp_path):
    result = portable_checkpoint(legacy_checkpoint())
    path = tmp_path / "current.ckpt"
    torch.save(result, path)
    loaded = DiffusionModel.from_checkpoint(path, map_location="cpu", compile_model=False)
    assert loaded.cond_mlp is not None
    result.pop("ema")
    torch.save(result, path)
    with pytest.raises(ValueError, match="no EMA"):
        DiffusionModel.from_checkpoint(path, map_location="cpu", compile_model=False, weights="ema")
    DiffusionModel.from_checkpoint(path, map_location="cpu", compile_model=False, weights="raw")


def test_missing_legacy_architecture_is_not_guessed():
    with pytest.raises(ValueError, match="architecture fields"):
        model_spec({"hyper_parameters": {"diffusion": asdict(DiffusionConfig())}})


def test_plain_config_does_not_coerce_ints_to_floats():
    config = plain_config({"path": Path("data.h5"), "shape": (1, 2), "flag": True, "x": 0.25})
    assert config == {"path": "data.h5", "shape": [1, 2], "flag": True, "x": 0.25}
    assert type(config["shape"][0]) is int


def test_plain_config_preserves_regular_enum_names():
    class Mode(Enum):
        VALUE = "value"

    assert plain_config(Mode.VALUE) == "VALUE"


def test_converter_verifies_portability_and_refuses_overwrite(tmp_path):
    convert = run_path(str(Path(__file__).parents[2] / "scripts/convert_checkpoint_format.py"))["convert_checkpoint"]
    source, output = tmp_path / "legacy.ckpt", tmp_path / "converted.ckpt"
    torch.save(legacy_checkpoint(), source)
    convert(source, output)
    assert torch.load(output, weights_only=True, mmap=True)["global_step"] == 17
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        convert(source, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("failure", ["write", "validation"])
def test_converter_removes_only_failed_output(tmp_path, monkeypatch, failure):
    convert = run_path(str(Path(__file__).parents[2] / "scripts/convert_checkpoint_format.py"))["convert_checkpoint"]
    source, output = tmp_path / "legacy.ckpt", tmp_path / "converted.ckpt"
    checkpoint = legacy_checkpoint()
    if failure == "validation":
        checkpoint["callbacks"] = {"legacy": object()}
    torch.save(checkpoint, source)
    original = source.read_bytes()
    if failure == "write":

        def failed_save(checkpoint, stream):
            stream.write(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError if failure == "write" else pickle.UnpicklingError):
        convert(source, output)
    assert not output.exists()
    assert source.read_bytes() == original


def test_saved_training_checkpoint_resumes_with_weights_only(tmp_path):
    from jsonargparse import Namespace
    from lightning.pytorch import Trainer
    from lightning.pytorch.cli import LightningArgumentParser
    from torch.utils.data import DataLoader

    from score_qmri.cli import WandbSaveConfigCallback
    from score_qmri.training.lightning_module import DiffusionLightningModule

    def module():
        unet = UNet(base_channels=8, num_res_blocks=1, channel_mult=(1, 2), attention_levels=(), cond_dim=16)
        return DiffusionLightningModule(unet=unet, diffusion=DiffusionConfig(cond_dim=16))

    samples = [
        {
            "image": torch.rand(2, 32, 32),
            "kind": torch.tensor(0),
            "background_mask": torch.zeros(32, 32, dtype=torch.bool),
            "penumbra_mask": torch.zeros(32, 32, dtype=torch.bool),
        }
        for _ in range(4)
    ]
    callback = WandbSaveConfigCallback(
        LightningArgumentParser(), Namespace(checkpoint_dir=ConfigPath(str(tmp_path), mode="dr")), save_to_log_dir=False
    )
    settings = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": str(tmp_path),
    }
    trainer = Trainer(max_steps=1, callbacks=[callback], **settings)
    trainer.fit(module(), train_dataloaders=DataLoader(samples, batch_size=2))
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["cli_config"]["checkpoint_dir"] == str(tmp_path)
    resumed = Trainer(max_steps=2, **settings)
    resumed.fit(module(), train_dataloaders=DataLoader(samples, batch_size=2), ckpt_path=path, weights_only=True)
    assert resumed.global_step == 2


def test_runtime_logger_and_callbacks_are_not_checkpoint_configuration(tmp_path):
    checkpoint = portable_checkpoint(legacy_checkpoint())
    checkpoint["cli_config"]["trainer"] = {"logger": object(), "callbacks": [object()], "max_steps": 100}
    # Architecture recovery does not inspect unrelated runtime services.
    unet, _ = model_spec(checkpoint)
    assert unet["class_path"] == "score_qmri.models.unet.UNet"
    converted = portable_checkpoint(checkpoint)
    assert converted["cli_config"]["trainer"] == {"max_steps": 100}
    assert checkpoint_config(checkpoint["cli_config"]) == converted["cli_config"]
    path = tmp_path / "runtime.ckpt"
    torch.save(converted, path)
    torch.load(path, weights_only=True)


@pytest.mark.parametrize("destination_compiled", [False, True])
def test_full_compiled_training_resume_preserves_optimizer_scheduler_and_ema(tmp_path, destination_compiled):
    from lightning.pytorch import Callback, Trainer
    from torch.utils.data import DataLoader

    from score_qmri.training.lightning_module import DiffusionLightningModule

    def make_module(compiled):
        unet = UNet(base_channels=8, num_res_blocks=1, channel_mult=(1,), attention_levels=(), cond_dim=16)
        # An eager compile backend exercises actual wrapper keys without an
        # expensive compiler invocation in this CPU state-restoration test.
        if compiled:
            with torch._dynamo.config.patch(disable=False):
                unet = torch.compile(unet, backend="eager")
        return DiffusionLightningModule(
            unet=unet, diffusion=DiffusionConfig(cond_dim=16), aux_weight=0.1, warmup_steps=0, cond_dropout_rate=0, compile_model=False
        )

    samples = [
        {
            "image": torch.rand(2, 32, 32) * 0.2,
            "kind": torch.tensor(1),
            "background_mask": torch.zeros(32, 32, dtype=torch.bool),
            "penumbra_mask": torch.zeros(32, 32, dtype=torch.bool),
        }
        for _ in range(4)
    ]
    settings = {
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": str(tmp_path),
        "gradient_clip_val": 1.0,
    }
    trainer = Trainer(max_steps=1, **settings)
    original = make_module(True)
    trainer.fit(original, train_dataloaders=DataLoader(samples, batch_size=2))
    path = tmp_path / "compiled-resume.ckpt"
    trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=True)
    assert any("_orig_mod." in name for name in saved["state_dict"])
    assert saved["optimizer_states"][0]["state"]

    class CheckRestoration(Callback):
        checked = False

        def on_train_start(self, resumed_trainer, module):
            from score_qmri.utils.checkpoint import remap_compiled_state

            assert resumed_trainer.global_step == 1
            for name, tensor in remap_compiled_state(saved["state_dict"], module.state_dict()).items():
                torch.testing.assert_close(module.state_dict()[name], tensor, rtol=0, atol=0)
            for name, tensor in remap_compiled_state(saved["ema"]["shadow"], module.ema.shadow.state_dict()).items():
                torch.testing.assert_close(module.ema.shadow.state_dict()[name], tensor, rtol=0, atol=0)
            actual = resumed_trainer.optimizers[0].state_dict()
            assert actual["param_groups"] == saved["optimizer_states"][0]["param_groups"]
            for index, state in saved["optimizer_states"][0]["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(actual["state"][index][name], value, rtol=0, atol=0)
            assert resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict() == saved["lr_schedulers"][0]
            self.checked = True

    check = CheckRestoration()
    resumed = Trainer(max_steps=2, callbacks=[check], **settings)
    loaded = make_module(destination_compiled)
    resumed.fit(loaded, train_dataloaders=DataLoader(samples, batch_size=2), ckpt_path=path, weights_only=True)
    assert check.checked and resumed.global_step == 2
    assert loaded._ema_step == 2
