"""Pure-data checkpoint metadata and recovery of recorded model specifications."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum, IntEnum
from os import PathLike, fspath
from typing import Any


def remap_compiled_state(state: Mapping[str, Any], target_keys: Iterable[str]) -> dict[str, Any]:
    """Map torch.compile wrapper paths to the destination without losing tensors."""
    targets = {key.replace("_orig_mod.", ""): key for key in target_keys}
    result = {}
    for key, value in state.items():
        target = targets.get(key.replace("_orig_mod.", ""), key)
        if target in result:
            raise ValueError(f"Ambiguous compiled checkpoint keys for {target!r}")
        result[target] = value
    return result


def plain_config(value: Any) -> Any:
    """Remove Python-specific config objects without changing numeric values."""
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, PathLike):
        return fspath(value)
    if is_dataclass(value) and not isinstance(value, type):
        return plain_config(asdict(value))
    if hasattr(value, "as_dict"):
        return plain_config(value.as_dict())
    if isinstance(value, Mapping):
        return {k: plain_config(v) for k, v in value.items() if k != "_instantiator"}
    if isinstance(value, (tuple, list)):
        return [plain_config(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # jsonargparse.Path predates os.PathLike support in some environments.
    if type(value).__module__ == "jsonargparse._paths" and type(value).__name__ == "Path":
        return str(value)
    raise TypeError(f"Cannot serialize checkpoint configuration value {type(value)!r}")


def checkpoint_config(config: Any) -> dict[str, Any]:
    """Save declarative CLI settings, excluding CLI-created runtime services.

    The training CLI constructs trainer.logger and trainer.callbacks in
    before_instantiate_classes. They are not portable configuration or state;
    Lightning saves callback state separately under the checkpoint callbacks key.
    """
    payload = config.as_dict() if hasattr(config, "as_dict") else dict(config)
    if "trainer" in payload:
        trainer = payload["trainer"]
        trainer = trainer.as_dict() if hasattr(trainer, "as_dict") else dict(trainer)
        payload["trainer"] = {key: value for key, value in trainer.items() if key not in ("logger", "callbacks")}
    return dict(plain_config(payload))


def model_spec(checkpoint: Mapping[str, Any], *, require_unet: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover UNet and diffusion specs from current or legacy recorded metadata.

    Legacy architecture fields must be present; tensor shapes are not guessed.
    This does not unpickle a file or modify the checkpoint.
    """
    hparams = plain_config(checkpoint.get("hyper_parameters", {}))
    cli = checkpoint.get("cli_config", {})
    model = plain_config(cli.get("model", {}))
    diffusion = hparams.get("diffusion", model.get("diffusion"))
    if diffusion is None:
        keys = ("cond_dim", "num_kinds", "sigma_min", "sigma_max", "sigma_data", "p_mean", "p_std", "rho", "normalization")
        diffusion = {k: hparams[k] for k in keys if k in hparams and hparams[k] is not None}
        if not diffusion:
            raise ValueError("Checkpoint has no recorded diffusion hyperparameters")
    if not isinstance(diffusion, dict):
        raise ValueError("Checkpoint diffusion configuration must be a mapping")

    unet = model.get("unet")
    if unet is None and not require_unet:
        return {}, diffusion
    if unet is None:
        fields = ("base_channels", "num_res_blocks", "channel_mult", "cond_dim", "attention_levels", "head_dim")
        missing = [k for k in fields if k not in hparams]
        if missing:
            raise ValueError(f"Checkpoint lacks a UNet spec and legacy architecture fields: {missing}")
        unet = {
            "class_path": "score_qmri.models.unet.UNet",
            "init_args": {"in_channels": 2, **{k: hparams[k] for k in fields}},
        }
    if not isinstance(unet, dict) or "class_path" not in unet:
        raise ValueError("Checkpoint must record a UNet class_path and init_args")
    return unet, diffusion


def portable_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Copy checkpoint metadata into the current format, preserving all states.

    Input must already have been loaded with the appropriate trust policy.
    Model/EMA/optimizer tensors are shared with the input rather than copied.
    """
    unet, diffusion = model_spec(checkpoint)
    result = dict(checkpoint)
    cli = checkpoint_config(checkpoint.get("cli_config", {}))
    cli["model"] = {**cli.get("model", {}), "unet": unet, "diffusion": diffusion}
    result["cli_config"] = cli
    hparams = plain_config(checkpoint.get("hyper_parameters", {}))
    legacy = {
        "base_channels",
        "num_res_blocks",
        "channel_mult",
        "cond_dim",
        "num_kinds",
        "attention_levels",
        "head_dim",
        "sigma_min",
        "sigma_max",
        "sigma_data",
        "p_mean",
        "p_std",
        "rho",
        "normalization",
        "edm",
    }
    result["hyper_parameters"] = {**{k: v for k, v in hparams.items() if k not in legacy}, "diffusion": diffusion}
    if "datamodule_hyper_parameters" in result:
        result["datamodule_hyper_parameters"] = plain_config(result["datamodule_hyper_parameters"])
    return result
