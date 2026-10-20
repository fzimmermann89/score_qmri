"""Convert legacy checkpoints to the new hyper-parameter format."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from score_qmri.datasets.kind import NUM_DATASET_KINDS
from score_qmri.models.diffusion import DiffusionConfig, NormalizationConfig


def _ensure_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    return tuple(value)


def _build_normalization(hparams: dict[str, Any]) -> NormalizationConfig:
    normalization = hparams.get("normalization")
    if isinstance(normalization, dict):
        return NormalizationConfig(**dict(normalization))
    if normalization is None:
        return NormalizationConfig()
    if isinstance(normalization, NormalizationConfig):
        return normalization
    raise TypeError(f"Unsupported normalization type: {type(normalization)!r}")


def _extract_diffusion_dict(hparams: dict[str, Any]) -> dict[str, Any]:
    diffusion = hparams.get("diffusion")
    if isinstance(diffusion, dict):
        return asdict(DiffusionConfig.from_mapping(diffusion))
    if isinstance(diffusion, DiffusionConfig):
        return asdict(DiffusionConfig.from_mapping(diffusion.__dict__))

    config = DiffusionConfig(
        cond_dim=int(hparams.get("cond_dim", DiffusionConfig.cond_dim)),
        num_kinds=int(hparams.get("num_kinds", NUM_DATASET_KINDS)),
        sigma_min=float(hparams.get("sigma_min", DiffusionConfig.sigma_min)),
        sigma_max=float(hparams.get("sigma_max", DiffusionConfig.sigma_max)),
        sigma_data=float(hparams.get("sigma_data", DiffusionConfig.sigma_data)),
        p_mean=float(hparams.get("p_mean", DiffusionConfig.p_mean)),
        p_std=float(hparams.get("p_std", DiffusionConfig.p_std)),
        rho=float(hparams.get("rho", DiffusionConfig.rho)),
        normalization=_build_normalization(hparams),
    )
    return asdict(config)


def _build_legacy_unet_spec(hparams: dict[str, Any], *, cond_dim: int) -> dict[str, Any]:
    return {
        "class_path": "score_qmri.models.unet.UNet",
        "init_args": {
            "in_channels": 2,
            "base_channels": int(hparams.get("base_channels", 128)),
            "num_res_blocks": int(hparams.get("num_res_blocks", 2)),
            "channel_mult": list(_ensure_tuple(hparams.get("channel_mult"), (1, 2, 2, 4, 4, 4))),
            "attention_levels": list(_ensure_tuple(hparams.get("attention_levels"), (-1, -2, -3))),
            "head_dim": int(hparams.get("head_dim", 64)),
            "cond_dim": int(cond_dim),
        },
    }


def _extract_unet_spec(checkpoint: dict[str, Any], hparams: dict[str, Any], *, cond_dim: int) -> dict[str, Any]:
    cli_config = checkpoint.get("cli_config")
    if isinstance(cli_config, dict):
        model_cfg = cli_config.get("model")
        if isinstance(model_cfg, dict):
            unet = model_cfg.get("unet")
            if isinstance(unet, dict) and "class_path" in unet:
                spec = dict(unet)
                init_args = dict(spec.get("init_args", {}))
                init_args["cond_dim"] = int(cond_dim)
                spec["init_args"] = init_args
                return spec

    for key in ("unet", "unet_spec"):
        unet = hparams.get(key)
        if isinstance(unet, dict) and "class_path" in unet:
            spec = dict(unet)
            init_args = dict(spec.get("init_args", {}))
            init_args["cond_dim"] = int(cond_dim)
            spec["init_args"] = init_args
            return spec

    return _build_legacy_unet_spec(hparams, cond_dim=cond_dim)


def convert_checkpoint(input_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters") or checkpoint.get("hparams")
    if not isinstance(hparams, dict):
        raise TypeError("Checkpoint hyper_parameters must be a mapping.")

    diffusion = _extract_diffusion_dict(hparams)
    unet_spec = _extract_unet_spec(checkpoint, hparams, cond_dim=int(diffusion["cond_dim"]))

    remove_keys = {
        "unet",
        "unet_spec",
        "edm",
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
    }

    new_hparams = {k: v for k, v in hparams.items() if k not in remove_keys}
    new_hparams["diffusion"] = diffusion

    cli_config = checkpoint.get("cli_config")
    new_cli_config: dict[str, Any] = dict(cli_config) if isinstance(cli_config, dict) else {}
    model_cfg = new_cli_config.get("model")
    new_model_cfg: dict[str, Any] = dict(model_cfg) if isinstance(model_cfg, dict) else {}
    new_model_cfg["unet"] = unet_spec
    new_model_cfg["diffusion"] = diffusion
    new_cli_config["model"] = new_model_cfg

    checkpoint.pop("hparams", None)
    checkpoint["cli_config"] = new_cli_config

    checkpoint["hyper_parameters"] = new_hparams
    torch.save(checkpoint, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert checkpoints to new hyper-parameter format.")
    parser.add_argument("input", type=Path, help="Input checkpoint path")
    parser.add_argument("--output", type=Path, help="Output checkpoint path")
    args = parser.parse_args()

    output = args.output
    if output is None:
        suffix = args.input.suffix or ".ckpt"
        output = args.input.with_suffix(f".converted{suffix}")

    convert_checkpoint(args.input, output)


if __name__ == "__main__":
    main()
