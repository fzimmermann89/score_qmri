"""VE/EDM diffusion model with continuous noise level sigma.

This repository uses a 2-channel real representation for both real and complex data:

- Complex kinds: channel 0 = real, channel 1 = imag
- Real kinds: channel 0 = value, channel 1 = zeros

The forward corruption process is Variance Exploding (VE):

    x = x0 + sigma * eps,   eps ~ N(0, I)

We follow the EDM (Karras et al., 2022) preconditioning scheme to stabilize training
across a wide range of sigma values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, cast, overload

import torch
from torch import Tensor

from score_qmri.datasets.kind import NUM_DATASET_KINDS, DatasetKind, is_complex_kind, is_mapping_kind

from .conditioning import ConditioningMLP

DEFAULT_KIND_SCALES: dict[str, float] = {
    "T1MAP": 1 / 4,
    "T2MAP": 2,
    "T2STARMAP": 8,
}


@dataclass
class NormalizationConfig:
    """Normalization configuration.

    Notes
    -----
    - Mapping kinds use fixed per-kind scaling (kind_scales) to map values into a
      stable training range and to allow conversion back to physical units.
    - train_clip/sample_clip clamp clean-image predictions (x0) in normalized space.
    """

    train_clip: float = 1.2
    sample_clip: float = 1.0
    kind_scales: dict[str, float] | None = None


@dataclass
class DiffusionConfig:
    """Diffusion model configuration.

    Parameters
    ----------
    cond_dim
        Conditioning dimension used by the conditioning MLP and UNet FiLM blocks.
    num_kinds
        Number of dataset kinds for the kind embedding.
    sigma_min, sigma_max
        Minimum and maximum sigmas used for training and sampling.
    sigma_data
        Characteristic standard deviation of normalized training data.
    p_mean, p_std
        Log-normal sigma sampling distribution for training.
    rho
        Exponent for Karras (poly-rho) sigma schedule in sampling.
    normalization
        Normalization configuration.

    Notes
    -----
    ``cond_dim`` is the single source of truth for the conditioning dimension.
    It is used for both:

    - the output dimension of :class:`~score_qmri.models.conditioning.ConditioningMLP`
    - the FiLM/AdaGN conditioning dimension inside the UNet
    """

    cond_dim: int = 128
    num_kinds: int = NUM_DATASET_KINDS
    sigma_min: float = 0.001
    sigma_max: float = 20.0
    sigma_data: float = 0.2
    p_mean: float = -1.6
    p_std: float = 1.4
    rho: float = 7.0
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DiffusionConfig:
        payload = dict(data)

        normalization = payload.get("normalization")
        if isinstance(normalization, Mapping):
            payload["normalization"] = NormalizationConfig(**dict(normalization))
        elif normalization is None:
            payload.pop("normalization", None)
        elif not isinstance(normalization, NormalizationConfig):
            raise TypeError(f"Unsupported normalization type: {type(normalization)!r}")

        return cls(**payload)


class InputFormat(IntEnum):
    REAL_1CH = auto()
    COMPLEX_1CH = auto()
    REAL_2CH = auto()
    LIST_2CH = auto()


@dataclass
class ListMetadata:
    """Metadata for reconstructing list outputs from concatenated 2ch tensor."""

    formats: list[InputFormat]
    batch_sizes: list[int]


@dataclass
class DenoiseOutput:
    """Denoiser outputs in the input's original format."""

    x0_main: Tensor
    x0_aux: Tensor


@dataclass
class DenoiseOutputList:
    x0_main: list[Tensor]
    x0_aux: list[Tensor]


def _to_2ch_single(x: Tensor) -> tuple[Tensor, InputFormat]:
    """Convert single tensor to canonical (B, 2, H, W) format."""
    match (x.ndim, x.is_complex()):
        case (4, False) if x.shape[1] == 2:
            return x, InputFormat.REAL_2CH
        case (3, True):
            return torch.moveaxis(torch.view_as_real(x), -1, 1), InputFormat.COMPLEX_1CH
        case (3, False):
            return torch.stack([x, torch.zeros_like(x)], dim=1), InputFormat.REAL_1CH
        case _:
            raise ValueError(f"Unsupported input shape/dtype: {x.shape}, complex={x.is_complex()}")


def _to_2ch(x: Tensor | Sequence[Tensor], kind: Tensor | None = None) -> tuple[Tensor, InputFormat, ListMetadata | None]:
    """Convert input to canonical (B, 2, H, W) format.

    For list/sequence inputs, tensors are concatenated along the batch dimension.
    Each element can be real (B_i, H, W), complex (B_i, H, W), or 2ch (B_i, 2, H, W).
    """
    if isinstance(x, (list, tuple)):
        if not all(isinstance(tensor, Tensor) for tensor in x):
            bad = next(tensor for tensor in x if not isinstance(tensor, Tensor))
            raise TypeError(f"List elements must be Tensors, got {type(bad)}")
        converted = [_to_2ch_single(tensor) for tensor in x]
        chunks_2ch = [tensor for tensor, _ in converted]
        formats = [fmt for _, fmt in converted]
        batch_sizes = [tensor.shape[0] for tensor in chunks_2ch]
        x_2ch = torch.cat(chunks_2ch, dim=0)
        if kind is not None:
            x_2ch = _apply_kind_constraints(x_2ch, kind)
        return x_2ch, InputFormat.LIST_2CH, ListMetadata(formats, batch_sizes)

    x_2ch, fmt = _to_2ch_single(cast(Tensor, x))
    if kind is not None:
        x_2ch = _apply_kind_constraints(x_2ch, kind)
    return x_2ch, fmt, None


def _from_2ch_single(x_2ch: Tensor, fmt: InputFormat) -> Tensor:
    match fmt:
        case InputFormat.REAL_2CH:
            return x_2ch
        case InputFormat.COMPLEX_1CH:
            return torch.view_as_complex(torch.moveaxis(x_2ch, 1, -1).contiguous())
        case InputFormat.REAL_1CH:
            return x_2ch[:, 0]
        case _:
            raise ValueError(f"Unsupported format: {fmt}")


def _from_2ch(
    x_2ch: Tensor,
    fmt: InputFormat,
    kind: Tensor | None = None,
    metadata: ListMetadata | None = None,
) -> Tensor | list[Tensor]:
    """Convert canonical (B, 2, H, W) back to original format."""
    if kind is not None:
        x_2ch = _apply_kind_constraints(x_2ch, kind)

    match fmt:
        case InputFormat.LIST_2CH:
            if metadata is None:
                raise ValueError("LIST_2CH format requires metadata")
            chunks = x_2ch.split(metadata.batch_sizes, dim=0)
            return [_from_2ch_single(chunk, elem_fmt) for chunk, elem_fmt in zip(chunks, metadata.formats, strict=True)]
        case _:
            return _from_2ch_single(x_2ch, fmt)


def _apply_kind_constraints(x_2ch: Tensor, kind: Tensor) -> Tensor:
    """Zero out imaginary channel for real kinds."""
    result = x_2ch.clone()
    for k in kind.unique():
        k_idx = int(k.item())
        if not is_complex_kind(k_idx):
            idx = kind == k
            result[idx, 1] = 0.0
    return result


def _create_noise(shape: tuple[int, ...], kind: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create noise with imag=0 for real kinds."""
    noise = torch.randn(shape, device=device, dtype=dtype)
    return _apply_kind_constraints(noise, kind)


def _extract_model_config(cli_config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_cfg = cli_config.get("model")
    if isinstance(model_cfg, Mapping):
        return model_cfg

    raise ValueError("Checkpoint cli_config does not contain model configuration.")


def _instantiate_unet_from_spec(unet_spec: Mapping[str, Any], *, cond_dim: int | None = None) -> torch.nn.Module:
    import jsonargparse

    spec = dict(unet_spec)
    init_args = dict(spec.get("init_args", {}))
    if cond_dim is not None:
        init_args["cond_dim"] = int(cond_dim)
    spec["init_args"] = init_args

    parser = jsonargparse.ArgumentParser()
    parser.add_subclass_arguments(torch.nn.Module, "unet")
    cfg = parser.parse_object({"unet": spec})
    unet = parser.instantiate_classes(cfg).unet
    if not isinstance(unet, torch.nn.Module):
        raise TypeError(f"UNet spec did not instantiate torch.nn.Module, got {type(unet)!r}")
    return unet


class DiffusionModel(torch.nn.Module):
    """VE/EDM diffusion model with continuous sigma.

    This module wraps a conditional UNet and exposes sigma-native primitives:
    - q_sample(x0, sigma)
    - denoise(x, sigma)
    - score(x, sigma)

    Conditioning is via FiLM with a conditioning MLP that embeds:
    - EDM noise embedding: c_noise = log(sigma) / 4
    - dataset kind embedding
    """

    sigma_min: float
    sigma_max: float
    sigma_data: float
    p_mean: float
    p_std: float
    rho: float
    train_clip: float
    sample_clip: float
    _kind_scale_lookup: dict[int, float]

    def __init__(
        self,
        *,
        unet: torch.nn.Module,
        diffusion: DiffusionConfig | dict[str, Any] | None = None,
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_info: dict[str, Any] = {}

        if diffusion is None:
            diffusion = DiffusionConfig()
        elif isinstance(diffusion, DiffusionConfig):
            diffusion = DiffusionConfig.from_mapping(diffusion.__dict__)
        else:
            if not isinstance(diffusion, dict):
                raise TypeError(f"Unsupported diffusion type: {type(diffusion)!r}")
            diffusion = DiffusionConfig.from_mapping(diffusion)

        self.sigma_min = float(diffusion.sigma_min)
        self.sigma_max = float(diffusion.sigma_max)
        self.sigma_data = float(diffusion.sigma_data)
        self.p_mean = float(diffusion.p_mean)
        self.p_std = float(diffusion.p_std)
        self.rho = float(diffusion.rho)

        norm = diffusion.normalization
        self.train_clip = float(norm.train_clip)
        self.sample_clip = float(norm.sample_clip)

        merged_kind_scales = {**DEFAULT_KIND_SCALES, **(norm.kind_scales or {})}
        self._kind_scale_lookup: dict[int, float] = {}
        for kind_name, scale in merged_kind_scales.items():
            kind_enum = DatasetKind[kind_name]
            self._kind_scale_lookup[int(kind_enum)] = float(scale)

        self.cond_mlp = ConditioningMLP(cond_dim=diffusion.cond_dim, num_kinds=diffusion.num_kinds)

        self.unet = cast(torch.nn.Module, torch.compile(unet, mode="reduce-overhead")) if compile_model else unet

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_prior_sigma(self) -> float:
        """Prior sigma for VE sampling: x ~ N(0, sigma_max^2 I)."""
        return self.sigma_max

    def sample_sigmas(self, batch_size: int, device: torch.device | None = None, *, generator: torch.Generator | None = None) -> Tensor:
        """Sample training sigmas from EDM log-normal distribution."""
        device = self.device if device is None else device
        log_sigma = torch.randn(batch_size, device=device, generator=generator) * self.p_std + self.p_mean
        sigma = log_sigma.exp().clamp(min=self.sigma_min, max=self.sigma_max)
        return sigma

    def _edm_preconditioning(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute EDM preconditioning scalars.

        Returns
        -------
        c_skip, c_out, c_in, c_noise
            Each tensor has shape (B,).
        """
        sigma_f = sigma.float().clamp(min=self.sigma_min)
        sigma_data = torch.tensor(self.sigma_data, device=sigma.device, dtype=torch.float32)
        sigma2 = sigma_f.square()
        sigma_data2 = sigma_data.square()
        denom = sigma2 + sigma_data2

        c_skip = sigma_data2 / denom
        c_out = sigma_f * sigma_data / denom.sqrt()
        c_in = denom.rsqrt()
        c_noise = sigma_f.log() / 4.0

        return c_skip, c_out, c_in, c_noise

    @overload
    def q_sample(self, x0: Tensor, sigma: Tensor, kind: Tensor, noise: Tensor | None = None) -> Tensor: ...
    @overload
    def q_sample(self, x0: Sequence[Tensor], sigma: Tensor, kind: Tensor, noise: Sequence[Tensor] | None = None) -> list[Tensor]: ...
    def q_sample(
        self,
        x0: Tensor | Sequence[Tensor],
        sigma: Tensor,
        kind: Tensor,
        noise: Tensor | Sequence[Tensor] | None = None,
    ) -> Tensor | list[Tensor]:
        """Sample x = x0 + sigma * eps in canonical 2ch space.

        Input/output formats follow the same rules as in `forward`.
        """
        x0_2ch, fmt, metadata = _to_2ch(x0, kind)
        B = x0_2ch.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        if sigma.shape[0] != B:
            raise ValueError(f"sigma must have shape (B,), got {sigma.shape} for B={B}")

        if noise is None:
            eps = _create_noise(x0_2ch.shape, kind, x0_2ch.device, x0_2ch.dtype)
        else:
            eps, _, _ = _to_2ch(noise, kind)

        sigma_b = sigma.to(device=x0_2ch.device, dtype=x0_2ch.dtype).view(-1, 1, 1, 1)
        x = x0_2ch + sigma_b * eps
        x = _apply_kind_constraints(x, kind)

        return _from_2ch(x, fmt, kind, metadata)

    def _denoise_both_2ch(self, x_2ch: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Denoise in canonical 2ch space. Returns (x0_main, x0_aux)."""
        B = x_2ch.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        if sigma.shape[0] != B:
            raise ValueError(f"sigma must have shape (B,), got {sigma.shape} for B={B}")

        x_2ch = _apply_kind_constraints(x_2ch, kind)

        c_skip, c_out, c_in, c_noise = self._edm_preconditioning(sigma)
        cond = self.cond_mlp(c_noise.to(device=x_2ch.device), kind, mask=cond_mask)

        x_in = (c_in.to(device=x_2ch.device, dtype=x_2ch.dtype).view(-1, 1, 1, 1) * x_2ch).contiguous()
        if x_2ch.is_contiguous(memory_format=torch.channels_last):
            x_in = x_in.contiguous(memory_format=torch.channels_last)

        out = self.unet(x_in, cond=cond)
        if isinstance(out, Tensor):
            f_main, f_aux = out.chunk(2, dim=1)
        else:
            f_main, f_aux = out

        c_skip_b = c_skip.to(device=x_2ch.device, dtype=x_2ch.dtype).view(-1, 1, 1, 1)
        c_out_b = c_out.to(device=x_2ch.device, dtype=x_2ch.dtype).view(-1, 1, 1, 1)
        x0_main = c_skip_b * x_2ch + c_out_b * f_main
        x0_aux = c_skip_b * x_2ch + c_out_b * f_aux

        x0_main = _apply_kind_constraints(x0_main, kind)
        x0_aux = _apply_kind_constraints(x0_aux, kind)

        return x0_main, x0_aux

    @overload
    def forward(self, x: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> DenoiseOutput: ...
    @overload
    def forward(self, x: Sequence[Tensor], sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> DenoiseOutputList: ...
    def forward(
        self,
        x: Tensor | Sequence[Tensor],
        sigma: Tensor,
        kind: Tensor,
        cond_mask: Tensor | None = None,
    ) -> DenoiseOutput | DenoiseOutputList:
        """Predict clean images from a noisy input.

        Parameters
        ----------
        x
            Noisy input. Supported formats:
            - real: (B, H, W)
            - complex: (B, H, W) complex
            - 2ch: (B, 2, H, W)
            - list/tuple of the above (concatenated along batch)
        sigma
            Noise level(s) of shape (B,) or scalar.
        kind
            Dataset kind indices of shape (B,).
        cond_mask
            Optional conditioning mask for CFG-style kind dropout.

        Returns
        -------
        DenoiseOutput | DenoiseOutputList
            Predictions in the same input format.
        """
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x0_main_2ch, x0_aux_2ch = self._denoise_both_2ch(x_2ch, sigma, kind, cond_mask=cond_mask)

        x0_main = _from_2ch(x0_main_2ch, fmt, kind, metadata)
        x0_aux = _from_2ch(x0_aux_2ch, fmt, kind, metadata)

        if fmt == InputFormat.LIST_2CH:
            return DenoiseOutputList(x0_main=cast(list[Tensor], x0_main), x0_aux=cast(list[Tensor], x0_aux))
        return DenoiseOutput(x0_main=cast(Tensor, x0_main), x0_aux=cast(Tensor, x0_aux))

    @overload
    def denoise(self, x: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> Tensor: ...
    @overload
    def denoise(self, x: Sequence[Tensor], sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> list[Tensor]: ...
    def denoise(
        self,
        x: Tensor | Sequence[Tensor],
        sigma: Tensor,
        kind: Tensor,
        cond_mask: Tensor | None = None,
    ) -> Tensor | list[Tensor]:
        """Convenience: return only the main x0 prediction."""
        out = self.forward(x, sigma, kind, cond_mask=cond_mask)
        return out.x0_main

    @overload
    def denoise_aux(self, x: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> Tensor: ...
    @overload
    def denoise_aux(self, x: Sequence[Tensor], sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> list[Tensor]: ...
    def denoise_aux(
        self,
        x: Tensor | Sequence[Tensor],
        sigma: Tensor,
        kind: Tensor,
        cond_mask: Tensor | None = None,
    ) -> Tensor | list[Tensor]:
        out = self.forward(x, sigma, kind, cond_mask=cond_mask)
        return out.x0_aux

    @overload
    def predict_eps(self, x: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> Tensor: ...
    @overload
    def predict_eps(self, x: Sequence[Tensor], sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> list[Tensor]: ...
    def predict_eps(
        self,
        x: Tensor | Sequence[Tensor],
        sigma: Tensor,
        kind: Tensor,
        cond_mask: Tensor | None = None,
    ) -> Tensor | list[Tensor]:
        """Compute eps_hat = (x - x0_hat_main) / sigma."""
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x0_main, _ = self._denoise_both_2ch(x_2ch, sigma, kind, cond_mask=cond_mask)

        B = x_2ch.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        sigma_b = sigma.to(device=x_2ch.device, dtype=x_2ch.dtype).view(-1, 1, 1, 1)
        eps_hat_2ch = (x_2ch - x0_main) / sigma_b.clamp(min=1e-8)
        eps_hat_2ch = _apply_kind_constraints(eps_hat_2ch, kind)

        return _from_2ch(eps_hat_2ch, fmt, kind, metadata)

    @overload
    def score(self, x: Tensor, sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> Tensor: ...
    @overload
    def score(self, x: Sequence[Tensor], sigma: Tensor, kind: Tensor, cond_mask: Tensor | None = None) -> list[Tensor]: ...
    def score(
        self,
        x: Tensor | Sequence[Tensor],
        sigma: Tensor,
        kind: Tensor,
        cond_mask: Tensor | None = None,
    ) -> Tensor | list[Tensor]:
        """Compute score ≈ (x0_hat_main - x) / sigma^2."""
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x0_main, _ = self._denoise_both_2ch(x_2ch, sigma, kind, cond_mask=cond_mask)

        B = x_2ch.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(B)
        sigma_b = sigma.to(device=x_2ch.device, dtype=x_2ch.dtype).view(-1, 1, 1, 1)
        score_2ch = (x0_main - x_2ch) / sigma_b.square().clamp(min=1e-8)
        score_2ch = _apply_kind_constraints(score_2ch, kind)

        return _from_2ch(score_2ch, fmt, kind, metadata)

    @overload
    def normalize(self, x: Tensor, kind: Tensor) -> Tensor: ...
    @overload
    def normalize(self, x: Sequence[Tensor], kind: Tensor) -> list[Tensor]: ...
    def normalize(self, x: Tensor | Sequence[Tensor], kind: Tensor) -> Tensor | list[Tensor]:
        """Normalize input using per-kind fixed scales (mapping kinds only)."""
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x_norm = self._apply_kind_scale(x_2ch, kind, inverse=False)
        return _from_2ch(x_norm, fmt, kind, metadata)

    @overload
    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor: ...
    @overload
    def denormalize(self, x: Sequence[Tensor], kind: Tensor) -> list[Tensor]: ...
    def denormalize(self, x: Tensor | Sequence[Tensor], kind: Tensor) -> Tensor | list[Tensor]:
        """Denormalize output back to original scale (mapping kinds only)."""
        x_2ch, fmt, metadata = _to_2ch(x, kind)
        x_denorm = self._apply_kind_scale(x_2ch, kind, inverse=True)
        return _from_2ch(x_denorm, fmt, kind, metadata)

    def _apply_kind_scale(self, x: Tensor, kind: Tensor, inverse: bool = False) -> Tensor:
        result = x.clone()
        for k in kind.unique():
            k_idx = int(k.item())
            if not is_mapping_kind(k_idx):
                continue
            scale = self._kind_scale_lookup.get(k_idx, 1.0)
            idx = kind == k
            if inverse:
                result[idx] = result[idx] / scale
            else:
                result[idx] = result[idx] * scale
        return result

    def clamp_x0(self, x: Tensor, kind: Tensor, limit: float | None = None) -> Tensor:
        """Clamp x0 predictions to physical ranges with phase preservation for complex."""
        if limit is None:
            limit = self.sample_clip
        result = x.clone()
        eps = 1e-8
        for k in kind.unique():
            k_idx = int(k.item())
            idx = kind == k
            if is_complex_kind(k_idx):
                x_k = result[idx]
                mag = (x_k[:, 0] ** 2 + x_k[:, 1] ** 2).sqrt()
                mag_clipped = mag.clamp(0.0, limit)
                scale_factor = mag_clipped / (mag + eps)
                result[idx, 0] = x_k[:, 0] * scale_factor
                result[idx, 1] = x_k[:, 1] * scale_factor
            else:
                result[idx, 0] = result[idx, 0].clamp(0.0, limit)
                result[idx, 1] = 0.0
        return result

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load state dict with support for Lightning/EMA checkpoint formats.

        Handles:
        - EMA shadow weights (prefers these if available)
        - Lightning checkpoint format (strips 'model.' prefix)
        - Plain state dicts

        Parameters
        ----------
        state_dict
            State dict to load. Can be a full checkpoint dict containing
            'ema' or 'state_dict' keys, or a plain model state dict.
        strict
            Whether to require exact key matching.
        assign
            Whether to assign tensors instead of copying.

        Returns
        -------
        _IncompatibleKeys
            Named tuple with missing and unexpected keys.
        """
        # Handle EMA checkpoint format
        sd: Mapping[str, Any] = state_dict
        if "ema" in sd and "shadow" in sd["ema"]:
            sd = sd["ema"]["shadow"]
        # Handle Lightning checkpoint format
        elif "state_dict" in sd:
            sd = sd["state_dict"]

        # Strip "model." prefix if present
        if any(k.startswith("model.") for k in sd):
            sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}

        # Compiled training checkpoints and eager inference use different wrapper
        # paths, but the underlying tensor names and shapes must match strictly.
        target_keys = {key.replace("_orig_mod.", ""): key for key in super().state_dict()}
        sd = {target_keys.get(key.replace("_orig_mod.", ""), key): value for key, value in sd.items()}

        return cast(
            torch.nn.modules.module._IncompatibleKeys,
            super().load_state_dict(sd, strict=strict, assign=assign),
        )

    def load_checkpoint(self, path: str | Path) -> torch.nn.modules.module._IncompatibleKeys:
        """Load checkpoint from file path.

        Convenience wrapper around load_state_dict that handles file loading.

        Parameters
        ----------
        path
            Path to checkpoint file (.pt or .ckpt).

        Returns
        -------
        _IncompatibleKeys
            Named tuple with missing and unexpected keys.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return self.load_state_dict(checkpoint)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        unet: torch.nn.Module | None = None,
        map_location: torch.device | str | None = None,
        strict: bool = True,
        eval_mode: bool = True,
        compile_model: bool = True,
        weights: str = "auto",
    ) -> DiffusionModel:
        """Instantiate a diffusion model from a Lightning-style checkpoint.

        Parameters
        ----------
        path
            Path to checkpoint file (.ckpt or .pt).
        unet
            Optional UNet instance. If not provided, the checkpoint must contain
            ``cli_config.model.unet``.
        map_location
            Torch load map_location.
        strict
            Whether to require exact key matching when loading weights.
        eval_mode
            If True, puts the model in eval mode before returning.
        compile_model
            Compile the UNet. Disable for bounded CPU smoke tests or eager inference.
        weights
            Select "ema" (requires saved EMA weights), "raw", or legacy-compatible
            "auto" (EMA when present, otherwise raw). Evaluations should select
            explicitly and record their choice.
        """
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        from score_qmri.utils.checkpoint import model_spec

        unet_spec, diffusion_config = model_spec(checkpoint, require_unet=unet is None)
        diffusion = DiffusionConfig.from_mapping(diffusion_config)

        if unet is None:
            unet = _instantiate_unet_from_spec(unet_spec, cond_dim=diffusion.cond_dim)

        model = cls(unet=unet, diffusion=diffusion, compile_model=compile_model)

        if weights == "auto":
            weights = "ema" if "ema" in checkpoint and "shadow" in checkpoint["ema"] else "raw"
        if weights == "ema":
            if "ema" not in checkpoint or "shadow" not in checkpoint["ema"]:
                raise ValueError("Checkpoint has no EMA shadow; explicitly select weights='raw' if intended")
            state = checkpoint["ema"]["shadow"]
        elif weights == "raw":
            state = checkpoint["state_dict"]
        else:
            raise ValueError("weights must be 'auto', 'ema' or 'raw'")
        model.load_state_dict(state, strict=strict)
        model.checkpoint_info = {
            "path": str(path),
            "global_step": checkpoint.get("global_step"),
            "weights": weights,
            "unet": unet_spec,
            "diffusion": diffusion_config,
        }
        if eval_mode:
            model.eval()
        return model
