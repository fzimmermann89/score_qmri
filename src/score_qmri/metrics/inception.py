"""InceptionV3 feature extractor for FID."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torchvision
from torch import Tensor


class InceptionV3Features(torch.nn.Module):
    """InceptionV3 returning pool3 (2048-dim) features.

    Parameters
    ----------
    weights_path
        Path to RadImageNet Inception v3 weights.
    """

    def __init__(self, weights_path: Path | None = None) -> None:
        super().__init__()
        if weights_path is None:
            raise ValueError(
                "Inception weights path is required. Please download the RadImageNet Inception v3 weights and provide --inception-weights."
            )
        resolved = weights_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Inception weights not found at {resolved}. "
                "Please download the RadImageNet Inception v3 weights and place them at this path."
            )

        self._model = torchvision.models.inception_v3(
            weights=None,
            aux_logits=False,
            init_weights=False,
        )
        self._model.transform_input = False
        _load_weights(self._model, resolved)

    def forward(self, x: Tensor) -> Tensor:
        """Extract pool3 features.

        Parameters
        ----------
        x
            Input tensor of shape (B, 3, H, W) in [0, 1].

        Returns
        -------
        Tensor
            Feature tensor of shape (B, 2048).
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {tuple(x.shape)}")

        x = torch.clamp(x, 0.0, 1.0)
        x = torch.nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        x = torch.clamp(x * 2 - 1, -1.0, 1.0)

        x = self._model.Conv2d_1a_3x3(x)
        x = self._model.Conv2d_2a_3x3(x)
        x = self._model.Conv2d_2b_3x3(x)
        x = self._model.maxpool1(x)
        x = self._model.Conv2d_3b_1x1(x)
        x = self._model.Conv2d_4a_3x3(x)
        x = self._model.maxpool2(x)
        x = self._model.Mixed_5b(x)
        x = self._model.Mixed_5c(x)
        x = self._model.Mixed_5d(x)
        x = self._model.Mixed_6a(x)
        x = self._model.Mixed_6b(x)
        x = self._model.Mixed_6c(x)
        x = self._model.Mixed_6d(x)
        x = self._model.Mixed_6e(x)
        x = self._model.Mixed_7a(x)
        x = self._model.Mixed_7b(x)
        x = self._model.Mixed_7c(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        return torch.flatten(x, 1)


def _load_weights(model: torch.nn.Module, weights_path: Path) -> None:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Weights file does not contain a state_dict mapping")

    state_dict = {str(k): v for k, v in state.items()}
    state_dict = _strip_prefix_if_present(state_dict, "model.")
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing or unexpected:
        raise ValueError(f"Unexpected keys while loading Inception weights. Missing keys: {missing}. Unexpected keys: {unexpected}.")


def _strip_prefix_if_present(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict
