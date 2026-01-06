"""Exponential Moving Average for model weights."""

import copy
from typing import Any, Generic, TypeVar

import torch
from torch import nn

T = TypeVar("T", bound=nn.Module)


class EMA(Generic[T]):
    """Exponential Moving Average of model parameters."""

    shadow: T

    def __init__(self, model: T, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: T) -> None:
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(model_p, 1 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"shadow": self.shadow.state_dict(), "decay": self.decay}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.shadow.load_state_dict(state_dict["shadow"])
        self.decay = state_dict["decay"]
