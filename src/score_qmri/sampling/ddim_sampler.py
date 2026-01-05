"""DDIM sampler for diffusion models."""

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) sampler."""

    def __init__(self, model: DiffusionModel, num_steps: int = 100, eta: float = 0.0) -> None:
        self.model = model
        self.num_steps = num_steps
        self.eta = eta

    def sample(
        self,
        shape: tuple[int, ...],
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Sample from model using DDIM. Shape is (B, H, W), dtype from initial_noise."""
        device = self.model.device
        B = shape[0]
        kind = torch.zeros(B, dtype=torch.long, device=device) if kind is None else kind.to(device)
        x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        with torch.no_grad():
            for i in range(self.num_steps):
                t = timesteps[i].expand(B)
                t_prev = timesteps[i + 1].expand(B)
                x = self._step(x, t, t_prev, kind)

        return x

    def _step(self, x: Tensor, t: Tensor, t_prev: Tensor, kind: Tensor) -> Tensor:
        output = self.model(x, t, kind)
        x0_pred = self.model.predict_x0(x, t, output.noise)

        ab_t = self.model.alpha_bar(t).view(-1, 1, 1)
        ab_prev = self.model.alpha_bar(t_prev).view(-1, 1, 1)

        one_minus_ab_t = (1 - ab_t).clamp(min=1e-8)
        one_minus_ab_prev = (1 - ab_prev).clamp(min=0.0)
        sigma_sq = self.eta**2 * (one_minus_ab_prev / one_minus_ab_t * (1 - ab_t / ab_prev.clamp(min=1e-8)))
        sigma = sigma_sq.clamp(min=0.0).sqrt()

        pred_dir = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt() * output.noise
        x_prev = ab_prev.sqrt() * x0_pred + pred_dir

        if self.eta > 0 and t_prev[0] > 1e-8:
            x_prev = x_prev + sigma * torch.randn_like(x)

        return x_prev


def create_fixed_noise(image_shape: tuple[int, int], seed: int = 42) -> dict[int, Tensor]:
    """Create reproducible noise for each dataset kind with correct dtype."""
    gen = torch.Generator().manual_seed(seed)
    result = {}
    for kind in DatasetKind:
        shape = (1, *image_shape)
        if kind.is_complex:
            noise = torch.complex(
                torch.randn(shape, generator=gen),
                torch.randn(shape, generator=gen),
            )
        else:
            noise = torch.randn(shape, generator=gen)
        result[int(kind)] = noise
    return result
