"""DDPM sampler for diffusion models."""

import torch
from torch import Tensor

from score_qmri.models.diffusion import DiffusionModel


class DDPMSampler:
    """DDPM sampler with continuous time."""

    def __init__(self, model: DiffusionModel, num_steps: int = 1000) -> None:
        self.model = model
        self.num_steps = num_steps

    def sample(
        self,
        shape: tuple[int, ...],
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Sample from model using DDPM. Shape is (B, H, W), dtype from initial_noise."""
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
        x0_pred = torch.complex(x0_pred.real.clamp(-6, 6), x0_pred.imag.clamp(-6, 6)) if x0_pred.is_complex() else x0_pred.clamp(-6, 6)

        if t_prev[0] <= 0.0:
            return x0_pred

        ab_t = self.model.alpha_bar(t).view(-1, 1, 1)
        ab_prev = self.model.alpha_bar(t_prev).view(-1, 1, 1)

        one_minus_ab_t = (1.0 - ab_t).clamp(min=1e-8)
        ab_prev_safe = ab_prev.clamp(min=1e-8)

        alpha_t = ab_t / ab_prev_safe
        beta_t = 1.0 - alpha_t

        coef_x0 = ab_prev.clamp(min=0.0).sqrt() * beta_t / one_minus_ab_t
        coef_xt = alpha_t.clamp(min=0.0).sqrt() * (1.0 - ab_prev) / one_minus_ab_t
        mean = coef_x0 * x0_pred + coef_xt * x

        var = beta_t * (1.0 - ab_prev) / one_minus_ab_t
        return mean + var.clamp(min=0.0).sqrt() * torch.randn_like(x)
