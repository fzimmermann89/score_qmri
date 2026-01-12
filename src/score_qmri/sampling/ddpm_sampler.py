"""DDPM sampler for diffusion models."""

import torch
from torch import Tensor

from score_qmri.models.diffusion import DiffusionModel, _apply_kind_constraints, _create_noise


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
        """Sample from model using DDPM. Shape is (B, 2, H, W)."""
        device = self.model.device
        B = shape[0]
        kind = torch.zeros(B, dtype=torch.long, device=device) if kind is None else kind.to(device)

        x = initial_noise.to(device) if initial_noise is not None else _create_noise(shape, kind, device, torch.float32)
        x = _apply_kind_constraints(x, kind)

        eps = 1e-3
        timesteps = torch.linspace(1.0 - eps, eps, self.num_steps + 1, device=device)

        with torch.no_grad():
            for i in range(self.num_steps):
                t = timesteps[i].expand(B)
                t_prev = timesteps[i + 1].expand(B)
                x = self._step(x, t, t_prev, kind)

        return x

    def _step(self, x: Tensor, t: Tensor, t_prev: Tensor, kind: Tensor) -> Tensor:
        output = self.model(x, t, kind)
        x0_pred = self.model.predict_x0(x, t, output.noise, kind)
        x0_pred = x0_pred.clamp(-6, 6)

        if t_prev[0] <= 1e-3:
            return x0_pred

        ab_t = self.model.alpha_bar(t).view(-1, 1, 1, 1)
        ab_prev = self.model.alpha_bar(t_prev).view(-1, 1, 1, 1)

        one_minus_ab_t = (1.0 - ab_t).clamp(min=1e-8)
        ab_prev_safe = ab_prev.clamp(min=1e-8)

        alpha_t = ab_t / ab_prev_safe
        beta_t = 1.0 - alpha_t

        coef_x0 = ab_prev.clamp(min=0.0).sqrt() * beta_t / one_minus_ab_t
        coef_xt = alpha_t.clamp(min=0.0).sqrt() * (1.0 - ab_prev) / one_minus_ab_t
        mean = coef_x0 * x0_pred + coef_xt * x

        var = beta_t * (1.0 - ab_prev) / one_minus_ab_t
        noise = _create_noise(x.shape, kind, x.device, x.dtype)
        x_prev = mean + var.clamp(min=0.0).sqrt() * noise

        return _apply_kind_constraints(x_prev, kind)
