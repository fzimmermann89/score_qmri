"""DDPM sampling algorithm for validation and inference."""

import torch
from torch import Tensor

from models.diffusion import DiffusionModel


class DDPMSampler:
    """DDPM reverse diffusion sampler.

    Implements the reverse diffusion process for generating samples
    from a trained diffusion model.
    """

    def __init__(self, model: DiffusionModel, num_steps: int | None = None) -> None:
        self.model = model
        self.num_steps = num_steps or model.num_timesteps

    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor | None = None,
    ) -> Tensor:
        """Sample from the diffusion model.

        Args:
            shape: Output shape (B, C, H, W).
            device: Device for computation.
            kind: Conditioning tensor of shape (B,). Defaults to zeros.

        Returns:
            Generated samples of shape (B, C, H, W).
        """
        with torch.no_grad():
            B = shape[0]
            if kind is None:
                kind = torch.zeros(B, dtype=torch.long, device=device)

            x = torch.randn(shape, device=device)
            timesteps = torch.linspace(self.model.num_timesteps - 1, 0, self.num_steps, device=device).long()

            for t in timesteps:
                x = self._ddpm_step(x, t.expand(B), kind)

            return x

    def _ddpm_step(self, x: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Execute single DDPM reverse step."""
        t_val = int(t[0].item())
        pred_x0: Tensor = self.model(x, t, kind)

        if t_val == 0:
            return pred_x0

        alpha_t = self.model.alphas_cumprod[t_val]
        alpha_prev = self.model.alphas_cumprod[t_val - 1]
        beta_t = self.model.betas[t_val]

        # Posterior mean coefficients
        coef_x0 = alpha_prev.sqrt() * beta_t / (1.0 - alpha_t)
        coef_xt = self.model.alphas[t_val].sqrt() * (1.0 - alpha_prev) / (1.0 - alpha_t)
        mean = coef_x0 * pred_x0 + coef_xt * x

        var = beta_t * (1.0 - alpha_prev) / (1.0 - alpha_t)
        result: Tensor = mean + var.sqrt() * torch.randn_like(x)
        return result

    def sample_conditional(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor,
    ) -> Tensor:
        """Sample with explicit conditioning. Alias for sample()."""
        return self.sample(shape, device, kind)
