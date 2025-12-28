"""DDPM sampler for diffusion models."""

import torch
from torch import Tensor

from models.diffusion import DiffusionModel


class DDPMSampler:
    """DDPM reverse diffusion sampler."""

    def __init__(self, model: DiffusionModel, num_steps: int | None = None) -> None:
        """
        Initialize DDPM sampler.

        Parameters
        ----------
        model
            Diffusion model instance.
        num_steps
            Number of sampling steps. Defaults to model's num_timesteps.
        """
        self.model = model
        self.num_steps = num_steps or model.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor | None = None,
    ) -> Tensor:
        """
        Sample from model using DDPM.

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        device
            Computation device.
        kind
            Dataset kind indices of shape (B,). Defaults to zeros.

        Returns
        -------
        Tensor
            Samples of shape (B, 2, H, W).
        """
        B = shape[0]
        if kind is None:
            kind = torch.zeros(B, dtype=torch.long, device=device)

        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.model.num_timesteps - 1, 0, self.num_steps, device=device).long()

        for t in timesteps:
            x = self._step(x, t.expand(B), kind)

        return x

    def _step(self, x: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Execute single DDPM reverse step."""
        t_val = int(t[0].item())
        output = self.model(x, t, kind)

        if t_val == 0:
            return self.model.predict_x0(x, t, output.noise)

        alpha_t = self.model.alphas_cumprod[t_val]
        alpha_prev = self.model.alphas_cumprod[t_val - 1]
        beta_t = self.model.betas[t_val]

        x0_pred = self.model.predict_x0(x, t, output.noise)

        coef_x0 = alpha_prev.sqrt() * beta_t / (1.0 - alpha_t)
        coef_xt = self.model.alphas[t_val].sqrt() * (1.0 - alpha_prev) / (1.0 - alpha_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        var = beta_t * (1.0 - alpha_prev) / (1.0 - alpha_t)
        return mean + var.sqrt() * torch.randn_like(x)
