"""DDPM sampling algorithm for validation and inference."""

import torch
from einops import rearrange
from torch import Tensor

from models.diffusion import DiffusionModel


class DDPMSampler:
    """DDPM reverse diffusion sampler.

    Parameters
    ----------
    model : DiffusionModel
        Diffusion model with noise schedule buffers.
    num_steps : int | None
        Number of sampling steps. Defaults to model.num_timesteps.
    """

    def __init__(self, model: DiffusionModel, num_steps: int | None = None) -> None:
        self.model = model
        self.num_steps = num_steps or model.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor | None = None,
    ) -> Tensor:
        """Sample from the diffusion model.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape (B, C, H, W).
        device : torch.device
            Device for computation.
        kind : Tensor | None
            Conditioning tensor of shape (B,). Defaults to zeros.

        Returns
        -------
        Tensor
            Generated samples of shape (B, C, H, W).
        """
        B = shape[0]
        if kind is None:
            kind = torch.zeros(B, dtype=torch.long, device=device)

        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(
            self.model.num_timesteps - 1, 0, self.num_steps, device=device
        ).long()

        for t in timesteps:
            x = self._ddpm_step(x, t.expand(B), kind)

        return x

    def _ddpm_step(self, x: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Execute single DDPM reverse step."""
        t_val = t[0].item()
        pred_x0 = self.model(x, t, kind)

        if t_val == 0:
            return pred_x0

        alpha_t = self.model.alphas_cumprod[t_val]
        alpha_prev = self.model.alphas_cumprod[t_val - 1]
        beta_t = self.model.betas[t_val]

        # Posterior mean coefficients for q(x_{t-1} | x_t, x_0)
        coef_x0 = alpha_prev.sqrt() * beta_t / (1.0 - alpha_t)
        coef_xt = self.model.alphas[t_val].sqrt() * (1.0 - alpha_prev) / (1.0 - alpha_t)
        mean = coef_x0 * pred_x0 + coef_xt * x

        # Posterior variance
        var = beta_t * (1.0 - alpha_prev) / (1.0 - alpha_t)

        return mean + var.sqrt() * torch.randn_like(x)

    def sample_conditional(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor,
    ) -> Tensor:
        """Sample with explicit conditioning. Alias for sample()."""
        return self.sample(shape, device, kind)
