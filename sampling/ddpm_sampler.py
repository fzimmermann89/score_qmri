"""DDPM sampler for diffusion models."""

import torch
from einops import rearrange
from torch import Tensor

from models.diffusion import DiffusionModel


class DDPMSampler:
    """DDPM sampler with continuous time.

    Uses the reverse diffusion process to generate samples from pure Gaussian noise.
    The sampler follows the DDPM (Denoising Diffusion Probabilistic Models) algorithm
    adapted for continuous time t ∈ [0, 1] used in VP-SDE diffusion.

    Key equations:
        - Predict x₀ from noise: x₀ = (xₜ - σ(t)·ε) / √ᾱ(t)
        - Compute mean: μₜ = ᾱₜ₋₁·x₀ + (1 - ᾱₜ₋₁)·xₜ₋₁
        - Compute variance: Σₜ = βₜ·(1 - ᾱₜ₋₁)²·ᾱₜ₋₁
        - Sample: xₜ₋₁ ~ N(μₜ, Σₜ)

    Parameters
    ----------
    model
        Diffusion model instance.
    num_steps
        Number of denoising steps.
    """

    def __init__(self, model: DiffusionModel, num_steps: int = 1000) -> None:
        self.model = model
        self.num_steps = num_steps

    def sample(
        self,
        shape: tuple[int, ...],
        kind: Tensor | None = None,
    ) -> Tensor:
        """
        Sample from model using DDPM reverse process.

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        kind
            Dataset kind indices of shape (B,). Defaults to zeros.

        Returns
        -------
        Tensor
            Generated sample of shape (B, 2, H, W).
        """
        device = self.model.device
        with torch.no_grad():
            B = shape[0]
            if kind is None:
                kind = torch.zeros(B, dtype=torch.long, device=device)

            x = torch.randn(shape, device=device)
            timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

            for i in range(self.num_steps):
                t = timesteps[i].expand(B)
                t_prev = timesteps[i + 1].expand(B)
                x = self._step(x, t, t_prev, kind)

            return x

    def _step(self, x: Tensor, t: Tensor, t_prev: Tensor, kind: Tensor) -> Tensor:
        """Execute single DDPM reverse step."""
        output = self.model(x, t, kind)
        x0_pred = self.model.predict_x0(x, t, output.noise)

        if t_prev[0].item() <= 0.0:
            return x0_pred

        ab_t = rearrange(self.model.alpha_bar(t), "b -> b 1 1 1")
        ab_prev = rearrange(self.model.alpha_bar(t_prev), "b -> b 1 1 1")

        # α_t = ᾱ_t / ᾱ_{t-1}, β_t = 1 - α_t
        alpha_t = ab_t / ab_prev.clamp(min=1e-8)
        beta_t = 1.0 - alpha_t

        coef_x0 = ab_prev.sqrt() * beta_t / (1.0 - ab_t).clamp(min=1e-8)
        coef_xt = alpha_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
        mean = coef_x0 * x0_pred + coef_xt * x

        var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
        return mean + var.sqrt() * torch.randn_like(x)
