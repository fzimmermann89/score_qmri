"""Reverse SDE sampler using Euler-Maruyama discretization.

Solves the reverse-time VP-SDE:
    dx = [-½β(t)x - β(t)∇ₓ log p(xₜ)] dt + √β(t) dw̄

where dw̄ is reverse-time Brownian motion.
"""

import torch
from einops import rearrange
from torch import Tensor

from models.diffusion import DiffusionModel


class SDESampler:
    """Euler-Maruyama sampler for reverse VP-SDE.

    Allows flexible discretization: choose any number of steps at inference
    time without retraining.
    """

    def __init__(self, model: DiffusionModel, num_steps: int = 1000) -> None:
        """
        Parameters
        ----------
        model
            VP-SDE diffusion model instance.
        num_steps
            Number of Euler-Maruyama discretization steps.
        """
        self.model = model
        self.num_steps = num_steps

    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """
        Sample by solving reverse SDE with Euler-Maruyama.

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        device
            Computation device.
        kind
            Dataset kind indices (B,). Defaults to zeros.
        initial_noise
            Optional fixed initial noise for reproducibility.
        """
        with torch.no_grad():
            B = shape[0]
            if kind is None:
                kind = torch.zeros(B, dtype=torch.long, device=device)
            kind = kind.to(device)

            x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
            dt = 1.0 / self.num_steps
            timesteps = torch.linspace(1.0, dt, self.num_steps, device=device)

            for t_val in timesteps:
                t = t_val.expand(B)
                x = self._euler_maruyama_step(x, t, dt, kind)

            return x

    def _euler_maruyama_step(self, x: Tensor, t: Tensor, dt: float, kind: Tensor) -> Tensor:
        """Single Euler-Maruyama step: x_{t-dt} = x_t - drift·dt + diffusion·√dt·z."""
        beta_t = rearrange(self.model.beta(t), "b -> b 1 1 1")
        score = self.model.score(x, t, kind)

        drift = -0.5 * beta_t * x - beta_t * score
        diffusion = beta_t.sqrt()
        noise = torch.randn_like(x)

        x_new: Tensor = x - drift * dt + diffusion * (dt**0.5) * noise
        return x_new


class ProbabilityFlowSampler:
    """Probability flow ODE sampler (deterministic).

    Solves the probability flow ODE:
        dx = [f(x,t) - ½g(t)²∇ₓ log p(xₜ)] dt

    This is a deterministic ODE that has the same marginals as the SDE,
    enabling exact likelihood computation and deterministic sampling.
    """

    def __init__(self, model: DiffusionModel, num_steps: int = 1000) -> None:
        """
        Parameters
        ----------
        model
            VP-SDE diffusion model instance.
        num_steps
            Number of ODE integration steps.
        """
        self.model = model
        self.num_steps = num_steps

    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor | None = None,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """
        Sample by solving probability flow ODE (deterministic).

        Parameters
        ----------
        shape
            Output shape (B, 2, H, W).
        device
            Computation device.
        kind
            Dataset kind indices (B,). Defaults to zeros.
        initial_noise
            Optional fixed initial noise for reproducibility.
        """
        with torch.no_grad():
            B = shape[0]
            if kind is None:
                kind = torch.zeros(B, dtype=torch.long, device=device)
            kind = kind.to(device)

            x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
            dt = 1.0 / self.num_steps
            timesteps = torch.linspace(1.0, dt, self.num_steps, device=device)

            for t_val in timesteps:
                t = t_val.expand(B)
                x = self._euler_step(x, t, dt, kind)

            return x

    def _euler_step(self, x: Tensor, t: Tensor, dt: float, kind: Tensor) -> Tensor:
        """Single Euler step: x_{t-dt} = x_t - drift·dt."""
        beta_t = rearrange(self.model.beta(t), "b -> b 1 1 1")
        score = self.model.score(x, t, kind)

        drift = -0.5 * beta_t * x - 0.5 * beta_t * score
        x_new: Tensor = x - drift * dt
        return x_new
