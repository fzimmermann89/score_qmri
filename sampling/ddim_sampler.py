"""DDIM sampling algorithm for validation and inference."""

import torch
from torch import Tensor

from models.diffusion import DiffusionModel


class DDIMSampler:
    """DDIM (Denoising Diffusion Implicit Models) sampler.

    Faster sampling with deterministic steps for reproducible validation.
    """

    def __init__(
        self,
        model: DiffusionModel,
        num_steps: int = 100,
        eta: float = 0.0,
    ) -> None:
        """Initialize DDIM sampler.

        Args:
            model: Diffusion model instance.
            num_steps: Number of sampling steps.
            eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM).
        """
        self.model = model
        self.num_steps = num_steps
        self.eta = eta

    def sample(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        kind: Tensor,
        initial_noise: Tensor | None = None,
    ) -> Tensor:
        """Sample from the diffusion model using DDIM.

        Args:
            shape: Output shape (B, 2, H, W).
            device: Device for computation.
            kind: Conditioning tensor of shape (B,).
            initial_noise: Fixed initial noise for reproducibility.

        Returns:
            Generated samples of shape (B, 2, H, W).
        """
        with torch.no_grad():
            B = shape[0]
            x = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)

            # Create timestep schedule (uniform spacing)
            timesteps = torch.linspace(
                self.model.num_timesteps - 1, 0, self.num_steps + 1, device=device
            ).long()

            for i in range(self.num_steps):
                t = timesteps[i].expand(B)
                t_prev = timesteps[i + 1].expand(B)

                x = self._ddim_step(x, t, t_prev, kind)

            return x

    def _ddim_step(self, x: Tensor, t: Tensor, t_prev: Tensor, kind: Tensor) -> Tensor:
        """Execute single DDIM reverse step.

        Uses the noise prediction head for sampling.
        """
        t_val = int(t[0].item())
        t_prev_val = int(t_prev[0].item())

        output = self.model(x, t, kind)
        noise_pred = output.noise_pred

        alpha_t = self.model.alphas_cumprod[t_val]
        alpha_prev = self.model.alphas_cumprod[t_prev_val] if t_prev_val >= 0 else torch.tensor(1.0)

        # Predict x0 from noise
        x0_pred = self.model.predict_x0_from_noise(x, t, noise_pred)

        # DDIM update
        sigma_t = (
            self.eta
            * ((1 - alpha_prev) / (1 - alpha_t)).sqrt()
            * (1 - alpha_t / alpha_prev).sqrt()
        )

        # Direction pointing to x_t
        dir_xt = (1 - alpha_prev - sigma_t**2).sqrt() * noise_pred

        # x_{t-1}
        x_prev: Tensor = alpha_prev.sqrt() * x0_pred + dir_xt

        if self.eta > 0 and t_prev_val > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma_t * noise

        return x_prev


def create_fixed_noise(
    num_kinds: int,
    image_shape: tuple[int, int],
    seed: int = 42,
) -> dict[int, Tensor]:
    """Create fixed noise tensors for each dataset kind.

    Args:
        num_kinds: Number of different dataset kinds.
        image_shape: (H, W) shape of images.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping kind index to fixed noise tensor of shape (1, 2, H, W).
    """
    generator = torch.Generator().manual_seed(seed)
    fixed_noise = {}
    for kind_idx in range(num_kinds):
        noise = torch.randn(1, 2, *image_shape, generator=generator)
        fixed_noise[kind_idx] = noise
    return fixed_noise
