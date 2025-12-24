"""Diffusion model wrapper around UNet."""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .film_conditioning import FilmConditioning
from .unet import UNet


class DiffusionModel(nn.Module):
    """Diffusion model with noise scheduling and FiLM conditioning.

    Wraps a UNet denoiser with linear noise schedule and dataset-kind
    conditioning via FiLM.

    Parameters
    ----------
    unet : UNet | None
        UNet model. If None, creates default with 2 in/out channels.
    num_kinds : int
        Number of dataset kinds for FiLM conditioning.
    num_timesteps : int
        Number of diffusion timesteps.
    beta_start : float
        Starting noise schedule value.
    beta_end : float
        Ending noise schedule value.
    """

    def __init__(
        self,
        unet: UNet | None = None,
        num_kinds: int = 3,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.unet = unet or UNet(in_channels=2, out_channels=2, use_film=True)
        self.film_conditioning = FilmConditioning(
            num_kinds=num_kinds, num_blocks=self.unet.num_blocks
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def prepare_input(self, image: Tensor, kind: Tensor) -> Tensor:
        """Prepare two-channel input from dataset sample.

        For magnitude/mapping data: [data, zeros]
        For complex data: expects image to already have 2 channels [real, imag]

        Parameters
        ----------
        image : Tensor
            Image tensor of shape (B, H, W) or (B, 2, H, W) for complex.
        kind : Tensor
            Kind tensor of shape (B,). 0=complex, 1=magnitude, 2=mapping.

        Returns
        -------
        Tensor
            Two-channel input of shape (B, 2, H, W).
        """
        if image.ndim == 4 and image.shape[1] == 2:
            return image

        if image.ndim == 4:
            image = image.squeeze(1)

        # Stack image with zeros for second channel
        zeros = torch.zeros_like(image)
        return rearrange([image, zeros], "c b h w -> b c h w")

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample from forward diffusion q(x_t | x_0).

        Parameters
        ----------
        x0 : Tensor
            Clean images of shape (B, C, H, W).
        t : Tensor
            Timesteps of shape (B,).
        noise : Tensor | None
            Optional noise tensor. If None, sampled from N(0, I).

        Returns
        -------
        Tensor
            Noisy images x_t of shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = rearrange(self.sqrt_alphas_cumprod[t], "b -> b 1 1 1")
        sqrt_one_minus = rearrange(self.sqrt_one_minus_alphas_cumprod[t], "b -> b 1 1 1")
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def forward(self, x_t: Tensor, t: Tensor, kind: Tensor) -> Tensor:
        """Predict x0 from noisy x_t.

        Parameters
        ----------
        x_t : Tensor
            Noisy images of shape (B, C, H, W).
        t : Tensor
            Timesteps of shape (B,).
        kind : Tensor
            Dataset kind of shape (B,).

        Returns
        -------
        Tensor
            Predicted x0 of shape (B, C, H, W).
        """
        film_params = self.film_conditioning(kind)
        return self.unet(x_t, film_params)
