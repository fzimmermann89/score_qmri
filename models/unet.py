"""Simple UNet architecture for diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .film_conditioning import apply_film


class ConvBlock(nn.Module):
    """Double convolution block with GroupNorm and optional FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, use_film: bool = False) -> None:
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(
        self,
        x: Tensor,
        film_scale: Tensor | None = None,
        film_shift: Tensor | None = None,
    ) -> Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        if self.use_film and film_scale is not None:
            out = apply_film(out, film_scale, film_shift)

        out = F.relu(self.norm2(self.conv2(out)))
        if self.use_film and film_scale is not None:
            out = apply_film(out, film_scale, film_shift)
        return out


class DownBlock(nn.Module):
    """Downsampling block: conv block followed by max pool."""

    def __init__(self, in_ch: int, out_ch: int, use_film: bool = False) -> None:
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, use_film)
        self.pool = nn.MaxPool2d(2)

    def forward(
        self,
        x: Tensor,
        film_scale: Tensor | None = None,
        film_shift: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (pooled_output, skip_connection)."""
        skip = self.block(x, film_scale, film_shift)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, use_film: bool = False) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch, use_film)

    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        film_scale: Tensor | None = None,
        film_shift: Tensor | None = None,
    ) -> Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, film_scale, film_shift)


class UNet(nn.Module):
    """Simple UNet for diffusion models with optional FiLM conditioning.

    Parameters
    ----------
    in_channels : int
        Input channels (default: 2 for real+imag or data+zeros).
    out_channels : int
        Output channels (default: 2).
    base_channels : int
        Base number of channels, doubled at each level.
    num_blocks : int
        Number of down/up blocks.
    use_film : bool
        Whether to use FiLM conditioning.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.use_film = use_film

        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Build encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for _ in range(num_blocks):
            self.down_blocks.append(DownBlock(ch, ch * 2, use_film))
            ch *= 2

        self.bottleneck = ConvBlock(ch, ch, use_film)

        # Build decoder
        self.up_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.up_blocks.append(UpBlock(ch, ch // 2, use_film))
            ch //= 2

        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def _get_film(
        self, film_params: dict[str, Tensor] | None, idx: int
    ) -> tuple[Tensor | None, Tensor | None]:
        """Extract FiLM scale/shift for a block."""
        if not self.use_film or film_params is None:
            return None, None
        scales = film_params.get("scales")
        if scales is None:
            return None, None
        return scales[:, idx : idx + 1], film_params["shifts"][:, idx : idx + 1]

    def forward(self, x: Tensor, film_params: dict[str, Tensor] | None = None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).
        film_params : dict[str, Tensor] | None
            FiLM parameters with 'scales' and 'shifts', each (B, num_blocks).

        Returns
        -------
        Tensor
            Output tensor of shape (B, out_channels, H, W).
        """
        x = self.initial_conv(x)

        skips = []
        for i, down in enumerate(self.down_blocks):
            scale, shift = self._get_film(film_params, i)
            x, skip = down(x, scale, shift)
            skips.append(skip)

        scale, shift = self._get_film(film_params, self.num_blocks - 1)
        x = self.bottleneck(x, scale, shift)

        for i, up in enumerate(self.up_blocks):
            scale, shift = self._get_film(film_params, self.num_blocks - 1 - i)
            x = up(x, skips.pop(), scale, shift)

        return self.final_conv(x)
