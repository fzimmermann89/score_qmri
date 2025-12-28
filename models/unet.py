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
        out: Tensor = F.relu(self.norm1(self.conv1(x)))
        if self.use_film and film_scale is not None and film_shift is not None:
            out = apply_film(out, film_scale, film_shift)

        out = F.relu(self.norm2(self.conv2(out)))
        if self.use_film and film_scale is not None and film_shift is not None:
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
        skip: Tensor = self.block(x, film_scale, film_shift)
        pooled: Tensor = self.pool(skip)
        return pooled, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connection concatenation."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, use_film: bool = False) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch // 2 + skip_ch, out_ch, use_film)

    def forward(
        self,
        x: Tensor,
        skip: Tensor,
        film_scale: Tensor | None = None,
        film_shift: Tensor | None = None,
    ) -> Tensor:
        upsampled: Tensor = self.upsample(x)
        concatenated = torch.cat([upsampled, skip], dim=1)
        out: Tensor = self.block(concatenated, film_scale, film_shift)
        return out


class UNet(nn.Module):
    """UNet for diffusion with dual output heads (x0 and noise prediction).

    Input: 2 channels (real/imag for complex, data/zeros for magnitude)
    Output: 4 channels (2 for x0 prediction, 2 for noise prediction)
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.use_film = use_film

        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for _ in range(num_blocks):
            self.down_blocks.append(DownBlock(ch, ch * 2, use_film))
            ch *= 2

        self.bottleneck = ConvBlock(ch, ch, use_film)

        skip_channels = [base_channels * (2 ** (i + 1)) for i in range(num_blocks)]
        skip_channels.reverse()

        self.up_blocks = nn.ModuleList()
        for skip_ch in skip_channels:
            self.up_blocks.append(UpBlock(ch, ch // 2, skip_ch, use_film))
            ch //= 2

        # Dual output heads: x0 prediction and noise prediction
        self.x0_head = nn.Conv2d(base_channels, 2, 3, padding=1)
        self.noise_head = nn.Conv2d(base_channels, 2, 3, padding=1)

    def _get_film(self, film_params: dict[str, Tensor] | None, idx: int) -> tuple[Tensor | None, Tensor | None]:
        """Extract FiLM scale/shift for a specific block index."""
        if not self.use_film or film_params is None:
            return None, None
        scales = film_params.get("scales")
        if scales is None:
            return None, None
        return scales[:, idx : idx + 1], film_params["shifts"][:, idx : idx + 1]

    def forward(self, x: Tensor, film_params: dict[str, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass through UNet.

        Args:
            x: Input tensor of shape (B, 2, H, W).
            film_params: Optional dict with 'scales' and 'shifts' tensors for FiLM.

        Returns:
            Tuple of (x0_pred, noise_pred), each of shape (B, 2, H, W).
        """
        x = self.initial_conv(x)

        skips: list[Tensor] = []
        for i, down in enumerate(self.down_blocks):
            scale, shift = self._get_film(film_params, i)
            x, skip = down(x, scale, shift)
            skips.append(skip)

        scale, shift = self._get_film(film_params, self.num_blocks - 1)
        x = self.bottleneck(x, scale, shift)

        for i, up in enumerate(self.up_blocks):
            scale, shift = self._get_film(film_params, self.num_blocks - 1 - i)
            x = up(x, skips.pop(), scale, shift)

        x0_pred: Tensor = self.x0_head(x)
        noise_pred: Tensor = self.noise_head(x)
        return x0_pred, noise_pred
