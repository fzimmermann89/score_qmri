"""UNet architecture with conditioning for diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning via linear projection.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        cond_dim: Conditioning vector dimension.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

        # Single projection for scale and shift
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

        # Skip connection projection if dimensions differ
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Forward with conditioning.

        Args:
            x: Input of shape (B, C, H, W).
            cond: Conditioning vector of shape (B, cond_dim).
        """
        h = F.silu(self.norm1(self.conv1(x)))

        # Apply FiLM: scale and shift from conditioning
        scale_shift = self.cond_proj(cond)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class DownBlock(nn.Module):
    """Downsampling: ResBlock + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        h = self.res(x, cond)
        return self.pool(h), h


class UpBlock(nn.Module):
    """Upsampling: Upsample + concat skip + ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, cond_dim: int = 128) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.res = ResBlock(in_ch // 2 + skip_ch, out_ch, cond_dim)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        out: Tensor = self.res(torch.cat([self.up(x), skip], dim=1), cond)
        return out


class OutputHead(nn.Module):
    """ResNet output head for predictions."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, in_ch, cond_dim)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out: Tensor = self.conv(self.res(x, cond))
        return out


class UNet(nn.Module):
    """UNet with dual output heads and conditioning.

    Input: 2 channels (real/imag or data/zeros)
    Output: (x0_pred, noise_pred), each (B, 2, H, W)

    Args:
        in_channels: Input channels.
        base_channels: Base channel count (doubled at each level).
        num_blocks: Number of down/up block pairs.
        cond_dim: Conditioning vector dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for _ in range(num_blocks):
            self.down_blocks.append(DownBlock(ch, ch * 2, cond_dim))
            ch *= 2

        # Bottleneck
        self.bottleneck = ResBlock(ch, ch, cond_dim)

        # Decoder
        skip_chs = [base_channels * (2 ** (i + 1)) for i in range(num_blocks)][::-1]
        self.up_blocks = nn.ModuleList()
        for skip_ch in skip_chs:
            self.up_blocks.append(UpBlock(ch, ch // 2, skip_ch, cond_dim))
            ch //= 2

        # Output heads (ResNet blocks)
        self.x0_head = OutputHead(base_channels, 2, cond_dim)
        self.noise_head = OutputHead(base_channels, 2, cond_dim)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input of shape (B, 2, H, W).
            cond: Conditioning vector of shape (B, cond_dim).

        Returns:
            (x0_pred, noise_pred), each of shape (B, 2, H, W).
        """
        x = self.input_conv(x)

        skips = []
        for down in self.down_blocks:
            x, skip = down(x, cond)
            skips.append(skip)

        x = self.bottleneck(x, cond)

        for up in self.up_blocks:
            x = up(x, skips.pop(), cond)

        return self.x0_head(x, cond), self.noise_head(x, cond)
