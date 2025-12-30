"""UNet architecture with conditioning for diffusion models."""

import torch
from torch import Tensor


class ResBlock(torch.nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        """
        Initialize residual block.

        Parameters
        ----------
        in_ch
            Input channels.
        out_ch
            Output channels.
        cond_dim
            Conditioning vector dimension.
        """
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.GroupNorm(min(8, out_ch), out_ch),
            torch.nn.SiLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.GroupNorm(min(8, out_ch), out_ch),
            torch.nn.SiLU(),
        )
        self.cond_proj = torch.nn.Linear(cond_dim, out_ch * 2)
        self.skip = torch.nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Forward pass with conditioning.

        Parameters
        ----------
        x
            Input of shape (B, C, H, W).
        cond
            Conditioning vector of shape (B, cond_dim).

        Returns
        -------
        Output of shape (B, out_ch, H, W).
        """
        h = self.block1(x)
        scale, shift = self.cond_proj(cond)[:, :, None, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.block2(h)
        result: Tensor = h + self.skip(x)
        return result


class DownBlock(torch.nn.Module):
    """Downsampling block: ResBlock + strided Conv."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        """
        Initialize downsampling block.

        Parameters
        ----------
        in_ch
            Input channels.
        out_ch
            Output channels.
        cond_dim
            Conditioning vector dimension.
        """
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.down = torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Returns
        -------
        tuple
            (downsampled output, skip connection).
        """
        h = self.res(x, cond)
        return self.down(h), h


class UpBlock(torch.nn.Module):
    """Upsampling block: PixelShuffle + concat skip + ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, cond_dim: int = 128) -> None:
        """
        Initialize upsampling block.

        Parameters
        ----------
        in_ch
            Input channels.
        out_ch
            Output channels.
        skip_ch
            Skip connection channels.
        cond_dim
            Conditioning vector dimension.
        """
        super().__init__()
        self.up_conv = torch.nn.Conv2d(in_ch, in_ch * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = torch.nn.PixelShuffle(2)
        self.res = ResBlock(in_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> Tensor:
        """Forward pass concatenating skip connection after pixelshuffle upsampling."""
        x_up = self.pixel_shuffle(self.up_conv(x))
        result: Tensor = self.res(torch.cat([x_up, skip], dim=1), cond)
        return result


class OutputHead(torch.nn.Module):
    """ResNet output head for predictions."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        """
        Initialize output head.

        Parameters
        ----------
        in_ch
            Input channels.
        out_ch
            Output channels.
        cond_dim
            Conditioning vector dimension.
        """
        super().__init__()
        self.res = ResBlock(in_ch, in_ch, cond_dim)
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Forward pass through ResBlock and final conv."""
        result: Tensor = self.conv(self.res(x, cond))
        return result


class UNet(torch.nn.Module):
    """UNet with dual prediction heads and FiLM conditioning.

    Predicts both clean image (x₀) and noise (ε) through separate output heads.
    This enables:
    - Standard diffusion training via noise prediction (L_simple)
    - Auxiliary denoising task via x₀ prediction (SSIM + L1)
    - Reuse as standalone denoiser outside diffusion sampling

    Input: 2 channels (real/imag for complex MRI)
    Output: (x0_pred, noise_pred), each (B, 2, H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_blocks: int = 4,
        cond_dim: int = 128,
    ) -> None:
        """
        Initialize UNet.

        Parameters
        ----------
        in_channels
            Input channels.
        base_channels
            Base channel count (doubled at each level).
        num_blocks
            Number of down/up block pairs.
        cond_dim
            Conditioning vector dimension.
        """
        super().__init__()
        self.num_blocks: int = num_blocks
        self.cond_dim: int = cond_dim

        self.input_conv = torch.nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = torch.nn.ModuleList()
        ch = base_channels
        for _ in range(num_blocks):
            self.down_blocks.append(DownBlock(ch, ch * 2, cond_dim))
            ch *= 2

        self.middle = ResBlock(ch, ch, cond_dim)

        skip_chs = [base_channels * (2 ** (i + 1)) for i in range(num_blocks)][::-1]
        self.up_blocks = torch.nn.ModuleList()
        for skip_ch in skip_chs:
            self.up_blocks.append(UpBlock(ch, ch // 2, skip_ch, cond_dim))
            ch //= 2

        self.x0_head = OutputHead(base_channels, 2, cond_dim)
        self.noise_head = OutputHead(base_channels, 2, cond_dim)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x
            Input of shape (B, 2, H, W).
        cond
            Conditioning vector of shape (B, cond_dim).

        Returns
        -------
        (x0_pred, noise_pred), each of shape (B, 2, H, W).
        """
        x = self.input_conv(x)

        skips = []
        for down in self.down_blocks:
            x, skip = down(x, cond)
            skips.append(skip)

        x = self.middle(x, cond)

        for up in self.up_blocks:
            x = up(x, skips.pop(), cond)

        return self.x0_head(x, cond), self.noise_head(x, cond)
