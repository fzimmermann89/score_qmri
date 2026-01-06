"""UNet architecture with conditioning for diffusion models."""

import torch
from torch import Tensor


class GroupNorm32(torch.nn.GroupNorm):
    """GroupNorm that always computes in float32 for numerical stability."""

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.float()).to(input.dtype)


class ResBlock(torch.nn.Module):
    """Residual block with FiLM conditioning and ReZero initialization."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            GroupNorm32(min(8, out_ch), out_ch),
            torch.nn.SiLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            GroupNorm32(min(8, out_ch), out_ch),
            torch.nn.SiLU(),
        )
        self.cond_proj = torch.nn.Linear(cond_dim, out_ch * 2)
        self.skip = torch.nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h: Tensor = self.block1(x)
        scale, shift = self.cond_proj(cond)[:, :, None, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.block2(h)
        out: Tensor = self.skip(x) + self.alpha * h
        return out


class Downsample(torch.nn.Module):
    """Strided convolution for 2x spatial downsampling."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.conv(x)
        return out


class Upsample(torch.nn.Module):
    """PixelShuffle for 2x spatial upsampling."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(ch, ch * 4, 3, padding=1)
        self.shuffle = torch.nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.shuffle(self.conv(x))
        return out


class EncoderBlock(torch.nn.Module):
    """ResBlock with optional downsampling. Returns (forward, skip)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, downsample: bool = False) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.down = Downsample(out_ch) if downsample else None

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.res(x, cond)
        out = self.down(skip) if self.down is not None else skip
        return out, skip


class DecoderBlock(torch.nn.Module):
    """Optional upsampling with ResBlock and skip concat. Returns (forward, skip)."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, cond_dim: int, upsample: bool = False) -> None:
        super().__init__()
        self.up = Upsample(in_ch) if upsample else None
        self.res = ResBlock(in_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        if self.up is not None:
            x = self.up(x)
        h = self.res(torch.cat([x, skip], dim=1), cond)
        return h, h


class OutputHead(torch.nn.Module):
    """ResBlock output head with optional input skip connection."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128, skip_ch: int = 0) -> None:
        super().__init__()
        self.res = ResBlock(in_ch + skip_ch, in_ch, cond_dim)
        self.conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: Tensor, cond: Tensor, skip: Tensor | None = None) -> Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out: Tensor = self.conv(self.res(x, cond))
        return out


class UNet(torch.nn.Module):
    """UNet with dual prediction heads and FiLM conditioning.

    Predicts both clean image (x₀) and noise (ε) through separate output heads.
    Architecture uses symmetric EncoderBlock/DecoderBlock with skip connections.

    Input: 2 channels (real/imag for complex MRI)
    Output: (x0_pred, noise_pred), each (B, 2, H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8, 16),
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.input_conv = torch.nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder = torch.nn.ModuleList()
        encoder_channels: list[int] = []
        ch = base_channels

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for i in range(num_res_blocks):
                is_last_at_level = i == num_res_blocks - 1
                do_down = is_last_at_level and level < len(channel_mult) - 1
                self.encoder.append(EncoderBlock(ch, out_ch, cond_dim, downsample=do_down))
                encoder_channels.append(out_ch)
                ch = out_ch

        self.middle = ResBlock(ch, ch, cond_dim)

        self.decoder = torch.nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks):
                is_first_at_level = i == 0
                do_up = is_first_at_level and level < len(channel_mult) - 1
                skip_ch = encoder_channels.pop()
                self.decoder.append(DecoderBlock(ch, out_ch, skip_ch, cond_dim, upsample=do_up))
                ch = out_ch

        self.x0_head = OutputHead(ch, 2, cond_dim, skip_ch=in_channels)
        self.noise_head = OutputHead(ch, 2, cond_dim, skip_ch=in_channels)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        x_in = x
        x = self.input_conv(x)

        skips = []
        for block in self.encoder:
            x, skip = block(x, cond)
            skips.append(skip)

        x = self.middle(x, cond)

        for block in self.decoder:
            x, _ = block(x, skips.pop(), cond)

        return self.x0_head(x, cond, x_in), self.noise_head(x, cond, x_in)
