"""UNet architecture with conditioning for diffusion models."""

from typing import Literal

import torch
from torch import Tensor

UpsampleMode = Literal["nearest", "pixelshuffle"]


class GroupNorm32(torch.nn.GroupNorm):
    """GroupNorm that always computes in float32 for numerical stability."""

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.float()).to(input.dtype)


class Attention(torch.nn.Module):
    """Efficient self-attention using SDPA (Flash Attention compatible).

    Uses fused QKV projection, pre-norm, and ReZero residual connection.
    PyTorch's scaled_dot_product_attention auto-selects the optimal backend
    (Flash Attention, Memory Efficient, or Math) based on inputs and hardware.
    """

    def __init__(self, channels: int, head_dim: int = 64) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = max(1, channels // head_dim)
        inner_dim = self.num_heads * head_dim

        self.norm = GroupNorm32(get_groups(channels), channels)
        self.to_qkv = torch.nn.Conv2d(channels, inner_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Sequential(
            torch.nn.Conv2d(inner_dim, channels, 1),
            torch.nn.Dropout(0.0),
        )
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.to_qkv(h)  # (B, 3*inner_dim, H, W)
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)  # (B, 3, heads, head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)

        out: Tensor = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 1, 3, 2)  # (B, heads, head_dim, N)
        out = out.reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.to_out(out)

        return x + self.alpha * out


def get_groups(channels: int) -> int:
    """Get the largest valid group size (<= 32) for GroupNorm."""
    for groups in [32, 16, 8, 4, 2]:
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(torch.nn.Module):
    """Residual block with FiLM conditioning and ReZero initialization.

    Uses Pre-Norm architecture (GroupNorm -> SiLU -> Conv) for better gradient flow.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128) -> None:
        super().__init__()
        groups1 = get_groups(in_ch)
        self.block1 = torch.nn.Sequential(
            GroupNorm32(groups1, in_ch),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        groups2 = get_groups(out_ch)
        self.block2 = torch.nn.Sequential(
            GroupNorm32(groups2, out_ch),
            torch.nn.SiLU(),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
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
    """2x spatial upsampling with selectable method.

    Parameters
    ----------
    ch
        Number of input/output channels.
    mode
        Upsampling method:
        - "nearest": Nearest-neighbor interpolation + 3×3 conv (Stable Diffusion style)
        - "pixelshuffle": PixelShuffle (can cause periodic artifacts)
    """

    def __init__(self, ch: int, mode: UpsampleMode = "nearest") -> None:
        super().__init__()
        self.mode: UpsampleMode = mode
        if mode == "pixelshuffle":
            self.conv = torch.nn.Conv2d(ch, ch * 4, 3, padding=1)
            self.shuffle: torch.nn.PixelShuffle | None = torch.nn.PixelShuffle(2)
        else:
            self.conv = torch.nn.Conv2d(ch, ch, 3, padding=1)
            self.shuffle = None

    def forward(self, x: Tensor) -> Tensor:
        if self.shuffle is not None:
            out: Tensor = self.shuffle(self.conv(x))
            return out
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        out = self.conv(x)
        return out


class EncoderBlock(torch.nn.Module):
    """ResBlock with optional attention and downsampling. Returns (forward, skip)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        downsample: bool = False,
        use_attention: bool = False,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.attn = Attention(out_ch, head_dim) if use_attention else None
        self.down = Downsample(out_ch) if downsample else None

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.res(x, cond)
        if self.attn is not None:
            skip = self.attn(skip)
        out = self.down(skip) if self.down is not None else skip
        return out, skip


class DecoderBlock(torch.nn.Module):
    """Optional upsampling with ResBlock, attention, and skip concat."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_ch: int,
        cond_dim: int,
        upsample: bool = False,
        upsample_mode: UpsampleMode = "nearest",
        use_attention: bool = False,
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.up = Upsample(in_ch, upsample_mode) if upsample else None
        self.res = ResBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.attn = Attention(out_ch, head_dim) if use_attention else None

    def forward(self, x: Tensor, skip: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        if self.up is not None:
            x = self.up(x)
        h = self.res(torch.cat([x, skip], dim=1), cond)
        if self.attn is not None:
            h = self.attn(h)
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
    """UNet with dual prediction heads, FiLM conditioning, and optional attention.

    Predicts both clean image (x₀) and noise (ε) through separate output heads.
    Architecture uses symmetric EncoderBlock/DecoderBlock with skip connections.
    Attention is applied at specified resolution levels for global context.

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
        upsample_mode: UpsampleMode = "nearest",
        attention_levels: tuple[int, ...] = (),
        head_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        attention_set = set(attention_levels)

        self.input_conv = torch.nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder = torch.nn.ModuleList()
        encoder_channels: list[int] = []
        ch = base_channels

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            use_attn = level in attention_set
            for i in range(num_res_blocks):
                is_last_at_level = i == num_res_blocks - 1
                do_down = is_last_at_level and level < len(channel_mult) - 1
                self.encoder.append(EncoderBlock(ch, out_ch, cond_dim, downsample=do_down, use_attention=use_attn, head_dim=head_dim))
                encoder_channels.append(out_ch)
                ch = out_ch

        self.middle = ResBlock(ch, ch, cond_dim)
        last_level = len(channel_mult) - 1
        self.middle_attn: Attention | None = Attention(ch, head_dim) if last_level in attention_set else None

        self.decoder = torch.nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            use_attn = level in attention_set
            for i in range(num_res_blocks):
                is_first_at_level = i == 0
                do_up = is_first_at_level and level < len(channel_mult) - 1
                skip_ch = encoder_channels.pop()
                self.decoder.append(
                    DecoderBlock(
                        ch,
                        out_ch,
                        skip_ch,
                        cond_dim,
                        upsample=do_up,
                        upsample_mode=upsample_mode,
                        use_attention=use_attn,
                        head_dim=head_dim,
                    )
                )
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
        if self.middle_attn is not None:
            x = self.middle_attn(x)

        for block in self.decoder:
            x, _ = block(x, skips.pop(), cond)

        return self.x0_head(x, cond, x_in), self.noise_head(x, cond, x_in)
