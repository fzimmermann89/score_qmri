"""UNet architecture with conditioning for diffusion models."""

import math
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor

UpsampleMode = Literal["nearest", "pixelshuffle"]


class CoordConcat(torch.nn.Module):
    """Concatenate normalized (x, y) coordinate channels to input.

    Breaks translation invariance by providing absolute spatial position.
    Caches the coordinate grid for efficiency when resolution is unchanged.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cached_grid: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        if (
            self._cached_grid is None
            or self._cached_grid.shape[-2:] != (H, W)
            or self._cached_grid.device != x.device
            or self._cached_grid.dtype != x.dtype
        ):
            y_coords = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            x_coords = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self._cached_grid = torch.stack([grid_y, grid_x], dim=0)

        coords = self._cached_grid.unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


class SinusoidalPosEmb2D(torch.nn.Module):
    """2D sinusoidal positional embeddings for attention layers.

    Generates separate embeddings for height and width dimensions,
    then combines them. Caches for efficiency when resolution is unchanged.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self._cached_emb: Tensor | None = None

    def _get_1d_sincos(self, positions: Tensor, dim: int) -> Tensor:
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=positions.device, dtype=torch.float32) * -(math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = positions[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        if self._cached_emb is None or self._cached_emb.shape[-2:] != (H, W) or self._cached_emb.device != x.device:
            dim_y = self.channels // 2
            dim_x = self.channels - dim_y

            y_pos = torch.arange(H, device=x.device)
            x_pos = torch.arange(W, device=x.device)

            y_emb = self._get_1d_sincos(y_pos, dim_y)
            x_emb = self._get_1d_sincos(x_pos, dim_x)

            y_emb = y_emb[:, None, :].expand(H, W, -1)
            x_emb = x_emb[None, :, :].expand(H, W, -1)

            emb = torch.cat([y_emb, x_emb], dim=-1)
            emb = emb.permute(2, 0, 1).unsqueeze(0)
            self._cached_emb = emb

        return self._cached_emb.to(dtype=x.dtype).expand(B, -1, -1, -1)


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

    def __init__(self, channels: int, head_dim: int = 64, positional_embedding: bool = False) -> None:
        super().__init__()
        if channels % head_dim != 0:
            raise ValueError(f"channels={channels} must be divisible by head_dim={head_dim}")

        self.head_dim = head_dim
        self.num_heads = channels // head_dim

        self.norm = GroupNorm32(get_groups(channels), channels)
        self.to_qkv = torch.nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(channels, channels, 1)
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))
        self.pos_emb = SinusoidalPosEmb2D(channels) if positional_embedding else None

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        x_input = x + self.pos_emb(x) if self.pos_emb is not None else x
        h = self.norm(x_input)

        qkv = self.to_qkv(h)  # (B, 3*heads*head_dim, H, W)
        q, k, v = rearrange(
            qkv,
            "b (qkv heads d) h w -> qkv b heads (h w) d",
            qkv=3,
            heads=self.num_heads,
            d=self.head_dim,
        ).unbind(0)

        dropout = 0.1 if self.training else 0.0
        out: Tensor = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout)

        out = rearrange(out, "b heads (h w) d -> b (heads d) h w", h=H, w=W)
        if x.is_contiguous(memory_format=torch.channels_last):
            out = out.contiguous(memory_format=torch.channels_last)
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
        positional_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, cond_dim)
        self.attn = Attention(out_ch, head_dim, positional_embedding) if use_attention else None
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
        positional_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.up = Upsample(in_ch, upsample_mode) if upsample else None
        self.res = ResBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.attn = Attention(out_ch, head_dim, positional_embedding) if use_attention else None

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
        positional_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.positional_embedding = positional_embedding

        num_levels = len(channel_mult)
        attention_set = {lvl if lvl >= 0 else num_levels + lvl for lvl in attention_levels}

        self.coord_concat = CoordConcat() if positional_embedding else None
        input_ch = in_channels + 2 if positional_embedding else in_channels
        self.input_conv = torch.nn.Conv2d(input_ch, base_channels, 3, padding=1)

        self.encoder = torch.nn.ModuleList()
        encoder_channels: list[int] = []
        ch = base_channels

        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            use_attn = level in attention_set
            for i in range(num_res_blocks):
                is_last_at_level = i == num_res_blocks - 1
                do_down = is_last_at_level and level < num_levels - 1
                self.encoder.append(
                    EncoderBlock(
                        ch,
                        out_ch,
                        cond_dim,
                        downsample=do_down,
                        use_attention=use_attn,
                        head_dim=head_dim,
                        positional_embedding=positional_embedding,
                    )
                )
                encoder_channels.append(out_ch)
                ch = out_ch

        self.middle = ResBlock(ch, ch, cond_dim)
        last_level = num_levels - 1
        self.middle_attn: Attention | None = Attention(ch, head_dim, positional_embedding) if last_level in attention_set else None

        self.decoder = torch.nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            use_attn = level in attention_set
            for i in range(num_res_blocks):
                is_first_at_level = i == 0
                do_up = is_first_at_level and level < num_levels - 1
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
                        positional_embedding=positional_embedding,
                    )
                )
                ch = out_ch

        self.x0_head = OutputHead(ch, 2, cond_dim, skip_ch=in_channels)
        self.noise_head = OutputHead(ch, 2, cond_dim, skip_ch=in_channels)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        x_in = x
        if self.coord_concat is not None:
            x = self.coord_concat(x)
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
