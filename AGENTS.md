## Project Stack

- Python 3.12
- PyTorch, PyTorch Lightning, Lightning CLI
- torchvision, kornia, einops
- Linting: ruff, ty, mypy
- Testing: pytest
- Package manager: uv

## Environment

**Always activate venv before running any tool or script:**

```bash
source .venv/bin/activate
```

```bash
# Linting/type checking
ruff check .
ruff format .
mypy .
ty check

# Tests
pytest
pytest tests/test_models/test_unet.py -v

# Training
python train.py --config config.yaml
```

## Project Structure

```
./
├── config.py                 # Global config / LightningCLI config
├── train.py                  # Training entry point
├── models/
│   ├── diffusion.py          # Diffusion model wrapper
│   ├── unet.py               # UNet architecture
│   └── film_conditioning.py  # FiLM layers for conditioning
├── datasets/
│   ├── base.py               # Base dataset class
│   ├── magnitude.py          # Magnitude image datasets
│   ├── complex.py            # Complex-valued datasets
│   ├── concat.py             # Dataset concatenation
│   ├── augmentation.py       # Data augmentations
│   ├── batch_sampler.py      # Custom batch sampling
│   ├── mapping.py            # qmri Mapping dataset
│   ├── lightning.py          # LightningDataModule
│   └── config.py             # Dataset configs
├── training/
│   ├── lightning_module.py   # LightningModule for training
│   ├── losses.py             # Loss functions
│   └── phase_augmentation.py # Phase augmentation for complex data
├── sampling/
│   └── ddpm_sampler.py       # DDPM sampling
├── tests/                    # pytest tests (mirror main structure)
└── pyproject.toml
```

## Code Style

### General

- Pythonic, concise. No boilerplate.
- `pathlib.Path` over `os.path`.
- f-strings for formatting.
- Modern syntax: `list[T]`, `dict[K, V]`, `X | None`.

### Type Hints

- All function signatures typed.
- `torch.Tensor` for tensors.
- Annotate shapes in comments: `# (B, C, H, W)`.

```python
def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # (B,C,H,W), (B,) -> (B,C,H,W)
    ...
```

### Docstrings

- NumPy style. NO types in docstrings.
- `__init__` parameters documented in `__init__` docstring.
- Document shapes, expected ranges, semantics.

```python
class UNet(nn.Module):
    """UNet with FiLM conditioning for diffusion."""

    def __init__(self, in_channels: int, base_dim: int = 64, time_emb_dim: int = 256):
        """
        Parameters
        ----------
        in_channels
            Input channels (2 for complex: real/imag).
        base_dim
            Base feature dimension, doubled at each downsampling.
        time_emb_dim
            Dimension of sinusoidal time embedding.
        """
        super().__init__()
        ...
```

### Comments

- NO trivial comments.
- Only explain non-obvious logic, numerical tricks, paper references.

```python
# Apply noise schedule from Ho et al. (2020) eq. 4
alpha_bar = torch.cumprod(1 - betas, dim=0)
```

## PyTorch Guidelines

### Performance

- Vectorize—no Python loops over batch/spatial dims.
- `torch.no_grad()` for inference.
- `torch.compile()` for hot paths.
- In-place ops when safe: `x.add_(1)`
- `pin_memory=True`, `num_workers>0` in DataLoader.
- Mixed precision via Lightning `precision="16-mixed"`.

### einops

```python
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# Good
x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=16, p2=16)
t_emb = repeat(t_emb, "b d -> b d h w", h=h, w=w)

# Avoid
x = x.view(...).permute(...).reshape(...)
```

make use of unflatten instead of view.
prefer moveaxis instead of permute (and einops.rearrange over permute)

### Lightning

- `LightningModule` in `training/lightning_module.py`.
- `LightningDataModule` in `datasets/lightning.py`.
- Use `LightningCLI` for config—no argparse.
- `self.log(..., sync_dist=True)` for multi-GPU.

### Use Libraries, Don't Reimplement

- **torchvision**: transforms, pretrained models
- **kornia**: gpu acceleratred augmentations

## Testing

- Mirror main structure in `tests/`.
- Use fixtures in `conftest.py` for common tensors/models.
- Test shapes, dtypes, gradient flow, edge cases.


## Complex MRI Conventions

- Store complex as 2-channel real tensor: `(B, 2, H, W)` where `[:,0]` = real, `[:,1]` = imag.
- Or use `torch.complex64` when native complex ops needed.
- k-space: apply `torch.fft.fft2` / `ifft2` with `norm="ortho"`.

## Imports
- do not impofr torch.nn as nn, or torch.nn.functional as F. Just use full path in the code.

