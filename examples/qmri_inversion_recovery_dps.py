"""Reconstruct a real T1 map from synthetic single-coil inversion-recovery data.

The input NPZ contains t1 (seconds), m0, ti (seconds), k_mask (unshifted),
measurement (complex k-space), and adjoint_fit (seconds). M0 and phase are known.
"""

from pathlib import Path

import numpy as np
import torch
from jsonargparse import CLI
from mrpro.operators import FastFourierOp, LinearOperator, Operator, models

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import DPSConfig, DPSSampler


class KnownM0IR(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Reuse the inversion-recovery model with fixed M0 and a declared T1 guard."""

    m0: torch.Tensor

    def __init__(self, ti: torch.Tensor, m0: torch.Tensor) -> None:
        super().__init__()
        self.signal = models.InversionRecovery(ti)
        self.register_buffer("m0", m0)

    def forward(self, t1: torch.Tensor) -> tuple[torch.Tensor]:
        # This guard extends the likelihood to proposals below 50 ms;
        # it does not enforce a positive T1 prior or calibrated posterior.
        signal: torch.Tensor
        (signal,) = self.signal(self.m0, t1.clamp_min(0.05))
        return (signal,)


class UnshiftedImageFFT(LinearOperator):
    """Encode unshifted images with mrpro's centered Fourier operator."""

    def __init__(self) -> None:
        super().__init__()
        self.fft = FastFourierOp(dim=(-2, -1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        encoded: torch.Tensor
        (encoded,) = self.fft(torch.fft.fftshift(x, dim=(-2, -1)))
        return (encoded,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        return (torch.fft.ifftshift(self.fft.H(y)[0], dim=(-2, -1)),)


def main(checkpoint: Path, case: Path, output: Path, draws: int = 2, steps: int = 1000, device: str = "cuda") -> None:
    """Save independent DPS T1 draws and compare their errors with adjoint+fit."""
    arrays = np.load(case)
    truth = torch.as_tensor(arrays["t1"], device=device)
    ti = torch.as_tensor(arrays["ti"], device=device)
    m0 = torch.as_tensor(arrays["m0"], device=device)
    mask = torch.as_tensor(arrays["k_mask"], device=device)[:, None]
    observed = torch.as_tensor(arrays["measurement"], device=device)[:, None]
    # FastFourierOp uses centered FFTs; the saved case uses unshifted FFTs.
    fft = UnshiftedImageFFT().to(device)
    encoding = torch.fft.fftshift(mask, dim=(-2, -1)) * fft
    forward = encoding @ KnownM0IR(ti, m0).to(device)
    observed = torch.fft.fftshift(observed, dim=(-2, -1))
    template = truth.new_empty(draws, *truth.shape)
    model = DiffusionModel.from_checkpoint(checkpoint, compile_model=False, weights="ema").to(device).eval()
    sampler = DPSSampler(model, DPSConfig(num_steps=steps, guidance="dps", guidance_scale=0.3))
    (t1,) = sampler.sample(
        forward,
        observed,
        noise_std=0.01,
        template=template,
        kind=DatasetKind.T1MAP,
        measurement_batch_dim=1,
        generator=torch.Generator(device=device).manual_seed(20261023),
    )
    support = truth > 0.05
    for index, sample in enumerate(t1):
        rmse = ((sample - truth)[support].square().mean().sqrt() * 1000).item()
        print(f"DPS draw {index + 1}: T1 RMSE {rmse:.1f} ms")
    baseline = torch.as_tensor(arrays["adjoint_fit"], device=device)
    print(f"Adjoint+fit: T1 RMSE {((baseline - truth)[support].square().mean().sqrt() * 1000).item():.1f} ms")
    np.savez_compressed(output, t1=t1.cpu().numpy(), reference=truth.cpu().numpy(), adjoint_fit=arrays["adjoint_fit"])


if __name__ == "__main__":
    CLI(main)
