"""Joint T1/complex-M0 saturation recovery with separate learned priors.

Input NPZ: t1 (H,W), m0 (H,W), times (T,), binary unshifted mask
(T,H,W), RSS-normalized csm (C,H,W). Optional measurement is (T,1,C,H,W).
T1 and times are in seconds. Without measurement, simulate Gaussian noise.
M0 is used as a reference, never as a known reconstruction parameter.

Run with --help for arguments. For example:
    python examples/qmri_joint.py model.ckpt case.npz results 0.2383 --method dps
    python examples/qmri_joint.py model.ckpt case.npz results 0.2383 --method daps
The example scale is case-specific; derive it from your training normalization.
Use qmri_joint_plot.py on the output folder.
Settings reproduce an exploratory case; sample spread is not calibrated.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from jsonargparse import CLI
from mrpro.operators import LinearOperator, models
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling._utils import conjugate_gradient, normal_like
from score_qmri.sampling.edm_sampler import karras_sigmas


class Encoding(LinearOperator):
    """Unshifted orthonormal Cartesian FFT with known coil sensitivities."""

    csm: Tensor
    mask: Tensor

    def __init__(self, csm: Tensor, mask: Tensor) -> None:
        super().__init__()
        self.register_buffer("csm", csm)
        self.register_buffer("mask", mask)

    def forward(self, images: Tensor) -> tuple[Tensor]:
        return (torch.fft.fft2(images[:, :, None] * self.csm, norm="ortho") * self.mask[:, None, None],)

    def adjoint(self, data: Tensor) -> tuple[Tensor]:
        return ((torch.fft.ifft2(data * self.mask[:, None, None], norm="ortho") * self.csm.conj()).sum(2),)


class JointPrior:
    """Product prior in state coordinates [T1/4, 0, Re(M0), Im(M0)]."""

    def __init__(self, model: DiffusionModel, m0_prior_scale: float) -> None:
        self.model, self.scale = model, m0_prior_scale
        self.sigma_min = max(model.sigma_min, model.sigma_min / self.scale)
        self.sigma_max = min(model.sigma_max, model.sigma_max / self.scale)
        self.sigma_data = model.sigma_data

    def denoise(self, x: Tensor, sigma: Tensor, kind: Tensor) -> Tensor:
        batch = len(x)
        kinds = torch.tensor([DatasetKind.T1MAP, DatasetKind.M0_COMPLEX], device=x.device).repeat_interleave(batch)
        inputs = torch.cat([x[:, :2], self.scale * x[:, 2:]])
        levels = torch.cat([sigma, self.scale * sigma])
        result = self.model.denoise(inputs, levels, kinds)
        return torch.cat([result[:batch], result[batch:] / self.scale], dim=1)


class JointProblem:
    """Joint likelihood adapter for the existing DPS and DAPS step routines."""

    linear = False

    def __init__(self, operator: Encoding, measurement: Tensor, times: Tensor, batch: int, noise_std: float) -> None:
        self.operator, self.measurement = operator, measurement.moveaxis(1, 0)
        self.times, self.noise_std = times, noise_std
        self.shape = (batch, 4, *measurement.shape[-2:])
        self.kind = torch.zeros(batch, dtype=torch.long, device=measurement.device)
        self.signal = models.SaturationRecovery(times).to(measurement.device)

    def constrain(self, x: Tensor) -> Tensor:
        return torch.stack([x[:, 0], torch.zeros_like(x[:, 1]), x[:, 2], x[:, 3]], dim=1)

    def noise(self, generator: torch.Generator) -> Tensor:
        return self.constrain(torch.randn(self.shape, device=self.measurement.device, generator=generator))

    def decode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return 4 * x[:, 0], torch.complex(x[:, 2], x[:, 3])

    def residual(self, x: Tensor) -> Tensor:
        t1, m0 = self.decode(x)
        # Extend the signal below 50 ms; sampler outputs are not clipped.
        signal: Tensor
        (signal,) = self.signal(m0, t1.clamp_min(0.05))
        predicted: Tensor
        (predicted,) = self.operator(signal)
        return predicted.moveaxis(1, 0) - self.measurement

    @torch.no_grad()
    def gradient(self, x: Tensor) -> Tensor:
        t1, m0 = self.decode(x)
        safe = t1.clamp_min(0.05)
        exponential = torch.exp(-self.times[:, None, None, None] / safe[None])
        basis = 1 - exponential
        residual = self.operator(m0[None] * basis)[0] - self.measurement.moveaxis(0, 1)
        back = self.operator.H(residual)[0]
        derivative = -self.times[:, None, None, None] * exponential / safe[None].square() * (t1 > 0.05)[None]
        t1_gradient = 4 * (back.conj() * m0[None] * derivative).real.sum(0)
        m0_gradient = (back * basis).sum(0)
        return torch.stack([t1_gradient, torch.zeros_like(t1_gradient), m0_gradient.real, m0_gradient.imag], dim=1) / self.noise_std**2


@torch.no_grad()
def cg_fit(operator: Encoding, measurement: Tensor, times: Tensor, regularization: float) -> tuple[Tensor, Tensor]:
    """Tikhonov SENSE followed by grid T1 fitting with profiled complex M0."""
    rhs = operator.H(measurement.to(torch.complex128))[0]
    images = conjugate_gradient(lambda x: operator.H(operator(x)[0])[0] + regularization * x, rhs, 1000, 1e-6)
    grid = torch.linspace(0.05, 5.5, 2181, dtype=images.real.dtype, device=images.device)
    basis = 1 - torch.exp(-times[:, None] / grid)
    norm = basis.square().sum(0)
    t1 = torch.empty(images.shape[-2:], device=images.device)
    m0 = torch.empty_like(t1, dtype=torch.complex64)
    for start in range(0, t1.numel(), 2048):
        chunk = images[:, 0].flatten(1)[:, start : start + 2048]
        cross = basis.to(chunk).T @ chunk
        index = (cross.abs().square() / norm[:, None]).argmax(0)
        columns = torch.arange(len(index), device=index.device)
        t1.flatten()[start : start + len(index)] = grid[index].float()
        m0.flatten()[start : start + len(index)] = (cross[index, columns] / norm[index]).to(m0.dtype)
    return t1, m0


def main(
    checkpoint: Path,
    case: Path,
    output: Path,
    m0_prior_scale: float,
    method: Literal["dps", "daps"] = "dps",
    draws: int = 30,
    batch_size: int = 5,
    noise_std: float = 0.01,
    steps: int | None = None,
    guidance_scale: float = 1.2,
    langevin_steps: int = 600,
    langevin_step: float = 3e-6,
    sigma_min: float = 0.02,
    regularization: float = 0.01,
    seed: int = 42,
    device: str = "cuda",
) -> None:
    """Save draws and a CG+fit baseline; scales must match checkpoint training.

    m0_prior_scale maps physical M0 to its training coordinates, including
    dataset scaling. It multiplies both denoiser input and noise level, with
    inverse output scaling. It does not rescale measurements or their noise.
    Supplied measurements must use noise_std per real coordinate and be zero
    outside the mask. Reuse one case and output folder for both methods.
    """
    if not all(np.isfinite(v) and v > 0 for v in (m0_prior_scale, noise_std, regularization)) or min(draws, batch_size) < 1:
        raise ValueError("Scales, noise, regularization and sample counts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"{method}.npz").exists():
        raise FileExistsError(output / f"{method}.npz")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(case) as arrays:
        t1 = torch.as_tensor(arrays["t1"], dtype=torch.float32, device=device)
        m0 = torch.as_tensor(arrays["m0"], dtype=torch.complex64, device=device)
        times = torch.as_tensor(arrays["times"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(arrays["mask"], dtype=torch.float32, device=device)
        csm = torch.as_tensor(arrays["csm"], dtype=torch.complex64, device=device)
        if times.ndim != 1 or not (times > 0).all() or csm.ndim != 3:
            raise ValueError("Require positive recovery times and coil maps with shape (C,H,W)")
        if t1.ndim != 2 or m0.shape != t1.shape or mask.shape != (len(times), *t1.shape) or csm.shape[1:] != t1.shape:
            raise ValueError("Case shapes must follow the module docstring")
        if not all(x.isfinite().all() for x in (t1, m0, times, mask, csm)) or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Case must be finite with a binary mask")
        operator = Encoding(csm, mask).to(device)
        if "measurement" in arrays:
            measurement = torch.as_tensor(arrays["measurement"], dtype=torch.complex64, device=device)
        else:
            signal = models.SaturationRecovery(times).to(device)(m0[None], t1.clamp_min(0.05)[None])[0]
            clean = operator(signal)[0]
            generator = torch.Generator(device=device).manual_seed(seed)
            measurement = clean + noise_std * normal_like(clean, generator) * mask[:, None, None]
    if measurement.shape != (len(times), 1, len(csm), *t1.shape) or not measurement.isfinite().all():
        raise ValueError("Measurement must be finite with shape (T,1,C,H,W)")
    if (measurement * (1 - mask[:, None, None])).abs().any():
        raise ValueError("Measurement must be zero outside the mask")
    saved_case = {
        "t1": t1.cpu().numpy(),
        "m0": m0.cpu().numpy(),
        "times": times.cpu().numpy(),
        "mask": mask.cpu().numpy(),
        "csm": csm.cpu().numpy(),
        "measurement": measurement.cpu().numpy(),
    }
    if (output / "case.npz").exists():
        with np.load(output / "case.npz") as previous:
            if any(not np.array_equal(previous[key], value) for key, value in saved_case.items()):
                raise ValueError("Output folder belongs to a different case")
    else:
        np.savez_compressed(
            output / "case.npz",
            t1=saved_case["t1"],
            m0=saved_case["m0"],
            times=saved_case["times"],
            mask=saved_case["mask"],
            csm=saved_case["csm"],
            measurement=saved_case["measurement"],
        )
    if (output / "cg-fit.npz").exists():
        with np.load(output / "cg-fit.npz") as previous_fit:
            if float(previous_fit["regularization"]) != regularization:
                raise ValueError("Existing CG+fit baseline uses different regularization")
    else:
        fit_t1, fit_m0 = cg_fit(operator, measurement, times, regularization)
        np.savez_compressed(output / "cg-fit.npz", t1=fit_t1.cpu().numpy(), m0=fit_m0.cpu().numpy(), regularization=regularization)
    model = DiffusionModel.from_checkpoint(checkpoint, compile_model=False, weights="ema").to(device).eval().requires_grad_(False)
    prior = JointPrior(model, m0_prior_scale)
    cfg = (
        DPSConfig(num_steps=1000 if steps is None else steps, guidance="dps", guidance_scale=guidance_scale)
        if method == "dps"
        else DAPSConfig(
            num_steps=200 if steps is None else steps,
            sigma_min=sigma_min,
            reverse_sigma_min=prior.sigma_min,
            reverse_steps=10,
            langevin_steps=langevin_steps,
            langevin_step=langevin_step,
            langevin_step_min_ratio=1.0,
        )
    )
    if method == "daps" and not prior.sigma_min < sigma_min < prior.sigma_max:
        raise ValueError("Require prior noise floor < sigma_min < prior noise maximum")
    # The public API describes one map kind. These local adapters use the same
    # step routines for a product prior without extending that library API.
    sampler = DPSSampler(cast(DiffusionModel, prior), cfg) if isinstance(cfg, DPSConfig) else DAPSSampler(cast(DiffusionModel, prior), cfg)
    samples_t1, samples_m0 = [], []
    for batch in range(0, draws, batch_size):
        problem = JointProblem(operator, measurement, times, min(batch_size, draws - batch), noise_std)
        generator = torch.Generator(device=device).manual_seed(seed + 1 + batch)
        x = prior.sigma_max * problem.noise(generator)
        grid = karras_sigmas(
            cfg.num_steps,
            sigma_min=prior.sigma_min if method == "dps" else sigma_min,
            sigma_max=prior.sigma_max,
            rho=cfg.rho,
            device=torch.device(device),
        )
        if isinstance(sampler, DPSSampler):
            grid = torch.cat([grid, grid.new_zeros(1)])
            for current, following in zip(grid[:-1], grid[1:], strict=True):
                x = sampler._step(x, cast(SamplingProblem, problem), float(current), float(following), problem.noise(generator))
        else:
            for i, sigma in enumerate(grid):
                result = sampler._conditional(x, cast(SamplingProblem, problem), float(sigma), generator, ratio=i / cfg.num_steps)
                x = result if i == len(grid) - 1 else result + float(grid[i + 1]) * problem.noise(generator)
        t, m = problem.decode(x.detach())
        samples_t1.append(t.cpu().numpy())
        samples_m0.append(m.cpu().numpy())
        print(f"{method.upper()}: {batch + len(t)}/{draws} draws", flush=True)
    np.savez_compressed(output / f"{method}.npz", t1=np.concatenate(samples_t1), m0=np.concatenate(samples_m0))
    (output / f"{method}.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "case": str(case.resolve()),
                "m0_prior_scale": m0_prior_scale,
                "noise_std": noise_std,
                "seed": seed,
                "draws": draws,
                "batch_size": batch_size,
                "config": asdict(cfg),
                "scope": "Exploratory approximate sampling; not calibrated posterior uncertainty.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    CLI(main)
