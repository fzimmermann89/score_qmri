"""Joint T1/complex-M0 saturation recovery with separate learned priors.

Input NPZ: t1 (H,W), m0 (H,W), times (T,), binary unshifted mask
(T,H,W), RSS-normalized csm (C,H,W). Optional measurement is (T,1,C,H,W).
T1 and times are in seconds. Without measurement, simulate Gaussian noise.
M0 is used as a reference, never as a known reconstruction parameter.

Examples:
    python examples/qmri_joint.py sample model.ckpt case.npz results 0.2383 --method dps
    python examples/qmri_joint.py sample model.ckpt case.npz results 0.2383 --method daps
    python examples/qmri_joint.py plot results overview --t1_colors lipari.txt

The M0 scale is case-specific and must match training normalization.
Both methods are approximate; sample spread is not calibrated.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import matplotlib
import numpy as np
import torch
from jsonargparse import CLI
from matplotlib.colors import ListedColormap, Normalize
from mrpro.operators import LinearOperator, models
from scipy.ndimage import gaussian_filter
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler
from score_qmri.sampling._problem import SamplingProblem
from score_qmri.sampling._utils import conjugate_gradient, normal_like
from score_qmri.sampling.edm_sampler import karras_sigmas

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def sample(
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
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(case) as arrays:
        t1 = torch.as_tensor(arrays["t1"], dtype=torch.float32, device=device)
        m0 = torch.as_tensor(arrays["m0"], dtype=torch.complex64, device=device)
        times = torch.as_tensor(arrays["times"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(arrays["mask"], dtype=torch.float32, device=device)
        csm = torch.as_tensor(arrays["csm"], dtype=torch.complex64, device=device)
        operator = Encoding(csm, mask).to(device)
        if "measurement" in arrays:
            measurement = torch.as_tensor(arrays["measurement"], dtype=torch.complex64, device=device)
        else:
            signal = models.SaturationRecovery(times).to(device)(m0[None], t1.clamp_min(0.05)[None])[0]
            clean = operator(signal)[0]
            generator = torch.Generator(device=device).manual_seed(seed)
            measurement = clean + noise_std * normal_like(clean, generator) * mask[:, None, None]
    np.savez_compressed(
        output / "case.npz",
        t1=t1.cpu().numpy(),
        m0=m0.cpu().numpy(),
        times=times.cpu().numpy(),
        mask=mask.cpu().numpy(),
        csm=csm.cpu().numpy(),
        measurement=measurement.cpu().numpy(),
    )
    fit_t1, fit_m0 = cg_fit(operator, measurement, times, regularization)
    np.savez_compressed(output / "cg-fit.npz", t1=fit_t1.cpu().numpy(), m0=fit_m0.cpu().numpy())
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


def scores(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Masked errors and local Gaussian SSIM, averaged at valid mask centers.

    SSIM uses an 11-pixel window, sigma 1.5, population covariance and the
    full reference range. Inputs are not zeroed outside the evaluation mask.
    """
    x, y = image.astype(float), reference.astype(float)
    ux, uy = gaussian_filter(x, 1.5, truncate=3.5), gaussian_filter(y, 1.5, truncate=3.5)
    vx = gaussian_filter(x * x, 1.5, truncate=3.5) - ux * ux
    vy = gaussian_filter(y * y, 1.5, truncate=3.5) - uy * uy
    covariance = gaussian_filter(x * y, 1.5, truncate=3.5) - ux * uy
    c1, c2 = (0.01 * np.ptp(y)) ** 2, (0.03 * np.ptp(y)) ** 2
    local = ((2 * ux * uy + c1) * (2 * covariance + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    valid = mask.copy()
    valid[:5] = valid[-5:] = False
    valid[:, :5] = valid[:, -5:] = False
    error = x[mask] - y[mask]
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))), "ssim": float(local[valid].mean())}


def plot(results: Path, output: Path, t1_colors: Path | None = None, mask_threshold: float = 0.2) -> None:
    """Plot saved ensembles as PNG/PDF, with unclipped metrics in JSON.

    M0 summaries use magnitudes. SD is sample SD, not the standard error
    of the mean. Coverage is descriptive for the reference-selected mask.
    t1_colors accepts a Lipari RGB table; otherwise use Matplotlib's magma.
    """
    with np.load(results / "case.npz") as case, np.load(results / "cg-fit.npz") as fit:
        truth = {"t1": case["t1"].copy(), "m0": np.abs(case["m0"])}
        baseline = {"t1": fit["t1"].copy(), "m0": np.abs(fit["m0"])}
    data = {}
    for method in ("dps", "daps"):
        with np.load(results / f"{method}.npz") as samples:
            data[method] = {"t1": samples["t1"].astype(float), "m0": np.abs(samples["m0"]).astype(float)}
    mask = truth["m0"] > mask_threshold
    coords = np.argwhere(mask)
    low, high = np.maximum(coords.min(0) - 4, 0), np.minimum(coords.max(0) + 5, mask.shape)
    crop = np.s_[low[0] : high[0], low[1] : high[1]]
    t1_cmap = ListedColormap(np.loadtxt(t1_colors)) if t1_colors else plt.get_cmap("magma")
    error_cmap = plt.get_cmap("magma").with_extremes(bad="white")
    metrics: dict = {"mask": f"reference |M0| > {mask_threshold}", "mask_pixels": int(mask.sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    ):
        fig = plt.figure(figsize=(10.2, 10.6))
        grid = fig.add_gridspec(
            7,
            5,
            height_ratios=[1, 1, 1, 0.16, 1, 1, 1],
            width_ratios=[1, 1, 1, 1, 1.06],
            left=0.035,
            right=0.98,
            top=0.975,
            bottom=0.035,
            hspace=0.31,
            wspace=0.055,
        )
        for start, parameter, label, cmap, maximum, error_max, factor in [
            (0, "t1", r"$T_1$", t1_cmap, 2.5, 300, 1000),
            (4, "m0", r"$|M_0|$", plt.get_cmap("viridis"), 2, 0.15, 1),
        ]:
            cmap = cmap.with_extremes(bad="white")
            reference = truth[parameter]
            for column, limit, palette, units in [
                (2, maximum, cmap, label + (" (s)" if parameter == "t1" else " (a.u.)")),
                (3, error_max, error_cmap, "SD / error" + (" (ms)" if parameter == "t1" else " (a.u.)")),
            ]:
                host = fig.add_subplot(grid[start, column])
                host.set_axis_off()
                bar = fig.colorbar(
                    plt.cm.ScalarMappable(norm=Normalize(0, limit), cmap=palette),
                    cax=host.inset_axes((0.08, 0.47, 0.84, 0.065)),
                    orientation="horizontal",
                    extend="max",
                )
                bar.set_ticks([0, limit / 2, limit])
                bar.set_label(units)
            coverage_ax = fig.add_subplot(grid[start + 1 : start + 3, 4])
            coverage_ax.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8)
            rows = [(start, "cg_fit", [reference, baseline[parameter]])]
            for index, (method, color, style) in enumerate([("dps", "#0072B2", "-"), ("daps", "#D55E00", "--")]):
                draws = data[method][parameter]
                mean, sd = draws.mean(0), draws.std(0, ddof=1)
                rows.append((start + index + 1, method, [draws[0], mean, factor * sd, factor * np.abs(mean - reference)]))
                ordered = np.sort(draws[:, mask], axis=0)
                count = len(draws)
                trim = np.arange((count - 2) // 2, -1, -1)
                expected = np.r_[(count - 1 - 2 * trim) / (count + 1), 1.0]
                observed = np.r_[((ordered[trim] <= reference[mask]) & (reference[mask] <= ordered[count - 1 - trim])).mean(1), 1.0]
                coverage_ax.plot(expected, observed, color=color, linestyle=style, lw=1, label=method.upper())
                metrics[f"{parameter}_{method}"] = {
                    "draws": count,
                    "expected_coverage": expected.tolist(),
                    "observed_coverage": observed.tolist(),
                }
            for row, method, fields in rows:
                for column, field in enumerate(fields):
                    ax = fig.add_subplot(grid[row, column])
                    ax.imshow(
                        np.where(mask, field, np.nan)[crop],
                        vmin=0,
                        vmax=error_max if column >= 2 else maximum,
                        cmap=error_cmap if column >= 2 else cmap,
                        interpolation="nearest",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if row == start:
                        ax.set_title(("Reference " if column == 0 else "CG+fit ") + label, pad=2)
                    elif method == "dps":
                        ax.set_title(["Sample", f"Mean (n={len(data[method][parameter])})", "SD", "|Mean − ref|"][column], pad=2)
                    if row != start and column == 0:
                        ax.set_ylabel(method.upper(), labelpad=7)
                    if column == 1 or (row != start and column == 0):
                        values = scores(field, reference, mask)
                        mae = f"{factor * values['mae']:.0f} ms" if parameter == "t1" else f"{values['mae']:.3f}"
                        ax.set_xlabel(f"MAE {mae}   SSIM {values['ssim']:.3f}", labelpad=2)
                        if row == start:
                            metrics[f"{parameter}_cg_fit"] = values
                        else:
                            metrics[f"{parameter}_{method}"]["sample" if column == 0 else "mean"] = values
            coverage_ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xticks=[0, 0.5, 1],
                yticks=[0, 0.5, 1],
                ylabel="Observed",
                title=label + " interval coverage",
            )
            if start == 0:
                coverage_ax.legend(frameon=False, fontsize=7, loc="lower right")
            else:
                coverage_ax.set_xlabel("Expected")
        for extension in ("png", "pdf"):
            fig.savefig(output.with_suffix("." + extension), dpi=600)
        plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    output.with_suffix(".txt").write_text(
        "Reference |M0| defines the display and evaluation mask. First saved draw, ensemble mean, sample SD (ddof=1), "
        "and absolute mean error. Color limits affect display only; metrics use unclipped values. "
        "Central order-statistic intervals have nominal coverage (N-1-2k)/(N+1) under exchangeability. "
        "The endpoint (1,1) represents the unbounded interval; connecting lines are visual guides. "
        "Single-case coverage with a reference-derived mask does not establish population calibration.\n"
    )


if __name__ == "__main__":
    CLI({"sample": sample, "plot": plot})
