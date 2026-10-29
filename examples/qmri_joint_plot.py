"""Plot saved joint T1/M0 ensembles without loading the model.

The folder contains case.npz (reference t1/m0), cg-fit.npz, dps.npz and
daps.npz (t1/m0 draws, N,H,W). For Lipari, pass its RGB text table through
--t1_colors. Run with --help for display limits and output paths.

Example: python examples/qmri_joint_plot.py results overview --t1_colors lipari.txt

Coverage pools pixels selected by reference |M0|>0.2. It is descriptive:
this truth-derived mask and a single case do not establish calibration.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from jsonargparse import CLI
from matplotlib.colors import ListedColormap, Normalize
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    if c1 == 0:
        raise ValueError("SSIM needs a nonconstant reference")
    local = ((2 * ux * uy + c1) * (2 * covariance + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    valid = mask.copy()
    valid[:5] = valid[-5:] = False
    valid[:, :5] = valid[:, -5:] = False
    if not valid.any():
        raise ValueError("SSIM needs mask centers at least five pixels from the image boundary")
    error = x[mask] - y[mask]
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))), "ssim": float(local[valid].mean())}


def coverage(draws: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central order-statistic coverage plus the exact unbounded endpoint."""
    ordered = np.sort(draws[:, mask], axis=0)
    count = len(draws)
    trim = np.arange((count - 2) // 2, -1, -1)
    expected = (count - 1 - 2 * trim) / (count + 1)
    truth = reference[mask]
    observed = ((ordered[trim] <= truth) & (truth <= ordered[count - 1 - trim])).mean(axis=1)
    return np.r_[expected, 1.0], np.r_[observed, 1.0]


def main(
    results: Path,
    output: Path,
    t1_colors: Path | None = None,
    mask_threshold: float = 0.2,
    t1_max: float = 2.5,
    m0_max: float = 2.0,
    t1_error_max_ms: float = 300.0,
    m0_error_max: float = 0.15,
) -> None:
    """Save a six-row overview as PNG/PDF and its unclipped metrics as JSON.

    output is a filename stem. SD is sample SD, not the standard error of
    the mean. M0 means, SD and errors are computed on draw magnitudes.
    Display limits apply equally to DPS and DAPS; values are not clipped
    when computing metrics. The first saved draw is shown without selection.
    """
    with np.load(results / "case.npz") as case, np.load(results / "cg-fit.npz") as fit:
        truth = {"t1": case["t1"].copy(), "m0": np.abs(case["m0"])}
        baseline = {"t1": fit["t1"].copy(), "m0": np.abs(fit["m0"])}
    data = {}
    for method in ("dps", "daps"):
        with np.load(results / f"{method}.npz") as samples:
            data[method] = {"t1": samples["t1"].astype(float), "m0": np.abs(samples["m0"]).astype(float)}
        for parameter in truth:
            values = data[method][parameter]
            if values.ndim != 3 or len(values) < 2 or values.shape[1:] != truth[parameter].shape:
                raise ValueError("Require at least two draws with shape (N,H,W) matching the reference")
    if not all(np.isfinite(v).all() for source in [truth, baseline, *data.values()] for v in source.values()):
        raise ValueError("Reference, baseline and draws must be finite")
    if not all(np.isfinite(v) and v > 0 for v in (t1_max, m0_max, t1_error_max_ms, m0_error_max)):
        raise ValueError("Display limits must be finite and positive")
    mask = truth["m0"] > mask_threshold
    if not mask.any():
        raise ValueError("Evaluation mask is empty")
    coords = np.argwhere(mask)
    low, high = np.maximum(coords.min(0) - 4, 0), np.minimum(coords.max(0) + 5, mask.shape)
    crop = np.s_[low[0] : high[0], low[1] : high[1]]
    t1_cmap = ListedColormap(np.loadtxt(t1_colors)) if t1_colors else plt.get_cmap("magma")
    palettes = {"t1": t1_cmap.with_extremes(bad="white"), "m0": plt.get_cmap("viridis").with_extremes(bad="white")}
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
        for start, parameter, label, maximum, error_max in [
            (0, "t1", r"$T_1$", t1_max, t1_error_max_ms),
            (4, "m0", r"$|M_0|$", m0_max, m0_error_max),
        ]:
            reference = truth[parameter]
            factor = 1000 if parameter == "t1" else 1

            def draw(
                row: int,
                column: int,
                field: np.ndarray,
                title: str = "",
                error: bool = False,
                maximum: float = maximum,
                error_max: float = error_max,
                parameter: str = parameter,
            ) -> plt.Axes:
                ax = fig.add_subplot(grid[row, column])
                ax.imshow(
                    np.where(mask, field, np.nan)[crop],
                    vmin=0,
                    vmax=error_max if error else maximum,
                    cmap=error_cmap if error else palettes[parameter],
                    interpolation="nearest",
                )
                ax.set_title(title, pad=2)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                return ax

            def annotate(
                ax: plt.Axes,
                image: np.ndarray,
                reference: np.ndarray = reference,
                factor: float = factor,
                parameter: str = parameter,
            ) -> dict[str, float]:
                values = scores(image, reference, mask)
                mae = f"{factor * values['mae']:.0f} ms" if parameter == "t1" else f"{values['mae']:.3f}"
                ax.set_xlabel(f"MAE {mae}   SSIM {values['ssim']:.3f}", labelpad=2)
                return values

            draw(start, 0, reference, "Reference " + label)
            metrics[parameter + "_cg_fit"] = annotate(draw(start, 1, baseline[parameter], "CG+fit " + label), baseline[parameter])
            for column, limit, cmap, units in [
                (2, maximum, palettes[parameter], label + (" (s)" if parameter == "t1" else " (a.u.)")),
                (3, error_max, error_cmap, "SD / error" + (" (ms)" if parameter == "t1" else " (a.u.)")),
            ]:
                host = fig.add_subplot(grid[start, column])
                host.set_axis_off()
                bar = fig.colorbar(
                    plt.cm.ScalarMappable(norm=Normalize(0, limit), cmap=cmap),
                    cax=host.inset_axes((0.08, 0.47, 0.84, 0.065)),
                    orientation="horizontal",
                    extend="max",
                )
                bar.set_ticks([0, limit / 2, limit])
                bar.set_label(units)
            ax = fig.add_subplot(grid[start + 1 : start + 3, 4])
            ax.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8)
            for index, (method, color, style) in enumerate([("dps", "#0072B2", "-"), ("daps", "#D55E00", "--")]):
                draws = data[method][parameter]
                mean, sd = draws.mean(0), draws.std(0, ddof=1)
                error = np.abs(mean - reference)
                titles = ["Sample", f"Mean (n={len(draws)})", "SD", "|Mean − ref|"]
                panels = [
                    draw(start + 1 + index, column, field, titles[column] if index == 0 else "", column >= 2)
                    for column, field in enumerate([draws[0], mean, factor * sd, factor * error])
                ]
                panels[0].set_ylabel(method.upper(), labelpad=7)
                sample_scores, mean_scores = annotate(panels[0], draws[0]), annotate(panels[1], mean)
                expected, observed = coverage(draws, reference, mask)
                ax.plot(expected, observed, color=color, linestyle=style, lw=1, label=method.upper())
                metrics[parameter + "_" + method] = {
                    "sample": sample_scores,
                    "mean": mean_scores,
                    "draws": len(draws),
                    "expected_coverage": expected.tolist(),
                    "observed_coverage": observed.tolist(),
                    "sd_above_display": float((factor * sd[mask] > error_max).mean()),
                    "error_above_display": float((factor * error[mask] > error_max).mean()),
                }
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xticks=[0, 0.5, 1],
                yticks=[0, 0.5, 1],
                ylabel="Observed",
                title=label + " interval coverage",
            )
            if start == 0:
                ax.legend(frameon=False, fontsize=7, loc="lower right")
            else:
                ax.set_xlabel("Expected")
        for extension in ("png", "pdf"):
            fig.savefig(output.with_suffix("." + extension), dpi=600)
        plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    output.with_suffix(".txt").write_text(
        "Joint saturation-recovery T1 and complex-M0 reconstruction. Reference |M0| defines the display and evaluation mask. "
        "First saved draw, ensemble mean, sample SD (ddof=1), and absolute mean error. "
        "Color limits affect display only; metrics use unclipped values. M0 summaries use magnitudes. "
        "Central intervals [s[k],s[N-1-k]] have nominal coverage (N-1-2k)/(N+1) under exchangeability. "
        "The endpoint (1,1) represents the unbounded interval; connecting lines do not estimate intermediate coverage. "
        "Coverage is descriptive: a reference-derived mask and a single case do not establish population calibration.\n"
    )


if __name__ == "__main__":
    CLI(main)
