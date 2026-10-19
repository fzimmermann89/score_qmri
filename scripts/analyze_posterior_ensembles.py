"""Compare ensemble uncertainty with known reconstruction errors without smoothing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from jsonargparse import ArgumentParser
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def posterior_maps(samples: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Return matched complex and magnitude estimators and sample standard deviations.

    The leading axis indexes independent draws; remaining axes match truth.
    Magnitude mean is E[|x|], which differs from |E[x]|. Complex spread is
    sqrt(Var(real)+Var(imag)). Standard deviations use n-1 denominators and
    describe draw spread, not the standard error of the estimated mean.
    """
    if samples.shape[0] < 2 or samples.shape[1:] != truth.shape:
        raise ValueError("Require >=2 draws with the same spatial shape as truth")
    if not np.isfinite(samples).all() or not np.isfinite(truth).all():
        raise ValueError("Samples and reference must be finite")
    mean = samples.mean(axis=0)
    magnitudes = np.abs(samples)
    magnitude_mean = magnitudes.mean(axis=0)
    return {
        "mean_complex": mean,
        "mean_magnitude": magnitude_mean,
        "error_magnitude": np.abs(magnitude_mean - np.abs(truth)),
        "std_magnitude": magnitudes.std(axis=0, ddof=1),
        "error_complex": np.abs(mean - truth),
        "std_complex": np.sqrt(samples.real.var(axis=0, ddof=1) + samples.imag.var(axis=0, ddof=1)),
    }


def diagnostic(error: np.ndarray, std: np.ndarray, roi: np.ndarray) -> dict[str, object]:
    """Spatial diagnostics on one fixed image; pixels are not independent trials."""
    e, s = error[roi], std[roi]
    if e.size < 8:
        raise ValueError("Foreground needs at least eight pixels for uncertainty bins")
    bins = []
    for indices in np.array_split(np.argsort(s), 8):
        bins.append(
            {"count": len(indices), "rms_std": float(np.sqrt(np.mean(s[indices] ** 2))), "rmse": float(np.sqrt(np.mean(e[indices] ** 2)))}
        )
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "rms_std": float(np.sqrt(np.mean(s**2))),
        "rmse_over_rms_std": float(np.sqrt(np.mean(e**2) / np.mean(s**2))),
        "spearman_error_vs_std": float(spearmanr(e, s).statistic),
        "uncertainty_bins": bins,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    reports = []
    truth = None
    measurement = None
    mask = None
    for run in args.runs:
        report = json.loads((run / "report.json").read_text())
        case = np.load(run / "case.npz")
        if truth is None:
            truth, measurement, mask = case["truth"].squeeze(), case["measurement"].squeeze(), case["mask"].squeeze()
        elif (
            not np.array_equal(truth, case["truth"].squeeze())
            or not np.array_equal(measurement, case["measurement"].squeeze())
            or not np.array_equal(mask, case["mask"].squeeze())
        ):
            raise ValueError("Comparison requires identical reference and measurement")
        paths = sorted(run.glob("draw-*.npz"))
        if len(paths) != report["arguments"]["draws"]:
            raise ValueError(f"Incomplete ensemble: {run}")
        samples = np.stack([np.load(p)["sample"].squeeze() for p in paths])
        maps = posterior_maps(samples, truth)
        spectral_maps = posterior_maps(np.fft.fft2(samples, norm="ortho"), np.fft.fft2(truth, norm="ortho"))
        roi = np.abs(truth) > 0.05 * np.abs(truth).max()
        method_label = f"DPS g={report['configuration']['guidance_scale']:g}" if report["arguments"]["sampler"] == "dps" else "DAPS score"
        label = f"{method_label} · n={len(samples)}"
        entry = {
            "run": str(run),
            "label": label,
            "draw_count": len(samples),
            "configuration": report["configuration"],
            "draw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "magnitude": diagnostic(maps["error_magnitude"], maps["std_magnitude"], roi),
            "complex": diagnostic(maps["error_complex"], maps["std_complex"], roi),
            "magnitude_mean_psnr": float(10 * np.log10(np.abs(truth).max() ** 2 / np.mean(maps["error_magnitude"] ** 2))),
            "complex_mean_nmse": float(np.sum(maps["error_complex"] ** 2) / np.sum(np.abs(truth) ** 2)),
            "split_half_magnitude_mean_rmse": float(
                np.sqrt(np.mean((np.abs(samples[: len(samples) // 2]).mean(0) - np.abs(samples[len(samples) // 2 :]).mean(0)) ** 2))
            ),
        }
        entry["prefix_diagnostics"] = []
        for count in sorted({2, min(4, len(samples)), len(samples)}):
            prefix = posterior_maps(samples[:count], truth)
            entry["prefix_diagnostics"].append(
                {
                    "draws": count,
                    "magnitude": diagnostic(prefix["error_magnitude"], prefix["std_magnitude"], roi),
                    "complex": diagnostic(prefix["error_complex"], prefix["std_complex"], roi),
                }
            )
        entry["kspace"] = {}
        for name, selected in (("observed", mask.astype(bool)), ("unobserved", ~mask.astype(bool))):
            entry["kspace"][name] = diagnostic(spectral_maps["error_complex"], spectral_maps["std_complex"], selected)
        reports.append(entry)
        entries.append((label, maps, samples))
        np.savez_compressed(args.output / f"maps-{len(entries) - 1:02d}-{report['arguments']['sampler']}.npz", **maps, roi=roi)
    assert truth is not None and measurement is not None and mask is not None
    zero_filled = np.fft.ifft2(measurement * mask, norm="ortho")
    plt.imsave(args.output / "reference.png", np.abs(truth), cmap="gray", vmin=0, vmax=float(np.abs(truth).max()))
    plt.imsave(args.output / "zero-filled.png", np.abs(zero_filled), cmap="gray", vmin=0, vmax=float(np.abs(truth).max()))
    display_limits = {}
    focused_limit = 0.1 * float(np.abs(truth).max())
    saturated_fractions = {}
    for domain in ("magnitude", "complex"):
        vmax = max(float(maps[f"{key}_{domain}"].max()) for _, maps, _ in entries for key in ("error", "std"))
        display_limits[domain] = vmax
        fig, axes = plt.subplots(len(entries), 5, figsize=(15, 3.4 * len(entries)), squeeze=False, layout="constrained")
        for row, (label, maps, _) in enumerate(entries):
            reconstruction = maps["mean_magnitude"] if domain == "magnitude" else np.abs(maps["mean_complex"])
            arrays = [np.abs(truth), np.abs(zero_filled), reconstruction, maps[f"error_{domain}"], maps[f"std_{domain}"]]
            titles = [
                "Reference magnitude",
                "Zero-filled magnitude",
                "Mean magnitude E[|x|]" if domain == "magnitude" else "Magnitude of complex mean |E[x]|",
                "Absolute magnitude error" if domain == "magnitude" else "Complex error |E[x] − truth|",
                "Magnitude SD across draws" if domain == "magnitude" else "Complex SD √(Var Re + Var Im)",
            ]
            for column, (array, title) in enumerate(zip(arrays, titles, strict=True)):
                ax = axes[row, column]
                if column >= 2:
                    stem = {2: "mean", 3: "error", 4: "std"}[column]
                    plt.imsave(
                        args.output / f"{domain}-{row}-{stem}.png",
                        array,
                        cmap="gray" if column == 2 else "magma",
                        vmin=0,
                        vmax=float(np.abs(truth).max()) if column == 2 else vmax,
                    )
                    if column >= 3:
                        plt.imsave(args.output / f"{domain}-{row}-{stem}-focused.png", array, cmap="magma", vmin=0, vmax=focused_limit)
                        saturated_fractions[f"{domain}-{row}-{stem}"] = float(np.mean(array > focused_limit))
                image = ax.imshow(
                    array,
                    cmap="gray" if column < 3 else "magma",
                    vmin=0,
                    vmax=float(np.abs(truth).max()) if column < 3 else vmax,
                    interpolation="nearest",
                )
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(label, fontsize=11)
                if column >= 3:
                    fig.colorbar(image, ax=ax, fraction=0.046, label="Normalized image units")
        fig.suptitle(f"{domain.capitalize()} uncertainty versus error · shared full-range error/SD scale · no smoothing", fontsize=12)
        fig.savefig(args.output / f"{domain}-maps.png", dpi=160)
        fig.savefig(args.output / f"{domain}-maps.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, domain in zip(axes, ("magnitude", "complex"), strict=True):
        maximum = 0.0
        for report in reports:
            bins = report[domain]["uncertainty_bins"]
            x, y = [b["rms_std"] for b in bins], [b["rmse"] for b in bins]
            ax.plot(x, y, "o-", label=report["label"])
            maximum = max(maximum, max(x), max(y))
        ax.plot([0, maximum], [0, maximum], "k--", linewidth=1, label="Equal RMS error and SD")
        ax.set(
            xlim=(0, maximum * 1.05),
            ylim=(0, maximum * 1.05),
            xlabel="RMS ensemble SD",
            ylabel="RMSE of ensemble mean",
            title=f"{domain.capitalize()} · 8 equal-count uncertainty bins",
        )
        ax.legend(fontsize=8)
    fig.suptitle("Spatial error–spread diagnostic · reference magnitude > 5% of peak · one fixed case")
    fig.savefig(args.output / "error-spread.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(entries), 4, figsize=(12, 3.1 * len(entries)), squeeze=False, layout="constrained")
    for row, (label, _, samples) in enumerate(entries):
        for col, draw in enumerate(samples[:4]):
            axes[row, col].imshow(np.abs(draw), cmap="gray", vmin=0, vmax=float(np.abs(truth).max()), interpolation="nearest")
            axes[row, col].set_title(f"Draw {col + 1}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(label)
    fig.suptitle("First four draws · identical magnitude display scale")
    fig.savefig(args.output / "samples.png", dpi=160)
    plt.close(fig)
    result = {
        "runs": reports,
        "roi": "reference magnitude > 5% of reference peak; diagnostics only; maps show all pixels",
        "semantics": "sample SD (n-1), not standard error; magnitude mean=E[abs(x)]; complex error=abs(E[x]-truth)",
        "limitations": "same fixed noisy measurement, training-store overlap, correlated pixels, finite ensemble; spatial plots are not calibration or coverage tests",
        "display": "no smoothing/interpolation; shared magnitude scale; error/SD share full-range maximum across both samplers per domain",
        "error_std_display_max": display_limits,
        "focused_display": {
            "maximum": focused_limit,
            "rule": "10% of reference magnitude peak; shared across samplers and domains",
            "fraction_above_max": saturated_fractions,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
