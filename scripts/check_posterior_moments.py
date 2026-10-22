"""Measure sampler moments against an analytic complex Gaussian posterior."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from jsonargparse import ArgumentParser

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling import DAPSConfig, DAPSSampler, DPSConfig, DPSSampler, FullScoreDAPSConfig, FullScoreDAPSSampler


class GaussianOracle:
    """Exact denoiser/score for independent N(0, 0.04) real coordinates."""

    device = torch.device("cpu")
    sigma_min = 0.001
    sigma_max = 20.0
    sigma_data = 0.2

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        sigma = sigma.expand(len(x))
        return x * 0.04 / (0.04 + sigma[:, None, None, None].square())

    def score(self, x: torch.Tensor, sigma: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        return -x / (0.04 + sigma[:, None, None, None].square())


class Mask:
    """Observe one complex coordinate; leave the second in the null space."""

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x * x.real.new_tensor([1.0, 0.0]),)

    @property
    def H(self) -> Mask:
        return self


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--sampler", choices=["dps", "daps", "full-score-daps"], required=True)
    args = parser.parse_args()
    if args.draws < 2:
        raise ValueError("At least two draws are needed for a variance estimate")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    model = cast(DiffusionModel, GaussianOracle())
    cfg: DPSConfig | DAPSConfig | FullScoreDAPSConfig
    sampler: DPSSampler | DAPSSampler | FullScoreDAPSSampler
    if args.sampler == "dps":
        cfg = DPSConfig(tau=0.01)
        sampler = DPSSampler(model, cfg)
    elif args.sampler == "daps":
        cfg = DAPSConfig(tau=0.01)
        sampler = DAPSSampler(model, cfg)
    else:
        cfg = FullScoreDAPSConfig(tau=0.01)
        sampler = FullScoreDAPSSampler(model, cfg)
    y = torch.zeros(args.draws, 1, 1, 1, 2, dtype=torch.complex128)
    y[..., 0] = y.new_tensor(0.15 + 0.05j)
    sample = sampler.sample(Mask(), y, DatasetKind.COMPLEX_T1W_COIL_COMBINED, generator=torch.Generator().manual_seed(args.seed))[
        :, 0, 0, 0
    ]
    prior_variance = 0.04
    if isinstance(sampler, FullScoreDAPSSampler):
        prior_variance += sampler.prior_sigma**2
    variance = torch.tensor([prior_variance * 0.0001 / (prior_variance + 0.0001), prior_variance], dtype=torch.float64)
    report = {
        "scope": "analytic Gaussian posterior control, no learned prior or dataset",
        "draws": args.draws,
        "seed": args.seed,
        "configuration": asdict(cfg),
        "target_mean_real": [0.15 * prior_variance / (prior_variance + 0.0001), 0.0],
        "target_mean_imag": [0.05 * prior_variance / (prior_variance + 0.0001), 0.0],
        "target_variance": variance.tolist(),
        "mean_real": sample.real.mean(0).tolist(),
        "mean_imag": sample.imag.mean(0).tolist(),
        "variance_real": sample.real.var(0).tolist(),
        "variance_imag": sample.imag.var(0).tolist(),
        "variance_ratio_real": (sample.real.var(0) / variance).tolist(),
        "variance_ratio_imag": (sample.imag.var(0) / variance).tolist(),
        "relative_mc_standard_error_of_variance": (2 / (args.draws - 1)) ** 0.5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
