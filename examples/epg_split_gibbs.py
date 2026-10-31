"""EPG joint T1/T2 and complex-M0 split Gibbs example.

Run a small synthetic EPG case with analytic Gaussian product priors::

    python examples/epg_split_gibbs.py /tmp/epg-sgs.npz

To use a checkpoint, also supply --checkpoint and --dataset_scales '[s1,s2,sm]'.
Scales convert seconds/amplitude into the training dataset units *before* the
checkpoint's kind normalization. EMA weights are selected explicitly by default.
Choose an image_size compatible with the checkpoint's downsampling levels.
Coupling SDs stay in physical units. The
analytic-prior example is an execution control, not a brain reconstruction.

An optional --acquisition NPZ supplies: parameters (J,2) in seconds, weights
(J,), flip_angles and rf_phases in radians, te/tr and inversion_time in seconds,
sensitivities (C,H,W), mask (L,H,W), data (B,L,C,1,H,W), support (H,W), noise_std
per real coordinate, initial_indices (B,H,W) and initial_m0 (B,H,W). Data use
mrpro's centered orthonormal FFT. Sequence: inversion followed by FISP. Other
sequences can construct SignalDictionary.from_model directly in Python.

Draws approximate a smoothed finite-dictionary posterior. SDE discretization,
terminal noise, learned scores and incomplete mixing can each introduce error.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jsonargparse import CLI
from mrpro.operators import FastFourierOp, SensitivityOp
from mrpro.operators.CartesianSamplingOp import CartesianMaskingOp
from mrpro.operators.models.EPG import EPGSequence, FispBlock, InversionBlock, Parameters
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.split_gibbs_sampler import (
    CartesianAcquisition,
    Coupling,
    DiffusionPriorDraw,
    SignalDictionary,
    SplitGibbsConfig,
    SplitGibbsSampler,
)

KINDS = (DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.M0_COMPLEX)


def gaussian_prior(observation: Tensor, coupling: Coupling, generator: torch.Generator | None) -> Tensor:
    """Exact conditional for explicit physical Gaussian product priors."""
    mean = observation.new_tensor([1.2, 0.1, 0.7, 0.0])[None, :, None, None]
    variance = observation.new_tensor([0.4, 0.04, 0.4, 0.4]).square()[None, :, None, None]
    r2 = observation.new_tensor([*coupling[:2], coupling[2], coupling[2]]).square()[None, :, None, None]
    conditional_variance = variance * r2 / (variance + r2)
    conditional_mean = (variance * observation + r2 * mean) / (variance + r2)
    noise = torch.randn(observation.shape, dtype=observation.dtype, device=observation.device, generator=generator)
    return conditional_mean + conditional_variance.sqrt() * noise


def synthetic_case(device: str, generator: torch.Generator, states: int = 64, size: int = 32) -> dict[str, Tensor]:
    """Small synthetic acquisition; sequence settings are example values only."""
    if size < 8:
        raise ValueError("Synthetic image size must be at least 8")
    t1 = torch.tensor([0.6, 0.9, 1.2, 1.5, 1.8], device=device)
    t2 = torch.tensor([0.04, 0.08, 0.12, 0.16], device=device)
    parameters = torch.cartesian_prod(t1, t2)
    pulse = torch.linspace(0, 1, 48, device=device)
    flip = torch.deg2rad(15 + 45 * torch.sin(torch.pi * pulse).square())
    phase = (pulse * 12).square()
    te = torch.full_like(pulse, 0.004)
    tr = 0.012 + 0.006 * torch.cos(2 * torch.pi * pulse).square()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size, device=device), torch.linspace(-1, 1, size, device=device), indexing="ij")
    support = xx.square() + yy.square() < 0.8
    coil_phase = torch.stack((xx, -yy))
    coils = torch.polar(torch.full_like(coil_phase, 2**-0.5), coil_phase)
    mask = torch.zeros(48, size, size, device=device, dtype=torch.bool)
    mask[:, size // 2 - 2 : size // 2 + 2] = True
    for frame in range(48):
        mask[frame, frame % 4 :: 4] = True
    indices = torch.where(xx > 0, 10, 5).long()[None]
    m0 = torch.polar((0.7 + 0.2 * yy) * support, 0.5 * xx)[None]
    case = {
        "parameters": parameters,
        "weights": torch.full((20,), 0.3 * 0.04, device=device),
        "flip_angles": flip,
        "rf_phases": phase,
        "te": te,
        "tr": tr,
        "inversion_time": torch.tensor(0.02, device=device),
        "sensitivities": coils,
        "mask": mask,
        "support": support,
        "noise_std": torch.tensor(0.03, device=device),
        "initial_indices": torch.zeros_like(indices),
        "initial_m0": torch.zeros_like(m0),
        "data": torch.zeros(1, 48, 2, 1, size, size, dtype=torch.complex64, device=device),
    }
    dictionary = build_dictionary(case, states=states)
    operator = build_operator(case)
    signal = dictionary.fingerprints[indices].moveaxis(-1, 1) * m0[:, None]
    (data,) = operator(signal[:, :, None, None])
    noise = torch.randn((*data.shape, 2), device=device, generator=generator)
    case["data"] = (data + case["noise_std"] * torch.complex(noise[..., 0], noise[..., 1])) * mask[None, :, None, None]
    case["truth"] = torch.cat((parameters[indices].moveaxis(-1, 1), m0.real[:, None], m0.imag[:, None]), 1) * support
    return case


def build_dictionary(case: dict[str, Tensor], states: int) -> SignalDictionary:
    """Build the supplied inversion-prepared FISP dictionary with mrpro EPG."""
    sequence = EPGSequence(
        [InversionBlock(case["inversion_time"]), FispBlock(case["flip_angles"], case["rf_phases"], case["te"], case["tr"])]
    )

    def signal_model(m0: Tensor, t1: Tensor, t2: Tensor) -> tuple[Tensor]:
        _, acquisitions = sequence(Parameters(m0=m0, t1=t1, t2=t2), states)
        return (torch.stack(acquisitions, 0),)

    return SignalDictionary.from_model(signal_model, case["parameters"], case["weights"])


def build_operator(case: dict[str, Tensor]) -> CartesianAcquisition:
    """Build P U S from explicit mrpro operators; the sampler supplies none."""
    return CartesianAcquisition(
        CartesianMaskingOp(case["mask"][None, :, None, None]),
        FastFourierOp(dim=(-2, -1)),
        SensitivityOp(case["sensitivities"][:, None]),
    )


def load_acquisition(path: Path, device: str) -> dict[str, Tensor]:
    """Load acquisition arrays, including from an earlier example output."""
    fields = (
        "parameters",
        "weights",
        "flip_angles",
        "rf_phases",
        "te",
        "tr",
        "inversion_time",
        "sensitivities",
        "mask",
        "data",
        "support",
        "noise_std",
        "initial_indices",
        "initial_m0",
    )
    with np.load(path, allow_pickle=False) as stored:
        case = {name: torch.as_tensor(stored[name], device=device) for name in fields}
        if "truth" in stored:
            case["truth"] = torch.as_tensor(stored["truth"], device=device)
    if case["data"].ndim == 5:
        case["data"] = case["data"].unsqueeze(3)
    return case


def main(
    output: Path,
    acquisition: Path | None = None,
    checkpoint: Path | None = None,
    checkpoint_weights: Literal["ema", "raw"] = "ema",
    compile_model: bool = False,
    dataset_scales: Coupling | None = None,
    coupling: Coupling = (0.2, 0.03, 0.15),
    draws: int = 8,
    burn_in: int = 64,
    thinning: int = 1,
    sde_steps: int = 128,
    sigma_min: float | None = None,
    epg_states: int = 64,
    dictionary_chunk: int = 256,
    image_size: int = 32,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    """Save physical draws and complete acquisition/settings for reproducibility."""
    if output.exists():
        raise FileExistsError(output)
    if checkpoint is not None and dataset_scales is None:
        raise ValueError("Specify dataset_scales explicitly when using a checkpoint")
    generator = torch.Generator(device=device).manual_seed(seed)
    case = synthetic_case(device, generator, epg_states, image_size) if acquisition is None else load_acquisition(acquisition, device)
    dictionary = build_dictionary(case, epg_states)
    model = None
    terminal_sigma = np.nan
    checkpoint_info = {}
    physical_scales = np.empty(0)
    if checkpoint is not None:
        model = (
            DiffusionModel.from_checkpoint(checkpoint, map_location=device, compile_model=compile_model, weights=checkpoint_weights)
            .to(device)
            .eval()
        )
        assert dataset_scales is not None
        prior = DiffusionPriorDraw(model, KINDS, dataset_scales, steps=sde_steps, sigma_min=sigma_min)
        checkpoint_info = model.checkpoint_info
        physical_scales = prior.physical_scales(dictionary.parameters).cpu().numpy()
        terminal_sigma = prior.sigma_min
    cfg = SplitGibbsConfig(
        coupling,
        num_samples=draws,
        burn_in=burn_in,
        thinning=thinning,
        prior_steps=sde_steps,
        prior_sigma_min=sigma_min,
        dictionary_chunk=dictionary_chunk,
    )
    sampler = SplitGibbsSampler(model, cfg)
    sampling_rng_state = generator.get_state().cpu().numpy()
    (samples,) = sampler.sample(
        build_operator(case),
        case["data"],
        noise_std=float(case["noise_std"]),
        template=case["initial_m0"],
        dictionary=dictionary,
        support=case["support"],
        kind=KINDS,
        dataset_scales=dataset_scales,
        initial_indices=case["initial_indices"],
        initial_m0=case["initial_m0"],
        prior_draw=gaussian_prior if model is None else None,
        generator=generator,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        allow_pickle=False,
        **{name: value.cpu().numpy() for name, value in case.items()},
        samples=samples.cpu().numpy(),
        fingerprints=dictionary.fingerprints.cpu().numpy(),
        coupling=np.asarray(coupling),
        dataset_scales=np.asarray(dataset_scales or ()),
        physical_to_model_scales=physical_scales,
        prior="diffusion_reverse_sde" if checkpoint else "analytic_gaussian",
        checkpoint=str(checkpoint or ""),
        checkpoint_info=json.dumps(checkpoint_info),
        compile_model=compile_model,
        sde_steps=sde_steps,
        sigma_min=terminal_sigma,
        epg_states=epg_states,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed,
        device=device,
        sampling_rng_state=sampling_rng_state,
        channels=np.asarray(["T1_seconds", "T2_seconds", "M0_real", "M0_imag"]),
        noise_convention="per_real_coordinate",
        fft_convention="centered_ortho",
    )
    print(f"Saved {draws} states per chain to {output}")


if __name__ == "__main__":
    CLI(main)
