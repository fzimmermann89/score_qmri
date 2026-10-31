"""Cartesian missing-data Gibbs sampling for EPG dictionaries.

Physical state: (B, 4, H, W), channels T1 [s], T2 [s], Re(M0), Im(M0).
All standard deviations are per real coordinate, including complex noise.
Exact prior conditionals preserve the finite weighted dictionary target with
Gaussian-smoothed product priors. Learned finite-step SDE updates approximate
those conditionals; they do not establish an exact stationary distribution.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch
from mrpro.operators import FastFourierOp
from mrpro.operators.models.EPG import EPGSequence, Parameters
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel

Coupling = tuple[float, float, float]
PriorDraw = Callable[[Tensor, Coupling, torch.Generator], Tensor]


def _positive(values: tuple[float, ...]) -> None:
    if not all(isfinite(x) and x > 0 for x in values):
        raise ValueError("Scales and standard deviations must be finite and positive")


def _noise(x: Tensor, generator: torch.Generator) -> Tensor:
    # torch.randn(complex) has variance 1/2 per real component.
    if x.is_complex():
        return torch.complex(_noise(x.real, generator), _noise(x.real, generator))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


@dataclass(frozen=True)
class EPGDictionary:
    """Physical (T1,T2) pairs (J,2), complex fingerprints (J,L), cell weights (J,).

    Fingerprints retain their unit-M0 amplitude; normalizing each atom would
    change the amplitude prior and hence the target. Weights define mass on
    the finite dictionary (cell areas for continuous-prior quadrature).
    """

    parameters: Tensor
    fingerprints: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        if self.parameters.ndim != 2 or self.parameters.shape[1] != 2 or self.parameters.shape[0] == 0:
            raise ValueError("parameters must have shape (J,2) with J > 0")
        j = self.parameters.shape[0]
        if self.fingerprints.ndim != 2 or self.fingerprints.shape[0] != j or self.fingerprints.shape[1] == 0:
            raise ValueError("fingerprints must have shape (J,L) with L > 0")
        if self.weights.shape != (j,) or not self.fingerprints.is_complex():
            raise ValueError("weights must have shape (J,) and fingerprints must be complex")
        if not self.parameters.is_floating_point() or self.weights.dtype != self.parameters.dtype:
            raise ValueError("parameters and weights must have the same real floating dtype")
        if self.fingerprints.real.dtype != self.parameters.dtype:
            raise ValueError("fingerprints and parameters must have matching precision")
        tensors = (self.parameters, self.fingerprints, self.weights)
        if any(t.device != self.parameters.device or not torch.isfinite(t).all() for t in tensors):
            raise ValueError("Dictionary tensors must be finite and on the same device")
        if (self.parameters <= 0).any() or (self.weights <= 0).any():
            raise ValueError("Relaxation times and weights must be positive")

    @classmethod
    @torch.no_grad()
    def from_sequence(cls, sequence: EPGSequence, parameters: Tensor, weights: Tensor, *, states: int = 64) -> "EPGDictionary":
        """Simulate a supplied mrpro sequence at unit M0, with times in seconds.

        Parameters
        ----------
        sequence
            EPG sequence on the same device as the parameter grid.
        parameters
            Positive physical T1/T2 pairs, shape (J,2).
        weights
            Positive mass or quadrature weights, shape (J,).
        states
            Number of EPG configuration states; verify truncation for the sequence.
        """
        if states < 2 or parameters.ndim != 2 or parameters.shape[1] != 2 or (parameters <= 0).any():
            raise ValueError("Need at least two EPG states and positive (J,2) parameters")
        tissue = Parameters(m0=torch.ones_like(parameters[:, 0]), t1=parameters[:, 0], t2=parameters[:, 1])
        _, signals = sequence(tissue, states)
        if not signals:
            raise ValueError("EPG sequence has no acquisitions")
        return cls(parameters, torch.stack(signals, dim=-1), weights)


class DiffusionPriorDraw:
    """Independent kind-conditioned reverse-SDE draws in physical coordinates."""

    def __init__(self, model: DiffusionModel, dataset_scales: Coupling, *, steps: int = 128, sigma_min: float | None = None) -> None:
        """
        Parameters
        ----------
        model
            Frozen evaluation-mode denoiser; no clipping or guidance is applied.
        dataset_scales
            Multipliers from physical T1, T2, M0 to each dataset's training input
            units, before model.normalize. Explicitly specify M0 scaling for the
            acquisition; there is no universal amplitude scale.
        steps
            Euler--Maruyama steps on a geometric noise schedule.
        sigma_min
            Terminal noise in model coordinates, at least model.sigma_min.
            The returned endpoint remains noisy; it is not replaced by a mean.
            Denoiser evaluations use the model's floating dtype; integration
            and returned maps retain the observation's dtype.
        """
        _positive(dataset_scales)
        self.model = model
        self.dataset_scales = dataset_scales
        self.steps = steps
        self.sigma_min = model.sigma_min if sigma_min is None else sigma_min
        _positive((self.sigma_min,))
        if steps < 1 or self.sigma_min < model.sigma_min or model.training:
            raise ValueError("Need an evaluation model, steps >= 1 and endpoint >= model.sigma_min")

    def physical_scales(self, reference: Tensor) -> Tensor:
        """Return physical-to-model multipliers for T1, T2 and M0."""
        kinds = torch.tensor([DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.M0_COMPLEX], device=reference.device)
        unit = torch.ones(3, 2, 1, 1, device=reference.device, dtype=reference.dtype)
        # Derive model scales through its public normalization API, not defaults.
        scales = self.model.normalize(unit, kinds)[:, 0, 0, 0] * reference.new_tensor(self.dataset_scales)
        if not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Physical-to-model scales must be finite and positive")
        return scales

    @torch.no_grad()
    def __call__(self, observation: Tensor, coupling: Coupling, generator: torch.Generator) -> Tensor:
        """Draw auxiliary maps; coupling SDs are in physical map units."""
        _positive(coupling)
        if observation.ndim != 4 or observation.shape[1] != 4 or not observation.is_floating_point():
            raise ValueError("observation must be real (B,4,H,W)")
        parameter = next(self.model.parameters(), observation)
        if parameter.device != observation.device:
            raise ValueError("Model and observation must be on the same device")
        batch = observation.shape[0]
        kinds = torch.tensor([DatasetKind.T1MAP, DatasetKind.T2MAP, DatasetKind.M0_COMPLEX], device=observation.device)
        scales = self.physical_scales(observation)
        start = scales * observation.new_tensor(coupling)
        if not torch.isfinite(start).all() or (start <= self.sigma_min).any() or (start > self.model.sigma_max).any():
            raise ValueError("Scaled coupling levels must lie above endpoint and within model.sigma_max")
        zero = torch.zeros_like(observation[:, :1])
        x = torch.cat((torch.cat((observation[:, :1], zero), 1), torch.cat((observation[:, 1:2], zero), 1), observation[:, 2:]), 0)
        scale = scales.repeat_interleave(batch)[:, None, None, None]
        x = x * scale
        kinds = kinds.repeat_interleave(batch)
        initial_levels = start.repeat_interleave(batch)
        levels = initial_levels
        for step in range(self.steps):
            next_levels = initial_levels * (self.sigma_min / initial_levels) ** ((step + 1) / self.steps)
            delta = (levels.square() - next_levels.square())[:, None, None, None]
            denoised = self.model.denoise(x.to(parameter.dtype), levels.to(parameter.dtype), kinds).to(x.dtype)
            noise = _noise(x, generator)
            noise[: 2 * batch, 1] = 0
            x = x + delta * (denoised - x) / levels[:, None, None, None].square() + delta.sqrt() * noise
            x[: 2 * batch, 1] = 0
            levels = next_levels
        x = x / scale
        return torch.cat((x[:batch, :1], x[batch : 2 * batch, :1], x[2 * batch :]), 1)


class EPGSplitGibbs:
    """Fixed-coupling sampler with centered, unitary Cartesian encoding.

    The prior callable returns full-grid auxiliary maps, including background.
    It may be an exact analytic conditional or DiffusionPriorDraw. Physical
    maps alone are zero outside support. No baseline sampler code is used.
    """

    def __init__(
        self,
        dictionary: EPGDictionary,
        sensitivities: Tensor,
        mask: Tensor,
        measurement: Tensor,
        support: Tensor,
        *,
        noise_std: float,
        coupling: Coupling,
        prior_draw: PriorDraw,
        dictionary_chunk: int = 256,
    ) -> None:
        """
        Parameters
        ----------
        dictionary
            EPG fingerprints in physical amplitude units.
        sensitivities
            Known whitened complex coil maps (C,H,W).
        mask
            Boolean Cartesian sampling mask (L,H,W), centered FFT convention.
        measurement
            Complex full-grid data (B,L,C,H,W). Unsampled entries are ignored.
        support
            Boolean fixed object mask (H,W).
        noise_std
            Acquisition noise SD per real coordinate in measurement units.
        coupling
            Physical per-real-coordinate SDs for T1, T2, complex M0. The
            draft's circular complex variance is twice coupling[2] squared.
        prior_draw
            Conditional draw(observation, coupling, generator) -> auxiliary maps.
        dictionary_chunk
            Number of dictionary atoms evaluated together to bound memory.
        """
        _positive((noise_std, *coupling))
        if sensitivities.ndim != 3 or not sensitivities.is_complex():
            raise ValueError("sensitivities must be complex (C,H,W)")
        c, h, w = sensitivities.shape
        frames = dictionary.fingerprints.shape[1]
        if mask.shape != (frames, h, w) or mask.dtype != torch.bool or support.shape != (h, w) or support.dtype != torch.bool:
            raise ValueError("Need boolean mask (L,H,W) and support (H,W)")
        if measurement.ndim != 5 or measurement.shape[1:] != (frames, c, h, w) or measurement.shape[0] == 0:
            raise ValueError("measurement must have shape (B,L,C,H,W)")
        if measurement.dtype != dictionary.fingerprints.dtype or sensitivities.dtype != measurement.dtype:
            raise ValueError("Data, coil maps and dictionary must have matching complex dtype")
        if any(t.device != measurement.device for t in (sensitivities, mask, support, dictionary.parameters)):
            raise ValueError("All tensors must be on the same device")
        if not torch.isfinite(sensitivities).all() or not torch.isfinite(measurement[:, mask[:, None].expand(-1, c, -1, -1)]).all():
            raise ValueError("Coil maps and measured samples must be finite")
        if dictionary_chunk < 1 or not support.any():
            raise ValueError("Need dictionary_chunk >= 1 and nonempty support")
        self.dictionary, self.sensitivities, self.mask = dictionary, sensitivities, mask
        self.measurement, self.support = measurement, support
        self.noise_std, self.coupling, self.prior_draw = noise_std, coupling, prior_draw
        self.dictionary_chunk = dictionary_chunk
        self.coil_power = sensitivities.abs().square().sum(0)
        self.fft = FastFourierOp(dim=(-2, -1))

    def maps(self, indices: Tensor, amplitude: Tensor) -> Tensor:
        """Decode physical maps, zero outside support, as (B,4,H,W)."""
        relaxation = self.dictionary.parameters[indices].moveaxis(-1, 1)
        return torch.cat((relaxation, amplitude.real[:, None], amplitude.imag[:, None]), 1) * self.support

    def forward(self, indices: Tensor, amplitude: Tensor) -> tuple[Tensor]:
        """Encode dictionary maps to full-grid masked data (B,L,C,H,W)."""
        signal = self.dictionary.fingerprints[indices].moveaxis(-1, 1) * (amplitude * self.support)[:, None]
        (kspace,) = self.fft(signal[:, :, None] * self.sensitivities)
        return (kspace * self.mask[None, :, None],)

    def impute(self, indices: Tensor, amplitude: Tensor, generator: torch.Generator) -> Tensor:
        """Draw missing k-space and copy observed entries without adding noise."""
        signal = self.dictionary.fingerprints[indices].moveaxis(-1, 1) * (amplitude * self.support)[:, None]
        (predicted,) = self.fft(signal[:, :, None] * self.sensitivities)
        return torch.where(self.mask[None, :, None], self.measurement, predicted + self.noise_std * _noise(predicted, generator))

    def collapsed_parameters(self, coil_images: Tensor, auxiliary: Tensor, start: int, stop: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, complex mean and real-component variance (B,H,W,Jchunk)."""
        combined = (coil_images * self.sensitivities.conj()).sum(2).moveaxis(1, -1)
        return self._collapse(combined, auxiliary, start, stop)

    def _collapse(self, combined: Tensor, auxiliary: Tensor, start: int, stop: int) -> tuple[Tensor, Tensor, Tensor]:
        atoms = self.dictionary.fingerprints[start:stop]
        # Working in real precision: CN variance = 2 * per-real variance.
        precision = self.coil_power[..., None] * atoms.abs().square().sum(-1) / self.noise_std**2 + 1 / self.coupling[2] ** 2
        amplitude = torch.complex(auxiliary[:, 2], auxiliary[:, 3])
        rhs = combined @ atoms.conj().T / self.noise_std**2 + amplitude[..., None] / self.coupling[2] ** 2
        mean = rhs / precision
        penalty = (
            (self.dictionary.parameters[start:stop] - auxiliary[:, :2].moveaxis(1, -1)[..., None, :])
            / auxiliary.new_tensor(self.coupling[:2])
        ).square().sum(-1) / 2
        logits = self.dictionary.weights[start:stop].log() + rhs.abs().square() / (2 * precision) - precision.log() - penalty
        return logits, mean, precision.reciprocal().expand_as(logits)

    def physics_draw(self, completed: Tensor, auxiliary: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """Joint dictionary/amplitude conditional using streaming Gumbel-max."""
        (images,) = self.fft.H(completed)
        combined = (images * self.sensitivities.conj()).sum(2).moveaxis(1, -1)
        shape = auxiliary[:, 0].shape
        best = auxiliary.new_full(shape, -torch.inf)
        indices = torch.zeros(shape, device=auxiliary.device, dtype=torch.long)
        mean = torch.zeros(shape, device=auxiliary.device, dtype=completed.dtype)
        variance = torch.zeros_like(best)
        for start in range(0, len(self.dictionary.weights), self.dictionary_chunk):
            stop = min(start + self.dictionary_chunk, len(self.dictionary.weights))
            logits, means, variances = self._collapse(combined, auxiliary, start, stop)
            uniform = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=generator).clamp_min(
                torch.finfo(logits.dtype).tiny
            )
            value, local = (logits - (-uniform.log()).log()).max(-1)
            take = value > best
            mean = torch.where(take, means.gather(-1, local[..., None]).squeeze(-1), mean)
            variance = torch.where(take, variances.gather(-1, local[..., None]).squeeze(-1), variance)
            indices = torch.where(take, start + local, indices)
            best = torch.maximum(best, value)
        amplitude = (mean + variance.sqrt() * _noise(mean, generator)) * self.support
        return indices.masked_fill(~self.support, 0), amplitude

    @torch.no_grad()
    def sample(
        self, initial_indices: Tensor, initial_m0: Tensor, *, num_samples: int, burn_in: int, generator: torch.Generator, thinning: int = 1
    ) -> Tensor:
        """Return (draws,B,4,H,W) physical maps at fixed coupling widths.

        Parameters
        ----------
        initial_indices
            Dictionary indices (B,H,W). Background indices must still be valid.
        initial_m0
            Complex amplitudes (B,H,W), same dtype and device as data.
        num_samples
            Number of retained states per chain; retained states are correlated.
        burn_in
            Number of discarded sweeps at the fixed target.
        generator
            Explicit device-matched random generator for every stochastic update.
        thinning
            Sweeps between retained states. This does not guarantee independence.
        """
        expected = (self.measurement.shape[0], *self.support.shape)
        if initial_indices.shape != expected or initial_indices.dtype != torch.long or initial_m0.shape != expected:
            raise ValueError("Initial state must have shape (B,H,W) and long indices")
        if (
            initial_indices.device != self.measurement.device
            or initial_m0.device != self.measurement.device
            or initial_m0.dtype != self.measurement.dtype
        ):
            raise ValueError("Initial state must match measurement device and amplitude dtype")
        if (initial_indices < 0).any() or (initial_indices >= len(self.dictionary.weights)).any() or not torch.isfinite(initial_m0).all():
            raise ValueError("Initial state has invalid indices or amplitudes")
        if num_samples < 1 or burn_in < 0 or thinning < 1:
            raise ValueError("Need num_samples >= 1, burn_in >= 0 and thinning >= 1")
        indices, amplitude = initial_indices.clone(), initial_m0.clone() * self.support
        auxiliary = self.maps(indices, amplitude)
        draws = []
        for sweep in range(burn_in + num_samples * thinning):
            completed = self.impute(indices, amplitude, generator)
            indices, amplitude = self.physics_draw(completed, auxiliary, generator)
            physical = self.maps(indices, amplitude)
            auxiliary = self.prior_draw(physical, self.coupling, generator)
            if auxiliary.shape != physical.shape or auxiliary.dtype != physical.dtype or auxiliary.device != physical.device:
                raise ValueError("Prior draw must preserve shape, real dtype and device")
            if not torch.isfinite(auxiliary).all():
                raise ValueError("Prior draw returned nonfinite maps")
            if sweep >= burn_in and (sweep - burn_in + 1) % thinning == 0:
                draws.append(physical)
        return torch.stack(draws)
