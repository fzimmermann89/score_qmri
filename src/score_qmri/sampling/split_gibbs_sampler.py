"""Dictionary split Gibbs with MR2/mrpro signal and acquisition operators.

The signal is m0*s(theta), with arbitrary real dictionary coordinates theta.
Missing-data augmentation preserves a white Gaussian likelihood for A=P U S,
where P selects Cartesian samples, U is unitary, and S has a pointwise Gram.
Exact prior conditionals preserve the weighted discrete posterior with
Gaussian-smoothed priors. A learned, finite reverse SDE approximates that step.
All public Gaussian standard deviations are per real coordinate.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from mrpro.operators import LinearOperator
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from .protocols import LinearOperatorProtocol

Coupling = tuple[float, ...]
PriorDraw = Callable[[Tensor, Coupling, torch.Generator | None], Tensor]
SignalModel = Callable[..., tuple[Tensor, ...]]


def _positive(values: tuple[float, ...]) -> None:
    if not all(isfinite(x) and x > 0 for x in values):
        raise ValueError("Scales and standard deviations must be finite and positive")


def _noise(x: Tensor, generator: torch.Generator | None) -> Tensor:
    # torch.randn(complex) has variance 1/2 per real component.
    if x.is_complex():
        return torch.complex(_noise(x.real, generator), _noise(x.real, generator))
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


@dataclass(frozen=True)
class SignalDictionary:
    """Physical coordinates (J,K), unit-M0 complex signals (J,L), weights (J,).

    Cell weights specify integration measure once. Atom normalization changes
    the amplitude prior and must not be applied. Coordinates may be any finite
    real values valid for the supplied signal model, including off-resonance.
    """

    parameters: Tensor
    fingerprints: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        if self.parameters.ndim != 2 or min(self.parameters.shape) == 0:
            raise ValueError("parameters must have nonempty shape (J,K)")
        j = self.parameters.shape[0]
        if self.fingerprints.ndim != 2 or self.fingerprints.shape[0] != j or self.fingerprints.shape[1] == 0:
            raise ValueError("fingerprints must have shape (J,L) with L > 0")
        if self.weights.shape != (j,) or not self.fingerprints.is_complex():
            raise ValueError("weights must have shape (J,) and fingerprints must be complex")
        if not self.parameters.is_floating_point() or self.weights.dtype != self.parameters.dtype:
            raise ValueError("parameters and weights must have the same real floating dtype")
        if self.fingerprints.real.dtype != self.parameters.dtype:
            raise ValueError("fingerprints and parameters must have matching precision")
        if any(
            t.device != self.parameters.device or not torch.isfinite(t).all() for t in (self.parameters, self.fingerprints, self.weights)
        ):
            raise ValueError("Dictionary tensors must be finite and on the same device")
        if (self.weights <= 0).any():
            raise ValueError("Dictionary weights must be positive")

    @classmethod
    @torch.no_grad()
    def from_model(cls, signal_model: SignalModel, parameters: Tensor, weights: Tensor) -> "SignalDictionary":
        """Evaluate a tuple-returning model at unit complex amplitude.

        The model is called as ``signal_model(m0, *theta)`` on vectors (J,)
        and must return one signal tensor (L,J), as mrpro signal models do.
        It must be homogeneous linear in complex m0. EPG sequences can use a
        caller adapter that packs their acquisition list into this convention.
        """
        if parameters.ndim != 2 or min(parameters.shape) == 0 or not parameters.is_floating_point():
            raise ValueError("parameters must be a nonempty real floating (J,K) grid")
        amplitude = torch.complex(torch.ones_like(parameters[:, 0]), torch.zeros_like(parameters[:, 0]))
        (signal,) = signal_model(amplitude, *parameters.unbind(-1))
        if signal.ndim != 2 or signal.shape[1] != len(parameters):
            raise ValueError("Signal model must return a single (L,J) tensor")
        return cls(parameters, signal.T.contiguous(), weights)


class CartesianAcquisition(LinearOperator):
    """A=P U S with explicit mrpro linear factors for exact data completion.

    Signal input is (B,L,1,1,H,W). S expands coils, U acts unitarily on that
    coil grid, and P selects samples or applies a binary Cartesian mask. P.H
    zero-fills, so compressed and full-grid measurements are both supported.
    Known white coil noise and a spatially local S.H S are required. Numerical
    checks catch common incompatible factors; the operator identities remain
    requirements on caller-supplied factors, not inferred properties.
    """

    def __init__(self, sampling: LinearOperatorProtocol, unitary: LinearOperatorProtocol, sensitivity: LinearOperatorProtocol) -> None:
        """Store P, U and S without constructing coils, masks or Fourier grids."""
        super().__init__()
        self.sampling, self.unitary, self.sensitivity = sampling, unitary, sensitivity

    def forward(self, signal: Tensor) -> tuple[Tensor]:
        (coils,) = self.sensitivity(signal)
        (full,) = self.unitary(coils)
        (result,) = self.sampling(full)
        return (result,)

    def adjoint(self, measurement: Tensor) -> tuple[Tensor]:
        (full,) = self.sampling.H(measurement)
        (coils,) = self.unitary.H(full)
        (result,) = self.sensitivity.H(coils)
        return (result,)

    @torch.no_grad()
    def validate(self, template: Tensor) -> Tensor:
        """Check the factorization and return pointwise power (1,H,W,L)."""
        ones = torch.ones_like(template)
        (coils,) = self.sensitivity(ones)
        (power,) = self.sensitivity.H(coils)
        (full,) = self.unitary(coils)
        grid = torch.arange(template.numel(), device=template.device, dtype=template.real.dtype).reshape(template.shape)
        probe = torch.complex(grid.sin(), (0.7 * grid).cos())
        kgrid = torch.arange(full.numel(), device=full.device, dtype=full.real.dtype).reshape(full.shape)
        kprobe = torch.complex((0.3 * kgrid).sin(), (0.9 * kgrid).cos())
        tolerance = 2e-5 if template.dtype == torch.complex64 else 1e-10
        try:
            torch.testing.assert_close(self.sensitivity.H(self.sensitivity(probe)[0])[0], power * probe, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(self.unitary.H(self.unitary(coils)[0])[0], coils, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(self.unitary(self.unitary.H(kprobe)[0])[0], kprobe, atol=tolerance, rtol=tolerance)
            (selected,) = self.sampling(torch.ones_like(full))
            (mask,) = self.sampling.H(selected)
            torch.testing.assert_close(mask, (mask.real > 0.5).to(mask.dtype), atol=tolerance, rtol=0)
            torch.testing.assert_close(self.sampling.H(self.sampling(kprobe)[0])[0], mask * kprobe, atol=tolerance, rtol=tolerance)
        except (AssertionError, RuntimeError) as error:
            raise ValueError("Exact Gibbs requires a Cartesian selector, unitary transform and pointwise sensitivity Gram") from error
        if power.shape != template.shape or (power.real < 0).any() or not torch.isfinite(power).all() or power.imag.abs().max() > tolerance:
            raise ValueError("Sensitivity Gram must be finite nonnegative pointwise power with signal shape")
        return power.real[:, :, 0, 0].moveaxis(1, -1)

    def complete(self, signal: Tensor, measurement: Tensor, noise_std: float, generator: torch.Generator | None) -> Tensor:
        """Draw unsampled data with acquisition noise; retain measured entries."""
        (full,) = self.unitary(self.sensitivity(signal)[0])
        noisy = full + noise_std * _noise(full, generator)
        (observed,) = self.sampling.H(measurement)
        (projection,) = self.sampling.H(self.sampling(noisy)[0])
        return observed + (noisy - projection)


@dataclass(frozen=True)
class SplitGibbsConfig:
    """Fixed physical coupling SDs: one per real map, then complex amplitude."""

    coupling: Coupling
    num_samples: int = 1
    burn_in: int = 0
    thinning: int = 1
    prior_steps: int = 128
    prior_sigma_min: float | None = None
    dictionary_chunk: int = 256

    def __post_init__(self) -> None:
        _positive(self.coupling)
        if len(self.coupling) < 2 or self.num_samples < 1 or self.burn_in < 0 or self.thinning < 1:
            raise ValueError("Need a real map and amplitude coupling, positive sample/thinning counts, nonnegative burn-in")
        if self.prior_steps < 1 or self.dictionary_chunk < 1:
            raise ValueError("Prior steps and dictionary chunk must be positive")
        if self.prior_sigma_min is not None:
            _positive((self.prior_sigma_min,))


class DiffusionPriorDraw:
    """Product-prior reverse SDEs starting at Gaussian observations."""

    def __init__(
        self,
        model: DiffusionModel,
        kind: Sequence[DatasetKind],
        dataset_scales: Coupling,
        *,
        steps: int = 128,
        sigma_min: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            Evaluation-mode denoiser. Its public normalization supplies kind scales.
        kind
            One real-map kind per dictionary coordinate, then a complex amplitude kind.
        dataset_scales
            Physical-to-training-dataset multipliers before kind normalization.
            Specify amplitude scaling for the acquisition; no universal M0 scale exists.
        steps
            Euler--Maruyama steps on a geometric noise schedule.
        sigma_min
            Nonzero model-coordinate endpoint, at least model.sigma_min. It remains
            noisy; no clipping, likelihood guidance or deterministic mean replacement.
        """
        _positive(dataset_scales)
        if len(kind) < 2 or len(kind) != len(dataset_scales) or any(is_complex_kind(k) for k in kind[:-1]) or not is_complex_kind(kind[-1]):
            raise ValueError("Need matching real map kinds/scales followed by one complex amplitude kind")
        self.model, self.kind, self.dataset_scales = model, tuple(kind), dataset_scales
        self.steps = steps
        self.sigma_min = model.sigma_min if sigma_min is None else sigma_min
        _positive((self.sigma_min,))
        if steps < 1 or self.sigma_min < model.sigma_min or model.training:
            raise ValueError("Need an evaluation model, steps >= 1 and endpoint >= model.sigma_min")

    def physical_scales(self, reference: Tensor) -> Tensor:
        """Return physical-to-model multipliers from the configured model."""
        kinds = torch.tensor(self.kind, device=reference.device)
        unit = torch.ones(len(self.kind), 2, 1, 1, device=reference.device, dtype=reference.dtype)
        scales = self.model.normalize(unit, kinds)[:, 0, 0, 0] * reference.new_tensor(self.dataset_scales)
        if not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Physical-to-model scales must be finite and positive")
        return scales

    @torch.no_grad()
    def __call__(self, observation: Tensor, coupling: Coupling, generator: torch.Generator | None) -> Tensor:
        """Return physical auxiliary maps, retaining the observation dtype."""
        _positive(coupling)
        nreal = len(self.kind) - 1
        if (
            observation.ndim != 4
            or observation.shape[1] != nreal + 2
            or not observation.is_floating_point()
            or len(coupling) != len(self.kind)
        ):
            raise ValueError("Need real observation (B,K+2,H,W) and one coupling per kind")
        parameter = next(self.model.parameters(), observation)
        if parameter.device != observation.device:
            raise ValueError("Model and observation must be on the same device")
        batch = observation.shape[0]
        scales = self.physical_scales(observation)
        start = scales * observation.new_tensor(coupling)
        if not torch.isfinite(start).all() or (start <= self.sigma_min).any() or (start > self.model.sigma_max).any():
            raise ValueError("Scaled coupling levels must lie above endpoint and within model.sigma_max")
        real = observation[:, :nreal].moveaxis(1, 0).flatten(0, 1)
        x = torch.cat((torch.stack((real, torch.zeros_like(real)), 1), observation[:, nreal:]), 0)
        scale = scales.repeat_interleave(batch)[:, None, None, None]
        kinds = torch.tensor(self.kind, device=observation.device).repeat_interleave(batch)
        x = x * scale
        initial_levels = start.repeat_interleave(batch)
        levels = initial_levels
        for step in range(self.steps):
            next_levels = initial_levels * (self.sigma_min / initial_levels) ** ((step + 1) / self.steps)
            delta = (levels.square() - next_levels.square())[:, None, None, None]
            denoised = self.model.denoise(x.to(parameter.dtype), levels.to(parameter.dtype), kinds).to(x.dtype)
            noise = _noise(x, generator)
            noise[: nreal * batch, 1] = 0
            x = x + delta * (denoised - x) / levels[:, None, None, None].square() + delta.sqrt() * noise
            x[: nreal * batch, 1] = 0
            levels = next_levels
        x = x / scale
        real = x[: nreal * batch, 0].unflatten(0, (nreal, batch)).moveaxis(0, 1)
        return torch.cat((real, x[nreal * batch :]), 1)


class _SplitGibbsProblem:
    """Conditional calculations for one operator, dictionary and acquisition."""

    def __init__(
        self,
        operator: CartesianAcquisition,
        measurement: Tensor,
        dictionary: SignalDictionary,
        support: Tensor,
        template: Tensor,
        noise_std: float,
        cfg: SplitGibbsConfig,
        prior_draw: PriorDraw,
    ) -> None:
        _positive((noise_std,))
        if template.ndim != 3 or template.dtype != dictionary.fingerprints.dtype or template.device != dictionary.parameters.device:
            raise ValueError("template must be complex (B,H,W), matching dictionary precision/device")
        if support.shape != template.shape[-2:] or support.dtype != torch.bool or not support.any() or support.device != template.device:
            raise ValueError("Need nonempty boolean support (H,W) on template device")
        if len(cfg.coupling) != dictionary.parameters.shape[1] + 1:
            raise ValueError("Need one coupling per dictionary coordinate and one for amplitude")
        self.operator, self.measurement, self.dictionary, self.support = operator, measurement, dictionary, support
        self.template, self.noise_std, self.cfg, self.prior_draw = template, noise_std, cfg, prior_draw
        signal = template.new_ones(1, dictionary.fingerprints.shape[1], 1, 1, *support.shape)
        self.power = operator.validate(signal)
        predicted = operator(signal.expand(template.shape[0], -1, -1, -1, -1, -1))[0]
        if measurement.shape != predicted.shape or measurement.device != template.device or measurement.dtype != template.dtype:
            raise ValueError("Measurements must match forward output shape, complex dtype and device")
        # Unsampled entries of a full-grid mask are not observations.
        selected = operator.sampling(operator.sampling.H(measurement)[0])[0]
        if not torch.isfinite(selected).all():
            raise ValueError("Measured data must be finite")

    def maps(self, indices: Tensor, amplitude: Tensor) -> Tensor:
        coordinates = self.dictionary.parameters[indices].moveaxis(-1, 1)
        return torch.cat((coordinates, amplitude.real[:, None], amplitude.imag[:, None]), 1) * self.support

    def signal(self, indices: Tensor, amplitude: Tensor) -> Tensor:
        signal = self.dictionary.fingerprints[indices].moveaxis(-1, 1) * (amplitude * self.support)[:, None]
        return signal[:, :, None, None]

    def impute(self, indices: Tensor, amplitude: Tensor, generator: torch.Generator | None) -> Tensor:
        return self.operator.complete(self.signal(indices, amplitude), self.measurement, self.noise_std, generator)

    def collapsed_parameters(self, completed: Tensor, auxiliary: Tensor, start: int, stop: int) -> tuple[Tensor, Tensor, Tensor]:
        combined = self.operator.sensitivity.H(self.operator.unitary.H(completed)[0])[0][:, :, 0, 0].moveaxis(1, -1)
        return self._collapse(combined, auxiliary, start, stop)

    def _collapse(self, combined: Tensor, auxiliary: Tensor, start: int, stop: int) -> tuple[Tensor, Tensor, Tensor]:
        atoms = self.dictionary.fingerprints[start:stop]
        rho = self.cfg.coupling
        precision = self.power @ atoms.abs().square().T / self.noise_std**2 + 1 / rho[-1] ** 2
        amplitude = torch.complex(auxiliary[:, -2], auxiliary[:, -1])
        rhs = combined @ atoms.conj().T / self.noise_std**2 + amplitude[..., None] / rho[-1] ** 2
        mean = rhs / precision
        penalty = (
            (self.dictionary.parameters[start:stop] - auxiliary[:, :-2].moveaxis(1, -1)[..., None, :]) / auxiliary.new_tensor(rho[:-1])
        ).square().sum(-1) / 2
        logits = self.dictionary.weights[start:stop].log() + rhs.abs().square() / (2 * precision) - precision.log() - penalty
        return logits, mean, precision.reciprocal().expand_as(logits)

    def physics_draw(self, completed: Tensor, auxiliary: Tensor, generator: torch.Generator | None) -> tuple[Tensor, Tensor]:
        combined = self.operator.sensitivity.H(self.operator.unitary.H(completed)[0])[0][:, :, 0, 0].moveaxis(1, -1)
        shape = auxiliary[:, 0].shape
        best = auxiliary.new_full(shape, -torch.inf)
        indices = torch.zeros(shape, device=auxiliary.device, dtype=torch.long)
        mean = torch.zeros(shape, device=auxiliary.device, dtype=completed.dtype)
        variance = torch.zeros_like(best)
        for start in range(0, len(self.dictionary.weights), self.cfg.dictionary_chunk):
            stop = min(start + self.cfg.dictionary_chunk, len(self.dictionary.weights))
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


class SplitGibbsSampler:
    """Dictionary likelihood conditionals and Gaussian auxiliary prior draws."""

    def __init__(self, model: DiffusionModel | None, cfg: SplitGibbsConfig) -> None:
        """Use model=None only with an explicit analytic prior_draw at sampling."""
        self.model, self.cfg = model, cfg

    @torch.no_grad()
    def sample(
        self,
        forward_op: CartesianAcquisition,
        measurement: Tensor,
        *,
        noise_std: float,
        template: Tensor,
        dictionary: SignalDictionary,
        support: Tensor,
        kind: Sequence[DatasetKind] | None = None,
        dataset_scales: Coupling | None = None,
        initial_indices: Tensor | None = None,
        initial_m0: Tensor | None = None,
        prior_draw: PriorDraw | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor]:
        """Return physical draws as ``(maps,)``, shape (draw,B,K+2,H,W).

        Real coordinates appear in dictionary order, followed by Re/Im(M0).
        template is complex (B,H,W); values are ignored. Priors see the full
        zero-padded physical maps; auxiliary backgrounds remain random. kind
        and dataset_scales have K+1 entries and are required for learned priors.
        A custom prior_draw samples p(u|maps,coupling) in these physical units.
        Retained states use fixed couplings and can remain correlated.
        """
        cfg = self.cfg
        if prior_draw is None:
            if self.model is None or kind is None or dataset_scales is None:
                raise ValueError("Learned priors require model, map kinds and explicit physical-to-dataset scales")
            prior_draw = DiffusionPriorDraw(self.model, kind, dataset_scales, steps=cfg.prior_steps, sigma_min=cfg.prior_sigma_min)
        problem = _SplitGibbsProblem(forward_op, measurement, dictionary, support, template, noise_std, cfg, prior_draw)
        indices = (
            torch.zeros(template.shape, device=template.device, dtype=torch.long) if initial_indices is None else initial_indices.clone()
        )
        amplitude = torch.zeros_like(template) if initial_m0 is None else initial_m0.clone()
        if indices.shape != template.shape or indices.dtype != torch.long or indices.device != template.device:
            raise ValueError("Initial dictionary indices must be long, template-shaped and on the same device")
        if amplitude.shape != template.shape or amplitude.dtype != template.dtype or amplitude.device != template.device:
            raise ValueError("Initial amplitude must match template shape, complex dtype and device")
        if (indices < 0).any() or (indices >= len(dictionary.weights)).any() or not torch.isfinite(amplitude).all():
            raise ValueError("Initial state has invalid indices or amplitudes")
        amplitude = amplitude * support
        auxiliary = problem.maps(indices, amplitude)
        draws = []
        for sweep in range(cfg.burn_in + cfg.num_samples * cfg.thinning):
            completed = problem.impute(indices, amplitude, generator)
            indices, amplitude = problem.physics_draw(completed, auxiliary, generator)
            physical = problem.maps(indices, amplitude)
            auxiliary = prior_draw(physical, cfg.coupling, generator)
            if (
                auxiliary.shape != physical.shape
                or auxiliary.dtype != physical.dtype
                or auxiliary.device != physical.device
                or not torch.isfinite(auxiliary).all()
            ):
                raise ValueError("Prior draw must return finite maps with the physical shape, dtype and device")
            if sweep >= cfg.burn_in and (sweep - cfg.burn_in + 1) % cfg.thinning == 0:
                draws.append(physical)
        return (torch.stack(draws),)
