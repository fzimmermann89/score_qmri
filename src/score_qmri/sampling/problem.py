"""Physical coordinates and Gaussian measurements shared by posterior samplers."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import isfinite

import torch
from mr2.operators import LinearOperator
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from ._utils import normal_like


@dataclass
class MRIProblem:
    """Gaussian measurements of a linear encoding applied to a signal model.

    Parameters
    ----------
    encoding
        MR2 linear operator mapping images to measurements.
    measurement
        Observed measurements in the encoding's output layout.
    noise_std
        Gaussian noise standard deviation per real coordinate, also for complex data.
    signal_model
        Optional tuple-returning signal model, called with the physical parameter
        maps. Without it, the parameter is the image itself.
    measurement_batch_dim
        Axis holding independent samples in the measurements.
    """

    encoding: LinearOperator
    measurement: Tensor
    noise_std: float
    signal_model: Callable[..., tuple[Tensor]] | None = None
    measurement_batch_dim: int = 0

    def __post_init__(self) -> None:
        if not isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be finite and positive, per real coordinate")

    def __call__(self, *parameters: Tensor) -> tuple[Tensor]:
        """Evaluate A(q(parameters)), or A(image) when no signal model is set."""
        (image,) = parameters if self.signal_model is None else self.signal_model(*parameters)
        prediction: tuple[Tensor] = self.encoding(image)
        return prediction

    def residual(self, *parameters: Tensor) -> Tensor:
        """Return the predicted measurements minus the observed measurements."""
        return self(*parameters)[0] - self.measurement

    @torch.enable_grad()
    def gradient(self, *parameters: Tensor) -> tuple[Tensor, ...]:
        """Gaussian negative-log-likelihood gradients in physical coordinates."""
        points = tuple(parameter.detach().requires_grad_(True) for parameter in parameters)
        (image,) = points if self.signal_model is None else self.signal_model(*points)
        with torch.no_grad():
            (backprojection,) = self.encoding.H(self.encoding(image.detach())[0] - self.measurement)
            if not image.is_complex():
                backprojection = backprojection.real
            backprojection = backprojection.to(image) / self.noise_std**2
        return torch.autograd.grad(image, points, backprojection)


class SamplingProblem:
    """Gaussian MRI likelihood in normalized, two-channel diffusion coordinates.

    Physical images have shape (B, ..., H, W), with singleton nonspatial
    dimensions. Measurements are stored batch-first, including when the
    operator places the batch after a contrast or time dimension.
    Operators must act independently on each batch sample.
    """

    def __init__(
        self,
        model: DiffusionModel,
        problem: MRIProblem,
        template: Tensor,
        kind: DatasetKind | int,
    ) -> None:
        """Adapt physical images to denoiser coordinates.

        Parameters
        ----------
        model
            Denoiser defining image scales, precision and device.
        problem
            Physical forward model and Gaussian measurements.
        template
            Physical image shape and real/complex dtype; values are ignored.
        kind
            Dataset kind used by the denoiser.
        """
        if template.ndim < 3 or any(size != 1 for size in template.shape[1:-2]):
            raise ValueError("template must have shape (B, ..., H, W) with singleton nonspatial dimensions")
        if not template.is_floating_point() and not template.is_complex():
            raise ValueError("template must have a floating-point or complex dtype")

        self.complex = is_complex_kind(int(kind))
        if template.is_complex() != self.complex:
            raise ValueError("template dtype must match the real/complex dataset kind")

        self.noise_std = problem.noise_std
        self.shape = template.shape
        self.batch_dim = problem.measurement_batch_dim
        self.linear = problem.signal_model is None

        model_dtype = next(model.parameters()).dtype if isinstance(model, torch.nn.Module) else template.real.dtype
        complex_dtype = torch.complex128 if model_dtype == torch.float64 else torch.complex64
        self.template = template.to(device=model.device, dtype=complex_dtype if self.complex else model_dtype)
        self.kind = torch.full((template.shape[0],), int(kind), device=model.device, dtype=torch.long)
        unit_image = torch.ones(1, 2, 1, 1, device=model.device, dtype=model_dtype)
        self.scale = model.normalize(unit_image, self.kind[:1])[:, :1].detach()

        measurement = problem.measurement
        measurement_dtype = complex_dtype if measurement.is_complex() else model_dtype
        self.problem = replace(problem, measurement=measurement.to(device=model.device, dtype=measurement_dtype).detach())
        self.measurement = self.problem.measurement.moveaxis(self.batch_dim, 0)
        self.measurement = self.measurement.expand(template.shape[0], *self.measurement.shape[1:])

    def pack(self, image: Tensor) -> Tensor:
        """Convert physical layout to (B, 2, H, W) without changing units."""
        image = image.reshape(self.shape[0], *self.shape[-2:])
        imaginary = image.imag if self.complex else torch.zeros_like(image.real)
        return torch.stack([image.real, imaginary], dim=1)

    def encode(self, image: Tensor) -> Tensor:
        """Scale a physical image into diffusion coordinates."""
        return self.pack(image) * self.scale

    def decode(self, state: Tensor) -> Tensor:
        """Convert a diffusion state to a physical image in the template layout."""
        state = state / self.scale
        image = torch.complex(state[:, 0], state[:, 1]) if self.complex else state[:, 0]
        return image.reshape(self.shape)

    def noise(self, generator: torch.Generator | None) -> Tensor:
        """Draw unit Gaussian noise per active real diffusion coordinate."""
        return self.pack(normal_like(self.template, generator))

    def start(self, sigma: float, generator: torch.Generator | None, noise: Tensor | None, initial: Tensor | None = None) -> Tensor:
        """Initialize from unit diffusion noise and an optional physical image."""
        if noise is None:
            standard_noise = self.noise(generator)
        else:
            if noise.shape != self.shape or noise.is_complex() != self.complex or not noise.isfinite().all():
                raise ValueError("noise must match template shape and real/complex kind and be finite")
            standard_noise = self.pack(noise.to(self.template)).detach()

        state = sigma * standard_noise
        if initial is not None:
            if initial.shape != self.shape or initial.is_complex() != self.complex or not initial.isfinite().all():
                raise ValueError("initial must match template shape and real/complex kind and be finite")
            state = state + self.encode(initial.to(self.template)).detach()
        return state

    def forward(self, state: Tensor) -> Tensor:
        """Evaluate physics in physical units, returning batch-first measurements."""
        prediction = self.problem(self.decode(state))[0].moveaxis(self.batch_dim, 0)
        if prediction.is_complex() and not self.measurement.is_complex():
            raise ValueError("Complex forward output requires complex measurements")
        if prediction.shape != self.measurement.shape:
            raise ValueError("Forward output must match measurements, including the explicit sample dimension")

        complex_dtype = torch.complex128 if state.dtype == torch.float64 else torch.complex64
        return prediction.to(complex_dtype if self.measurement.is_complex() else state.dtype)

    def residual(self, state: Tensor) -> Tensor:
        """Return prediction minus measurement in batch-first layout."""
        return self.forward(state) - self.measurement

    def linearize(self, state: Tensor) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
        """Return Jacobian and adjoint products, frozen at the diffusion state."""
        if self.linear:

            def linear_adjoint(cotangent: Tensor) -> Tensor:
                image = self.problem.encoding.H(cotangent.moveaxis(0, self.batch_dim))[0]
                return (self.pack(image) / self.scale).to(state)

            return self.forward, linear_adjoint

        point = state.detach()
        _, pullback, *_ = torch.func.vjp(self.forward, point)

        def jacobian(tangent: Tensor) -> Tensor:
            result: Tensor = torch.func.jvp(self.forward, (point,), (tangent,))[1]
            return result

        def adjoint(cotangent: Tensor) -> Tensor:
            result: Tensor = pullback(cotangent)[0]
            return result

        return jacobian, adjoint

    def gradient(self, state: Tensor) -> Tensor:
        """Evaluate the Gaussian negative-log-likelihood gradient."""
        (gradient,) = self.problem.gradient(self.decode(state))
        return (self.pack(gradient) / self.scale).to(state)
