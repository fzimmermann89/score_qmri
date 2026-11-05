"""Physical coordinates and Gaussian measurements shared by posterior samplers."""

from collections.abc import Callable
from math import isfinite

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from ._utils import normal_like
from .protocols import LinearOperatorProtocol, OperatorProtocol


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
        operator: OperatorProtocol,
        measurement: Tensor,
        template: Tensor,
        kind: DatasetKind | int,
        noise_std: float,
        measurement_batch_dim: int,
    ) -> None:
        """Initialize image coordinates and the Gaussian likelihood.

        Parameters
        ----------
        model
            Denoiser defining the image scales, precision and device.
        operator
            Tuple-returning forward model, with an adjoint if linear.
        measurement
            Observed data. A single measurement can be shared across chains.
        template
            Physical image shape and real/complex dtype; values are ignored.
        kind
            Dataset kind used by the denoiser.
        noise_std
            Measurement noise standard deviation per real coordinate.
        measurement_batch_dim
            Axis holding independent samples in the operator output.
        """
        if not isfinite(noise_std) or noise_std <= 0:
            raise ValueError("noise_std must be finite and positive, per real coordinate")
        if template.ndim < 3 or any(size != 1 for size in template.shape[1:-2]):
            raise ValueError("template must have shape (B, ..., H, W) with singleton nonspatial dimensions")
        if not template.is_floating_point() and not template.is_complex():
            raise ValueError("template must have a floating-point or complex dtype")

        self.complex = is_complex_kind(int(kind))
        if template.is_complex() != self.complex:
            raise ValueError("template dtype must match the real/complex dataset kind")

        self.operator = operator
        self.noise_std = noise_std
        self.shape = template.shape
        self.batch_dim = measurement_batch_dim
        self.linear = isinstance(operator, LinearOperatorProtocol)

        model_dtype = next(model.parameters()).dtype if isinstance(model, torch.nn.Module) else template.real.dtype
        complex_dtype = torch.complex128 if model_dtype == torch.float64 else torch.complex64
        self.template = template.to(device=model.device, dtype=complex_dtype if self.complex else model_dtype)
        self.kind = torch.full((template.shape[0],), int(kind), device=model.device, dtype=torch.long)
        unit_image = torch.ones(1, 2, 1, 1, device=model.device, dtype=model_dtype)
        self.scale = model.normalize(unit_image, self.kind[:1])[:, :1].detach()

        measurement_dtype = complex_dtype if measurement.is_complex() else model_dtype
        self.measurement = measurement.to(device=model.device, dtype=measurement_dtype).detach().moveaxis(measurement_batch_dim, 0)
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
        prediction = self.operator(self.decode(state))[0].moveaxis(self.batch_dim, 0)
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

            def adjoint(cotangent: Tensor) -> Tensor:
                image = self.operator.H(cotangent.moveaxis(0, self.batch_dim))[0]  # type: ignore[attr-defined]
                return (self.pack(image) / self.scale).to(state)

            return self.forward, adjoint

        from mrpro.operators import Jacobian

        # Jacobian accepts callables at runtime despite its Operator annotation.
        jacobian = Jacobian(lambda point: (self.forward(point),), state.detach())  # type: ignore[arg-type, unused-ignore]
        return lambda tangent: jacobian(tangent)[0], lambda cotangent: jacobian.H(cotangent)[0]

    @torch.enable_grad()
    def gradient(self, state: Tensor) -> Tensor:
        """Evaluate the Gaussian negative-log-likelihood gradient."""
        if self.linear:
            _, adjoint = self.linearize(state)
            return adjoint(self.residual(state)) / self.noise_std**2

        point = state.detach().requires_grad_(True)
        loss = 0.5 * self.residual(point).abs().square().sum() / self.noise_std**2
        (gradient,) = torch.autograd.grad(loss, point)
        return gradient.detach()
