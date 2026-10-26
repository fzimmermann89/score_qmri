"""Physical coordinates and Gaussian measurements shared by the samplers."""

from collections.abc import Callable
from math import isfinite

import torch
from torch import Tensor

from score_qmri.datasets.kind import DatasetKind, is_complex_kind
from score_qmri.models.diffusion import DiffusionModel

from ._utils import normal_like
from .protocols import LinearOperatorProtocol, OperatorProtocol


class SamplingProblem:
    """One unknown per sample; physics uses physical units, diffusion uses 2ch.

    The state has shape (B, ..., H, W), with singleton dimensions between B
    and H. Measurements may place B on another axis, e.g. after IR time.
    Internally measurements are moved to batch-first for independent CG solves.
    The operator must act independently on each sample; it must not mix chains.
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
        if not isfinite(noise_std) or noise_std <= 0:
            raise ValueError("noise_std must be finite and positive, per real coordinate")
        if template.ndim < 3 or any(n != 1 for n in template.shape[1:-2]):
            raise ValueError("template must have shape (B, ..., H, W) with singleton nonspatial dimensions")
        self.complex = is_complex_kind(int(kind))
        if template.is_complex() != self.complex:
            raise ValueError("template dtype must match the real/complex dataset kind")
        if not template.is_floating_point() and not template.is_complex():
            raise ValueError("template must have a floating-point or complex dtype")
        self.model, self.operator, self.noise_std = model, operator, noise_std
        self.shape = template.shape
        self.batch_dim = measurement_batch_dim
        dtype = next(model.parameters()).dtype if isinstance(model, torch.nn.Module) else template.real.dtype
        physical_dtype = (torch.complex128 if dtype == torch.float64 else torch.complex64) if self.complex else dtype
        self.template = template.to(device=model.device, dtype=physical_dtype)
        measurement_dtype = (torch.complex128 if dtype == torch.float64 else torch.complex64) if measurement.is_complex() else dtype
        self.measurement = measurement.to(device=model.device, dtype=measurement_dtype).detach().moveaxis(measurement_batch_dim, 0)
        if self.measurement.shape[0] not in (1, template.shape[0]):
            raise ValueError("measurement sample dimension must be 1 or match template batch size")
        self.measurement = self.measurement.expand(template.shape[0], *self.measurement.shape[1:])
        self.kind = torch.full((template.shape[0],), int(kind), device=model.device, dtype=torch.long)
        self.scale = model.normalize(torch.ones(1, 2, 1, 1, device=model.device, dtype=dtype), self.kind[:1])[:, :1].detach()
        self.linear = isinstance(operator, LinearOperatorProtocol)

    def pack(self, value: Tensor) -> Tensor:
        value = value.reshape(self.shape[0], *self.shape[-2:])
        return torch.stack([value.real, value.imag if self.complex else torch.zeros_like(value.real)], dim=1)

    def unpack(self, value: Tensor) -> Tensor:
        image = torch.complex(value[:, 0], value[:, 1]) if self.complex else value[:, 0]
        return image.reshape(self.shape)

    def encode(self, value: Tensor) -> Tensor:
        return self.pack(value) * self.scale

    def decode(self, value: Tensor) -> Tensor:
        return self.unpack(value / self.scale)

    def constrain(self, value: Tensor) -> Tensor:
        if not self.complex:
            value = torch.stack([value[:, 0], torch.zeros_like(value[:, 0])], dim=1)
        return value

    def noise(self, generator: torch.Generator | None) -> Tensor:
        if self.complex:
            return self.pack(normal_like(self.template, generator))
        real = torch.randn(self.shape, device=self.template.device, dtype=self.template.dtype, generator=generator)
        return self.pack(real)

    def start(self, sigma: float, generator: torch.Generator | None, noise: Tensor | None, initial: Tensor | None = None) -> Tensor:
        if noise is None:
            epsilon = self.noise(generator)
        else:
            if noise.shape != self.shape or noise.is_complex() != self.complex or not noise.isfinite().all():
                raise ValueError("noise must match template shape and real/complex kind and be finite")
            # Supplied noise has unit variance in normalized diffusion coordinates.
            epsilon = self.pack(noise.to(self.template)).detach()
        x = sigma * epsilon
        if initial is not None:
            if initial.shape != self.shape or initial.is_complex() != self.complex or not initial.isfinite().all():
                raise ValueError("initial must match template shape and real/complex kind and be finite")
            x = x + self.encode(initial.to(self.template)).detach()
        return x

    def forward(self, x: Tensor) -> Tensor:
        result = self.operator(self.decode(x))[0].moveaxis(self.batch_dim, 0)
        if result.is_complex() and not self.measurement.is_complex():
            raise ValueError("Complex forward output requires complex measurements")
        result = result.to(self.measurement.dtype)
        if result.shape != self.measurement.shape:
            raise ValueError("Forward output must match measurements, including the explicit sample dimension")
        return result

    def residual(self, x: Tensor) -> Tensor:
        return self.forward(x) - self.measurement

    def linearize(self, x: Tensor) -> tuple[Callable[[Tensor], Tensor], Callable[[Tensor], Tensor]]:
        """Return Jacobian products in diffusion coordinates, frozen at x."""
        if self.linear:

            def jvp(v: Tensor) -> Tensor:
                return self.forward(v)

            def vjp(v: Tensor) -> Tensor:
                physical = self.operator.H(v.moveaxis(0, self.batch_dim))[0]  # type: ignore[attr-defined]
                # The fixed decoder is diagonal scaling plus a real/complex layout.
                return self.pack(physical) / self.scale

            return jvp, vjp
        from mrpro.operators import Jacobian

        # Jacobian uses only __call__, despite its concrete Operator annotation.
        jacobian = Jacobian(lambda z: (self.forward(z),), x.detach())  # type: ignore[arg-type, unused-ignore]
        return lambda v: jacobian(v)[0], lambda v: self.constrain(jacobian.H(v)[0])

    @torch.enable_grad()
    def gradient(self, x: Tensor) -> Tensor:
        """Gaussian negative-log-likelihood gradient at the actual state."""
        if self.linear:
            _, adjoint = self.linearize(x)
            return adjoint(self.residual(x)) / self.noise_std**2
        point = x.detach().requires_grad_(True)
        loss = 0.5 * self.residual(point).abs().square().sum() / self.noise_std**2
        (gradient,) = torch.autograd.grad(loss, point)
        return self.constrain(gradient).detach()
