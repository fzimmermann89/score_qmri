"""Tuple-returning operators compatible with mr2/mrpro."""

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class OperatorProtocol(Protocol):
    """A differentiable forward model with one tensor input and output."""

    def __call__(self, x: Tensor) -> tuple[Tensor, ...]: ...


@runtime_checkable
class LinearOperatorProtocol(Protocol):
    """Protocol for linear operators compatible with mrpro.

    Operators return tuples to match mrpro's convention where
    forward/adjoint operations return `(output,)`.
    """

    def __call__(self, x: Tensor) -> tuple[Tensor, ...]:
        """Apply forward operator A: image → k-space."""
        ...

    @property
    def H(self) -> "LinearOperatorProtocol":
        """Adjoint operator A†: k-space → image."""
        ...
