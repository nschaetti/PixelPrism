"""Helper utilities for math tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pixelprism.math import MathExpr, Shape, Value


@dataclass(frozen=True)
class FakeTensor:
    """Simple tensor-like object supporting arithmetic operations."""

    value: Any

    def _coerce(self, other: Any) -> Any:
        """Extract the underlying value."""
        return other.value if isinstance(other, FakeTensor) else other
    # end def _coerce

    def __add__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self.value + self._coerce(other))
    # end def __add__

    def __sub__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self.value - self._coerce(other))
    # end def __sub__

    def __mul__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self.value * self._coerce(other))
    # end def __mul__

    def __truediv__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self.value / self._coerce(other))
    # end def __truediv__

    def __neg__(self) -> "FakeTensor":
        return FakeTensor(-self.value)
    # end def __neg__

    def __pow__(self, exponent: Any, modulo=None) -> "FakeTensor":
        return FakeTensor(self.value ** self._coerce(exponent))
    # end def __pow__

    def __matmul__(self, other: Any) -> "FakeTensor":
        return FakeTensor(self.value * self._coerce(other))
    # end def __matmul__
# end class FakeTensor


def make_value(data: Any, dims: Sequence[int | None], dtype: str = "float32") -> Value:
    """Create a Value helper."""
    return Value(data, Shape(dims), dtype)
# end def make_value


class ValueExpr(MathExpr):
    """Expression that returns a preset Value."""

    def __init__(self, value: Value):
        super().__init__(value.shape, value.dtype, ())
        self._value = value
    # end def __init__

    def evaluate(self, env: Mapping):  # type: ignore[override]
        """Return the stored value."""
        return self._value
    # end def evaluate
# end class ValueExpr

