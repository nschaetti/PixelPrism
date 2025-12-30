"""Elementwise exponentiation operation."""

from __future__ import annotations

from typing import Tuple

from ._helpers import select_ops
from .math_expr import MathExpr
from .op import Op
from .value import Value

__all__ = ["Pow"]


class Pow(Op):
    """Elementwise exponentiation."""

    def __init__(self, base: MathExpr, exponent: MathExpr):
        """Initialize a Pow operation.

        Args:
            base: Base expression.
            exponent: Exponent expression.

        Raises:
            ValueError: If operand dtypes are incompatible.
        """
        if base.dtype != exponent.dtype:
            raise ValueError("Pow requires matching dtypes.")
        # end if
        shape = base.shape.merge_elementwise(exponent.shape)
        super().__init__((base, exponent), shape, base.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate exponentiation.

        Args:
            child_values: Tuple containing base and exponent Values.

        Returns:
            Value: Runtime result of exponentiation.
        """
        base, exponent = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "pow"):
            data = backend.pow(base.get(), exponent.get())
        else:
            data = pow(base.get(), exponent.get())
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl
# end class Pow

