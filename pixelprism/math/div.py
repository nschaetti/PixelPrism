"""Elementwise division operation."""

from __future__ import annotations

from typing import Tuple

from ._helpers import select_ops
from .math_expr import MathExpr
from .op import Op
from .value import Value

__all__ = ["Div"]


class Div(Op):
    """Elementwise division."""

    def __init__(self, left: MathExpr, right: MathExpr):
        """Initialize a Div operation.

        Args:
            left: Numerator operand.
            right: Denominator operand.

        Raises:
            ValueError: If the operand dtypes are incompatible.
        """
        if left.dtype != right.dtype:
            raise ValueError("Div requires matching dtypes.")
        # end if
        shape = left.shape.merge_elementwise(right.shape)
        super().__init__((left, right), shape, left.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the division.

        Args:
            child_values: Tuple containing numerator and denominator Values.

        Returns:
            Value: Runtime result of the division.
        """
        left, right = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "div"):
            data = backend.div(left.get(), right.get())
        else:
            data = left.get() / right.get()
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl
# end class Div

