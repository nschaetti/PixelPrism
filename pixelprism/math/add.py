"""Elementwise addition operation."""

from __future__ import annotations

from typing import Tuple

from ._helpers import select_ops
from .math_expr import MathExpr
from .op import Op
from .shape import Shape
from .value import Value

__all__ = ["Add"]


class Add(Op):
    """Elementwise addition."""

    def __init__(self, left: MathExpr, right: MathExpr):
        """Initialize an Add operation.

        Args:
            left: Left operand.
            right: Right operand.

        Raises:
            ValueError: If the operand dtypes are incompatible.
        """
        if left.dtype != right.dtype:
            raise ValueError("Add requires matching dtypes.")
        # end if
        shape = left.shape.merge_elementwise(right.shape)
        super().__init__((left, right), shape, left.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the addition.

        Args:
            child_values: Tuple containing left and right Values.

        Returns:
            Value: Runtime result of the addition.
        """
        left, right = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "add"):
            data = backend.add(left.get(), right.get())
        else:
            data = left.get() + right.get()
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl
# end class Add

