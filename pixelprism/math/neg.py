"""Elementwise negation operation."""

from __future__ import annotations

from typing import Tuple

from ._helpers import select_ops
from .math_expr import MathExpr
from .op import Op
from .value import Value

__all__ = ["Neg"]


class Neg(Op):
    """Elementwise negation."""

    def __init__(self, operand: MathExpr):
        """Initialize a Neg operation.

        Args:
            operand: Expression to negate.
        """
        super().__init__((operand,), operand.shape, operand.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the negation.

        Args:
            child_values: Tuple containing the operand Value.

        Returns:
            Value: Runtime result of the negation.
        """
        (value,) = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "neg"):
            data = backend.neg(value.get())
        else:
            data = -value.get()
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl
# end class Neg

