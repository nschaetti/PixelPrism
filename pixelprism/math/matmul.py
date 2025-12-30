"""Tensor or matrix multiplication operation."""

from __future__ import annotations

from typing import Tuple

from ._helpers import select_ops
from .math_expr import MathExpr
from .op import Op
from .value import Value

__all__ = ["MatMul"]


class MatMul(Op):
    """Tensor or matrix multiplication."""

    def __init__(self, left: MathExpr, right: MathExpr):
        """Initialize a MatMul operation.

        Args:
            left: Left operand.
            right: Right operand.

        Raises:
            ValueError: If operand dtypes or shapes are incompatible.
        """
        if left.dtype != right.dtype:
            raise ValueError("MatMul requires matching dtypes.")
        # end if
        shape = left.shape.matmul_result(right.shape)
        super().__init__((left, right), shape, left.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the matrix multiplication.

        Args:
            child_values: Tuple containing left and right Values.

        Returns:
            Value: Runtime result of matrix multiplication.
        """
        left, right = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "matmul"):
            data = backend.matmul(left.get(), right.get())
        else:
            data = left.get() @ right.get()
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl
# end class MatMul

