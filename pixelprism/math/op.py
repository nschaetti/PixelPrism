"""Base class for symbolic operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping, Sequence, Tuple, TYPE_CHECKING

from .math_expr import MathExpr
from .shape import Shape
from .value import Value

if TYPE_CHECKING:
    from .var import Var
# end if

__all__ = ["Op"]


class Op(MathExpr):
    """Base class for symbolic operations."""

    def __init__(self, children: Sequence[MathExpr], shape: "Shape", dtype: Any):
        """Initialize an Op.

        Args:
            children: Operands of the operation.
            shape: Resulting shape.
            dtype: Resulting dtype.
        """
        super().__init__(shape, dtype, children=children)
    # end def __init__

    def evaluate(self, env: Mapping["Var", Value]) -> Value:
        """Evaluate the operation recursively.

        Args:
            env: Mapping from Var to Value.

        Returns:
            Value: Result of evaluating the operation.
        """
        child_values = tuple(child.evaluate(env) for child in self.children)
        return self._eval_impl(child_values)
    # end def evaluate

    @abstractmethod
    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Implementation hook for operation evaluation.

        Args:
            child_values: Values produced by the operation children.

        Returns:
            Value: Resulting runtime value.
        """
    # end def _eval_impl
# end class Op
