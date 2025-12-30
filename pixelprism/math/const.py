"""Symbolic constant expression."""

from __future__ import annotations

from typing import Any, Dict, Mapping, TYPE_CHECKING

from .math_expr import MathExpr
from .value import Value

if TYPE_CHECKING:
    from .var import Var
# end if

__all__ = ["Const"]


class Const(MathExpr):
    """Symbolic constant backed by an immutable Value."""

    def __init__(self, value: Value):
        """Initialize a Const.

        Args:
            value: Runtime value representing the constant.
        """
        stored = value.copy()
        super().__init__(stored.shape, stored.dtype, children=())
        self._value = stored
    # end def __init__

    def evaluate(self, env: Mapping["Var", Value]) -> Value:
        """Return a copy of the underlying constant value.

        Args:
            env: Evaluation environment (unused).

        Returns:
            Value: Deep copy of the constant.
        """
        return self._value.copy()
    # end def evaluate

    def _graph_params(self) -> Dict[str, Any]:
        """Return graph parameters describing the constant.

        Returns:
            Dict[str, Any]: Metadata indicating constant nodes.
        """
        return {"constant": True}
    # end def _graph_params
# end class Const

