"""Reshape operation."""

from __future__ import annotations

from typing import Dict, Tuple

from ._helpers import reshape_python, select_ops
from .math_expr import MathExpr
from .op import Op
from .shape import Shape
from .value import Value

__all__ = ["Reshape"]


class Reshape(Op):
    """Symbolic reshape operation."""

    def __init__(self, tensor: MathExpr, new_shape: Shape):
        """Initialize a Reshape operation.

        Args:
            tensor: Expression to reshape.
            new_shape: Target shape.
        """
        tensor.shape.reshape(new_shape)
        self._target = new_shape
        super().__init__((tensor,), new_shape, tensor.dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the reshape.

        Args:
            child_values: Tuple containing the Value to reshape.

        Returns:
            Value: Runtime result with the new shape.
        """
        (value,) = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "reshape"):
            data = backend.reshape(value.get(), self._target.as_tuple())
        else:
            if backend is not None:
                backend = None
            # end if
            data = reshape_python(value.get(), self._target.as_tuple())
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl

    def _graph_params(self) -> Dict[str, Tuple[int | None, ...]]:
        """Describe the target shape.

        Returns:
            Dict[str, Tuple[int | None, ...]]: Metadata containing the new shape.
        """
        return {"target": self._target.as_tuple()}
    # end def _graph_params
# end class Reshape

