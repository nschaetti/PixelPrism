"""Concatenation operation."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from ._helpers import concat_python, select_ops
from .math_expr import MathExpr
from .op import Op
from .shape import Shape
from .value import Value

__all__ = ["Concat"]


class Concat(Op):
    """Concatenate tensors along an axis."""

    def __init__(self, tensors: Sequence[MathExpr], axis: int = 0):
        """Initialize a Concat operation.

        Args:
            tensors: Expressions to concatenate.
            axis: Axis along which to concatenate.

        Raises:
            ValueError: If there are insufficient tensors or incompatible dtypes.
        """
        if len(tensors) < 2:
            raise ValueError("Concat requires at least two tensors.")
        # end if
        dtype = tensors[0].dtype
        shape = tensors[0].shape
        axis_norm = Shape._normalize_axis(axis, shape.rank)
        for tensor in tensors[1:]:
            if tensor.dtype != dtype:
                raise ValueError("Concat requires matching dtypes for all inputs.")
            # end if
            shape = shape.concat_result(tensor.shape, axis_norm)
        # end for
        self._axis = axis_norm
        super().__init__(tuple(tensors), shape, dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the concatenation.

        Args:
            child_values: Runtime values to concatenate.

        Returns:
            Value: Concatenated runtime value.
        """
        backend = select_ops(child_values)
        data_items = [value.get() for value in child_values]
        if backend is not None:
            if hasattr(backend, "concat"):
                data = backend.concat(data_items, self._axis)
            elif hasattr(backend, "concatenate"):
                data = backend.concatenate(data_items, self._axis)
            else:
                backend = None
                data = concat_python(data_items, self._axis)
            # end if
        else:
            data = concat_python(data_items, self._axis)
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl

    def _graph_params(self) -> Dict[str, int]:
        """Describe the concatenation axis for graph visualization.

        Returns:
            Dict[str, int]: Metadata containing the axis index.
        """
        return {"axis": self._axis}
    # end def _graph_params
# end class Concat

