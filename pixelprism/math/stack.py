# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2025 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""Stack operation."""

# Imports
from __future__ import annotations
from typing import Dict, Sequence, Tuple
from ._helpers import select_ops, stack_python
from .math_expr import MathExpr
from .op import Op
from .shape import Shape
from .value import Value


__all__ = ["Stack"]


class Stack(Op):
    """Stack tensors along a new axis."""

    def __init__(self, tensors: Sequence[MathExpr], axis: int = 0):
        """Initialize a Stack operation.

        Args:
            tensors: Expressions to stack.
            axis: Axis index where the new dimension is inserted.

        Raises:
            ValueError: If no tensors are provided or dtypes mismatch.
        """
        if not tensors:
            raise ValueError("Stack requires at least one tensor.")
        # end if
        dtype = tensors[0].dtype
        for tensor in tensors[1:]:
            if tensor.dtype != dtype:
                raise ValueError("Stack requires matching dtypes for all inputs.")
            # end if
        # end for
        shape = Shape.stack_result(tuple(t.shape for t in tensors), axis)
        self._axis = Shape._normalize_axis(axis, tensors[0].shape.rank, allow_new_axis=True)
        super().__init__(tuple(tensors), shape, dtype)
    # end def __init__

    def _eval_impl(self, child_values: Tuple[Value, ...]) -> Value:
        """Evaluate the stacking.

        Args:
            child_values: Runtime values to stack.

        Returns:
            Value: Stacked runtime value.
        """
        backend = select_ops(child_values)
        data_items = [value.get() for value in child_values]
        if backend is not None and hasattr(backend, "stack"):
            data = backend.stack(data_items, self._axis)
        else:
            if backend is not None:
                backend = None
            # end if
            data = stack_python(data_items, self._axis)
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl

    def _graph_params(self) -> Dict[str, int]:
        """Describe the stacking axis.

        Returns:
            Dict[str, int]: Metadata containing the axis index.
        """
        return {"axis": self._axis}
    # end def _graph_params
# end class Stack

