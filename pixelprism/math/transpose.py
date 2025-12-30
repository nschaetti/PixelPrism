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
"""Transpose operation."""

# Imports
from __future__ import annotations
from typing import Dict, Sequence, Tuple
from ._helpers import select_ops, transpose_python
from .math_expr import MathExpr
from .op import Op
from .value import Value


__all__ = ["Transpose"]


class Transpose(Op):
    """Axis permutation operation."""

    def __init__(
            self,
            tensor: MathExpr,
            permutation: Sequence[int]
    ):
        """Initialize a Transpose operation.

        Args:
            tensor: Expression to transpose.
            permutation: Axis permutation to apply.
        """
        target_shape = tensor.shape.transpose(permutation)
        self._permutation = tuple(permutation)
        super().__init__((tensor,), target_shape, tensor.dtype)
    # end def __init__

    def _eval_impl(
            self,
            child_values: Tuple[Value, ...]
    ) -> Value:
        """Evaluate the transpose.

        Args:
            child_values: Tuple containing the Value to transpose.

        Returns:
            Value: Runtime result after permutation.
        """
        (value,) = child_values
        backend = select_ops(child_values)
        if backend is not None:
            if hasattr(backend, "transpose"):
                data = backend.transpose(value.get(), self._permutation)
            elif hasattr(backend, "permute"):
                data = backend.permute(value.get(), self._permutation)
            else:
                backend = None
                data = transpose_python(value.get(), self._permutation)
            # end if
        else:
            data = transpose_python(value.get(), self._permutation)
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl

    def _graph_params(self) -> Dict[str, Tuple[int, ...]]:
        """Describe the axis permutation.

        Returns:
            Dict[str, Tuple[int, ...]]: Metadata describing the permutation.
        """
        return {"perm": self._permutation}
    # end def _graph_params

# end class Transpose

