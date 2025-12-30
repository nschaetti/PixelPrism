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
"""Elementwise subtraction operation."""

# Imports
from __future__ import annotations
from typing import Tuple
from ._helpers import select_ops
from .math_expr import MathExpr
from .op import Op
from .value import Value

__all__ = ["Sub"]


class Sub(Op):
    """Elementwise subtraction."""

    def __init__(
            self,
            left: MathExpr,
            right: MathExpr
    ):
        """Initialize a Sub operation.

        Args:
            left: Left operand.
            right: Right operand.

        Raises:
            ValueError: If the operand dtypes are incompatible.
        """
        if left.dtype != right.dtype:
            raise ValueError("Sub requires matching dtypes.")
        # end if
        shape = left.shape.merge_elementwise(right.shape)
        super().__init__((left, right), shape, left.dtype)
    # end def __init__

    def _eval_impl(
            self,
            child_values: Tuple[Value, ...]
    ) -> Value:
        """Evaluate the subtraction.

        Args:
            child_values: Tuple containing left and right Values.

        Returns:
            Value: Runtime result of the subtraction.
        """
        left, right = child_values
        backend = select_ops(child_values)
        if backend is not None and hasattr(backend, "sub"):
            data = backend.sub(left.get(), right.get())
        else:
            data = left.get() - right.get()
        # end if
        return Value(data, self.shape, self.dtype, backend)
    # end def _eval_impl

# end class Sub

