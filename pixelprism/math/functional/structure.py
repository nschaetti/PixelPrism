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

# Imports
from typing import Union, List, Sequence, Optional

from ..math_node import MathNode
from ..math_slice import SliceExpr
from ..build import as_expr
from .helpers import apply_operator


__all__ = [
    "getitem",
    "flatten",
    "squeeze",
    "unsqueeze",
    "reshape"
]


def reshape(op1: MathNode, shape: Sequence[int]) -> MathNode:
    """Reshape a tensor."""
    op1 = as_expr(op1)
    return apply_operator(
        op_name="reshape",
        operands=(op1,),
        display_name=f"reshape({op1.name})",
        shape=tuple(shape)
    )
# end def reshape


def getitem(op1: MathNode, indices: List[Union[SliceExpr, int]]) -> MathNode:
    """
    Sum of a tensor.
    """
    op1 = as_expr(op1)
    return apply_operator(
        op_name="getitem",
        operands=(op1,),
        display_name=f"getitem({op1.name})",
        indices=indices
    )
# end def sum


def flatten(op1: MathNode) -> MathNode:
    """Flatten a tensor."""
    op1 = as_expr(op1)
    return apply_operator(
        op_name="flatten",
        operands=(op1,),
        display_name=f"flatten({op1.name})"
    )
# end def flatten


def squeeze(op1: MathNode, axes: Optional[Sequence[int]] = None) -> MathNode:
    """Remove size-1 axes from a tensor."""
    op1 = as_expr(op1)
    axes_param = tuple(axes) if axes is not None else None
    return apply_operator(
        op_name="squeeze",
        operands=(op1,),
        display_name=f"squeeze({op1.name})",
        axes=axes_param
    )
# end def squeeze


def unsqueeze(op1: MathNode, axes: Sequence[int]) -> MathNode:
    """Insert size-1 axes at the requested positions."""
    op1 = as_expr(op1)
    return apply_operator(
        op_name="unsqueeze",
        operands=(op1,),
        display_name=f"unsqueeze({op1.name})",
        axes=tuple(axes)
    )
# end def unsqueeze
