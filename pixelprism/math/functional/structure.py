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
from typing import Union, List
from pixelprism.math.math_expr import MathNode, SliceExpr
from pixelprism.math.build import as_expr
from .helpers import apply_operator

__all__ = [
    "getitem"
]


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
