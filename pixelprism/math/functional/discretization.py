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
from ..build import as_expr
from .helpers import apply_operator
from ..math_node import MathNode


__all__ = [
    "sign",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "clip",
]


def sign(op: MathNode) -> MathNode:
    """Element-wise sign."""
    op = as_expr(op)
    return apply_operator("sign", (op,), f"sign({op.name})")
# end def sign


def floor(op: MathNode) -> MathNode:
    """Element-wise floor."""
    op = as_expr(op)
    return apply_operator("floor", (op,), f"floor({op.name})")
# end def floor


def ceil(op: MathNode) -> MathNode:
    """Element-wise ceiling."""
    op = as_expr(op)
    return apply_operator("ceil", (op,), f"ceil({op.name})")
# end def ceil


def trunc(op: MathNode) -> MathNode:
    """Element-wise truncation."""
    op = as_expr(op)
    return apply_operator("trunc", (op,), f"trunc({op.name})")
# end def trunc


def rint(op: MathNode) -> MathNode:
    """Element-wise rounding to the nearest integer."""
    op = as_expr(op)
    return apply_operator("rint", (op,), f"rint({op.name})")
# end def rint


def round(op: MathNode, decimals: int = 0) -> MathNode:
    """Element-wise rounding with configurable decimals."""
    op = as_expr(op)
    return apply_operator("round", (op,), f"round({op.name})", decimals=decimals)
# end def round


def clip(
        op: MathNode,
        min_value: MathNode | float | int | None = None,
        max_value: MathNode | float | int | None = None
) -> MathNode:
    """Element-wise clipping with optional bounds."""
    if min_value is None and max_value is None:
        raise ValueError("clip requires at least one of min_value or max_value.")
    op = as_expr(op)

    def _normalize(bound):
        return as_expr(bound) if bound is not None else None
    # end def _normalize

    return apply_operator(
        "clip",
        (op,),
        f"clip({op.name})",
        min_value=_normalize(min_value),
        max_value=_normalize(max_value)
    )
# end def clip
