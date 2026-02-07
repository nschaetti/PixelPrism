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
from typing import Union
from pixelprism.math.math_expr import MathNode
from pixelprism.math.build import as_expr
from .helpers import apply_operator


__all__ = [
    "sum",
    "mean",
    "std",
    "median",
    "max",
    "min",
    "q1",
    "q3",
    "summation",
    "product"
]


def sum(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """
    Sum of a tensor.
    """
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator(
        op_name="sum",
        operands=(op1,),
        display_name=f"sum({op1.name})",
        axis=axis
    )
# end def sum


def mean(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """
    Mean of a tensor.
    """
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator(
        "mean",
        (op1,),
        f"mean({op1.name})",
        axis=axis
    )
# end def mean


def std(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """
    Sum of a tensor.
    """
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator(
        "std",
        (op1,),
        f"std({op1.name})",
        axis=axis
    )
# end def std


def median(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """Median of a tensor."""
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator("median", (op1,), f"median({op1.name})", axis=axis)
# end def median


def max(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """Tensor maximum along the given axis."""
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator("max", (op1,), f"max({op1.name})", axis=axis)
# end def max


def min(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """Tensor minimum along the given axis."""
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator("min", (op1,), f"min({op1.name})", axis=axis)
# end def min


def q1(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """First quartile (25th percentile) along an axis."""
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator("q1", (op1,), f"q1({op1.name})", axis=axis)
# end def q1


def q3(
        op1: MathNode,
        axis: MathNode | int | None = None
) -> MathNode:
    """Third quartile (75th percentile) along an axis."""
    op1 = as_expr(op1)
    axis = as_expr(axis) if axis is not None else None
    return apply_operator("q3", (op1,), f"q3({op1.name})", axis=axis)
# end def q3


def summation(
        op1: MathNode,
        lower: Union["MathNode", int],
        upper: Union["MathNode", int],
        i: str
) -> MathNode:
    """
    Sum of a tensor.
    """
    op1 = as_expr(op1)
    return apply_operator(
        "summation",
        (op1,),
        f"sum({op1.name})",
        lower=lower,
        upper=upper,
        i=i
    )
# end def sum


def product(
        op1: MathNode,
        lower: Union["MathNode", int],
        upper: Union["MathNode", int],
        i: str
) -> MathNode:
    """
    Product of a tensor over a bounded variable.
    """
    op1 = as_expr(op1)
    return apply_operator(
        "product",
        (op1,),
        f"product({op1.name})",
        lower=lower,
        upper=upper,
        i=i
    )
# end def product
