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
from pixelprism.math.math_expr import MathExpr
from pixelprism.math.build import as_expr
from .helpers import apply_operator

__all__ = [
    "sum",
    "mean",
    "std",
    "summation"
]


def sum(
        op1: MathExpr,
        axis: MathExpr | int | None = None
) -> MathExpr:
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
        op1: MathExpr,
        axis: MathExpr | int | None = None
) -> MathExpr:
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
        op1: MathExpr,
        axis: MathExpr | int | None = None
) -> MathExpr:
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


def summation(
        op1: MathExpr,
        lower: "MathExpr",
        upper: "MathExpr",
        bounded_variable: "Tensor"
) -> MathExpr:
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
        bounded_variable=bounded_variable
    )
# end def sum

