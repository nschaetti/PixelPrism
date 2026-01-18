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
from typing import Sequence, Union, Optional

from pixelprism.math.math_expr import MathExpr
from pixelprism.math.build import as_expr
from .helpers import apply_operator

__all__ = [
    "matmul",
    "dot",
    "outer",
    "trace",
    "transpose",
    "det",
    "inverse",
]


def matmul(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Matrix-multiplication of two operands.
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "matmul",
        (op1, op2),
        f"{op1.name} @ {op2.name}"
    )
# end def matmul


def dot(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Dot product of two operands.
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "dot",
        (op1, op2),
        f"{op1.name} ⋅ {op2.name}"
    )
# end def matmul


def outer(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Outer product of two operands.
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "outer",
        (op1, op2),
        f"{op1.name} ⊗ {op2.name}"
    )
# end def outer


def trace(
        op1: MathExpr
) -> MathExpr:
    """
    Trace of a matrix.
    """
    op1 = as_expr(op1)
    return apply_operator(
        "trace",
        (op1,),
        f"tr({op1.name})"
    )
# end def trace


def transpose(
        op1: MathExpr,
        axes: Optional[Union[MathExpr, Sequence[int]]] = None
) -> MathExpr:
    """
    Transpose of a matrix.
    """
    return apply_operator(
        op_name="transpose",
        operands=(op1,),
        display_name=f"transpose({axes})" if axes is not None else "transpose()",
        axes=axes
    )
# end def transpose


def det(
        op1: MathExpr,
) -> MathExpr:
    """
    Determinant of a matrix.
    """
    return apply_operator(
        op_name="det",
        operands=(op1,),
        display_name="det()"
    )
# end def det


def inverse(
        op1: MathExpr,
) -> MathExpr:
    """
    Inverse of a matrix.
    """
    return apply_operator(
        op_name="inv",
        operands=(op1,),
        display_name="inv()"
    )
# end def inverse

