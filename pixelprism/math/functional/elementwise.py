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

from typing import Union, Sequence

from .helpers import apply_operator
from ..math_node import MathNode
from ..build import as_expr
from ..typing import ExprLike, MathExpr


__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    "deg2rad",
    "rad2deg",
    "absolute",
    "abs",
    "neg",
]


def add(
        *operands: ExprLike,
) -> MathExpr:
    """
    Element-wise addition of two operands.

    Parameters
    ----------
    operands : Sequence[ExprLike]

    Returns
    -------
    MathNode
        Expression node representing the addition.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import add
    >>> a = pm.const("a", 2.0)
    >>> b = pm.const("b", 3.0)
    >>> add(a, b).eval()
    array(5.)

    Args:
        operands:
    """
    operands = [as_expr(o) for o in operands]
    display_name = f" + ".join(o.name for o in operands)
    return apply_operator(
        op_name="add",
        operands=operands,
        display_name=display_name
    )
# end def add


def sub(
        *operands: ExprLike,
) -> MathExpr:
    """
    Element-wise subtraction of two operands.

    Parameters
    ----------
    op1 : ExprLike
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : ExprLike
        Right operand. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathNode
        Expression node representing the subtraction.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import sub
    >>> a = pm.const("a", 5.0)
    >>> b = pm.const("b", 3.0)
    >>> sub(a, b).eval()
    array(2.)
    """
    operands = [as_expr(o) for o in operands]
    display_name = f" - ".join(o.name for o in operands)
    return apply_operator(
        op_name="sub",
        operands=operands,
        display_name=display_name
    )
# end def sub


def mul(
        *operands: ExprLike,
) -> MathExpr:
    """
    Element-wise multiplication of two operands.

    Parameters
    ----------
    op1 : ExprLike
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : ExprLike
        Right operand. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathNode
        Expression node representing the multiplication.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import mul
    >>> a = pm.const("a", 2.0)
    >>> b = pm.const("b", 4.0)
    >>> mul(a, b).eval()
    array(8.)
    """
    operands = [as_expr(o) for o in operands]
    display_name = f" * ".join(o.name for o in operands)
    return apply_operator(
        op_name="mul",
        operands=operands,
        display_name=display_name
    )
# end def mul


def div(
        op1: ExprLike,
        op2: ExprLike
) -> MathExpr:
    """
    Element-wise division of two operands.

    Parameters
    ----------
    op1 : ExprLike
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : ExprLike
        Right operand. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathExpr
        Expression node representing the division.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import div
    >>> a = pm.const("a", 8.0)
    >>> b = pm.const("b", 2.0)
    >>> div(a, b).eval()
    array(4.)
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        op_name="div",
        operands=[op1, op2],
        display_name=f"{op1.name} / {op2.name}"
    )
# end def div


def pow(
        op1: ExprLike,
        op2: ExprLike
) -> MathExpr:
    """
    Element-wise exponentiation of two operands.

    Parameters
    ----------
    op1 : ExprLike
        Base operand.
    op2 : ExprLike
        Exponent operand.

    Returns
    -------
    MathExpr
        Expression node representing the exponentiation.
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "pow",
        [op1, op2],
        f"{op1.name} ** {op2.name}"
    )
# end def pow


def exp(op: ExprLike) -> MathExpr:
    """
    Element-wise exponential of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "exp",
        (op,),
        f"exp({op.name})"
    )
# end def exp


def exp2(op: ExprLike) -> MathExpr:
    """
    Element-wise base-2 exponential of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "exp2",
        (op,),
        f"exp2({op.name})"
    )
# end def exp2


def expm1(op: ExprLike) -> MathExpr:
    """
    Element-wise exp(x) - 1 of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "expm1",
        (op,),
        f"expm1({op.name})"
    )
# end def expm1


def log(op: ExprLike) -> MathExpr:
    """
    Element-wise natural logarithm of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "log",
        (op,),
        f"log({op.name})"
    )
# end def log


def log1p(op: ExprLike) -> MathExpr:
    """
    Element-wise log(1 + x) of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "log1p",
        (op,),
        f"log1p({op.name})"
    )
# end def log1p


def log2(op: ExprLike) -> MathExpr:
    """
    Element-wise base-2 logarithm of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "log2",
        (op,),
        f"log2({op.name})"
    )
# end def log2


def log10(op: ExprLike) -> MathExpr:
    """
    Element-wise base-10 logarithm of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "log10",
        (op,),
        f"log10({op.name})"
    )
# end def log10


def sqrt(op: ExprLike) -> MathExpr:
    """
    Element-wise square root of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "sqrt",
        (op,),
        f"sqrt({op.name})"
    )
# end def sqrt


def square(op: ExprLike) -> MathExpr:
    """
    Element-wise square of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "square",
        (op,),
        f"square({op.name})"
    )
# end def square


def cbrt(op: ExprLike) -> MathExpr:
    """
    Element-wise cubic root of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "cbrt",
        (op,),
        f"cbrt({op.name})"
    )
# end def cbrt


def reciprocal(op: ExprLike) -> MathExpr:
    """
    Element-wise reciprocal of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "reciprocal",
        (op,),
        f"reciprocal({op.name})"
    )
# end def reciprocal


def deg2rad(op: ExprLike) -> MathExpr:
    """
    Convert degrees to radians element-wise.
    """
    op = as_expr(op)
    return apply_operator(
        "deg2rad",
        (op,),
        f"deg2rad({op.name})"
    )
# end def deg2rad


def rad2deg(op: ExprLike) -> MathExpr:
    """
    Convert radians to degrees element-wise.
    """
    op = as_expr(op)
    return apply_operator(
        "rad2deg",
        (op,),
        f"rad2deg({op.name})"
    )
# end def rad2deg


def absolute(op: ExprLike) -> MathExpr:
    """
    Element-wise absolute value of an operand.
    """
    op = as_expr(op)
    return apply_operator(
        "absolute",
        (op,),
        f"|{op.name}|"
    )
# end def absolute


def abs(op: ExprLike) -> MathExpr:
    """
    Element-wise absolute value alias.
    """
    op = as_expr(op)
    return apply_operator(
        "abs",
        (op,),
        f"abs({op.name})"
    )
# end def abs


def neg(op: ExprLike) -> MathExpr:
    """
    Element-wise negation of a single operand.

    Parameters
    ----------
    op : MathNode
        Operand to negate. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathNode
        Expression node representing the negation.

    Raises
    ------
    TypeError
        If the operator registry does not provide a unary negation operator.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> import pixelprism.math as pm
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import neg
    >>> x = pm.const("x", 3.0)
    >>> neg(x).eval()
    array(-3.)
    >>> neg(2.5).eval()
    array(-2.5)
    """
    op = as_expr(op)
    return apply_operator(
        "neg",
        (op,),
        f"-{op.name}"
    )
# end def neg
