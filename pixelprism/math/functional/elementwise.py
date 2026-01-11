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
from pixelprism.math.functional.helpers import apply_operator
from pixelprism.math.math_expr import MathExpr
from pixelprism.math.build import as_expr


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
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Element-wise addition of two operands.

    Parameters
    ----------
    op1 : MathExpr
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : MathExpr
        Right operand. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathExpr
        Expression node representing the addition.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import add
    >>> a = tensor("a", 2.0)
    >>> b = tensor("b", 3.0)
    >>> add(a, b).eval()
    array(5.)
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "add",
        (op1, op2),
        f"{op1.name} + {op2.name}"
    )
# end def add


def sub(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Element-wise subtraction of two operands.

    Parameters
    ----------
    op1 : MathExpr
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : MathExpr
        Right operand. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathExpr
        Expression node representing the subtraction.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import sub
    >>> a = tensor("a", 5.0)
    >>> b = tensor("b", 3.0)
    >>> sub(a, b).eval()
    array(2.)
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "sub",
        (op1, op2),
        f"{op1.name} - {op2.name}"
    )
# end def sub


def mul(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Element-wise multiplication of two operands.

    Parameters
    ----------
    op1 : MathExpr
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : MathExpr
        Right operand. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathExpr
        Expression node representing the multiplication.

    Raises
    ------
    TypeError
        If operand counts are invalid.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import mul
    >>> a = tensor("a", 2.0)
    >>> b = tensor("b", 4.0)
    >>> mul(a, b).eval()
    array(8.)
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "mul",
        (op1, op2),
        f"{op1.name} * {op2.name}"
    )
# end def mul


def div(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Element-wise division of two operands.

    Parameters
    ----------
    op1 : MathExpr
        Left operand. Non-expressions are converted via :func:`as_expr`.
    op2 : MathExpr
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
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import div
    >>> a = tensor("a", 8.0)
    >>> b = tensor("b", 2.0)
    >>> div(a, b).eval()
    array(4.)
    """
    op1 = as_expr(op1)
    op2 = as_expr(op2)
    return apply_operator(
        "div",
        (op1, op2),
        f"{op1.name} / {op2.name}"
    )
# end def div


def pow(
        op1: MathExpr,
        op2: MathExpr
) -> MathExpr:
    """
    Element-wise exponentiation of two operands.

    Parameters
    ----------
    op1 : MathExpr
        Base operand.
    op2 : MathExpr
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
        (op1, op2),
        f"{op1.name} ** {op2.name}"
    )
# end def pow


def exp(op: MathExpr) -> MathExpr:
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


def exp2(op: MathExpr) -> MathExpr:
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


def expm1(op: MathExpr) -> MathExpr:
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


def log(op: MathExpr) -> MathExpr:
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


def log1p(op: MathExpr) -> MathExpr:
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


def log2(op: MathExpr) -> MathExpr:
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


def log10(op: MathExpr) -> MathExpr:
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


def sqrt(op: MathExpr) -> MathExpr:
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


def square(op: MathExpr) -> MathExpr:
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


def cbrt(op: MathExpr) -> MathExpr:
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


def reciprocal(op: MathExpr) -> MathExpr:
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


def deg2rad(op: MathExpr) -> MathExpr:
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


def rad2deg(op: MathExpr) -> MathExpr:
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


def absolute(op: MathExpr) -> MathExpr:
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


def abs(op: MathExpr) -> MathExpr:
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


def neg(op: MathExpr) -> MathExpr:
    """
    Element-wise negation of a single operand.

    Parameters
    ----------
    op : MathExpr
        Operand to negate. Non-expressions are converted via :func:`as_expr`.

    Returns
    -------
    MathExpr
        Expression node representing the negation.

    Raises
    ------
    TypeError
        If the operator registry does not provide a unary negation operator.
    ValueError
        If operand shapes are incompatible.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.elementwise import neg
    >>> x = tensor("x", 3.0)
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
