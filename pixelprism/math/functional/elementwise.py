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


from pixelprism.math.math_expr import MathExpr
from pixelprism.math.operators import operator_registry
from pixelprism.math.build import as_expr


__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "neg",
]


Operands = tuple[MathExpr, ...]


def _apply_operator(
        op_name: str,
        operands: Operands,
        display_name: str
) -> MathExpr:
    """
    Build a MathExpr by applying a registered operator to operands.

    Parameters
    ----------
    op_name : str
        Name of the operator registered in :class:`OperatorRegistry`.
    operands : tuple[MathExpr, ...]
        Operands to apply the operator to.
    display_name : str
        Human-readable name assigned to the resulting expression.

    Returns
    -------
    MathExpr
        A new operator node wrapping the operands.

    Raises
    ------
    KeyError
        If `op_name` is not registered.
    TypeError
        If the number of operands does not match the operator arity.
    ValueError
        If operand shapes are incompatible for the operator.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.operators import operator_registry
    >>> a = tensor("a", 1.0)
    >>> b = tensor("b", 2.0)
    >>> expr = _apply_operator("add", (a, b), "a + b")
    >>> expr.eval()
    array(3.)
    """
    op = operator_registry.get(op_name)

    if not op.check_arity(operands):
        raise TypeError(
            f"Operator {op.name}({op.arity}) expected {op.arity} operands, "
            f"got {len(operands)}"
        )
    # end if

    if not op.check_shapes(operands):
        shapes = ", ".join(str(o.shape) for o in operands)
        raise TypeError(
            f"Incompatible shapes for operator {op.name}: {shapes}"
        )
    # end if

    return MathExpr(
        name=display_name,
        op=op,
        children=operands,
        dtype=op.infer_dtype(operands),
        shape=op.infer_shape(operands),
    )
# end def _apply_operator


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
    return _apply_operator(
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
    return _apply_operator(
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
    return _apply_operator(
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
    return _apply_operator(
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
    return _apply_operator(
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
    return _apply_operator(
        "exp",
        (op,),
        f"exp({op.name})"
    )
# end def exp


def log(op: MathExpr) -> MathExpr:
    """
    Element-wise natural logarithm of an operand.
    """
    op = as_expr(op)
    return _apply_operator(
        "log",
        (op,),
        f"log({op.name})"
    )
# end def log


def log2(op: MathExpr) -> MathExpr:
    """
    Element-wise base-2 logarithm of an operand.
    """
    op = as_expr(op)
    return _apply_operator(
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
    return _apply_operator(
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
    return _apply_operator(
        "sqrt",
        (op,),
        f"sqrt({op.name})"
    )
# end def sqrt


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
    return _apply_operator(
        "neg",
        (op,),
        f"-{op.name}"
    )
# end def neg
