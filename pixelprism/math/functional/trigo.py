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
"""
Trigonometric functional helpers.
"""

from pixelprism.math.build import as_expr
from pixelprism.math.math_expr import MathNode

from .helpers import apply_operator

__all__ = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sec",
    "csc",
    "cot",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
]


def _unary(name: str, operand: MathNode, display: str) -> MathNode:
    """
    Apply a registered unary trigonometric operator.

    Parameters
    ----------
    name : str
        Operator name registered in :class:`pixelprism.math.operators.OperatorRegistry`.
    operand : MathNode
        Operand to convert via :func:`as_expr`.
    display : str
        Display template referencing ``{op.name}``.

    Returns
    -------
    MathNode
        Expression node wrapping the operator application.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.trigo import _unary
    >>> x = tensor("theta", 0.0)
    >>> _unary("sin", x, "sin({op.name})").eval()
    array(0.)
    """
    op = as_expr(operand)
    return apply_operator(name, (op,), display.format(op=op))
# end def _unary


def _binary(name: str, op1: MathNode, op2: MathNode, display: str) -> MathNode:
    """
    Apply a registered binary trigonometric operator.

    Parameters
    ----------
    name : str
        Operator name.
    op1 : MathNode
        First operand.
    op2 : MathNode
        Second operand.
    display : str
        Template referencing ``{lhs.name}`` and ``{rhs.name}``.

    Returns
    -------
    MathNode
        Resulting operator node.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional.trigo import _binary
    >>> y = tensor("y", 1.0)
    >>> x = tensor("x", 1.0)
    >>> _binary("atan2", y, x, "atan2({lhs.name}, {rhs.name})").eval()
    array(0.78539816)
    """
    left = as_expr(op1)
    right = as_expr(op2)
    return apply_operator(name, (left, right), display.format(lhs=left, rhs=right))
# end def _binary


def sin(op: MathNode) -> MathNode:
    """
    Element-wise sine of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\sin(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.sin(tensor("theta", 0.0)).eval()
    array(0.)
    """
    return _unary("sin", op, "sin({op.name})")
# end def sin


def cos(op: MathNode) -> MathNode:
    """
    Element-wise cosine of ``op``.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.cos(tensor("theta", 0.0)).eval()
    array(1.)
    """
    return _unary("cos", op, "cos({op.name})")
# end def cos


def tan(op: MathNode) -> MathNode:
    """
    Element-wise tangent of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\tan(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.tan(tensor("theta", 0.0)).eval()
    array(0.)
    """
    return _unary("tan", op, "tan({op.name})")
# end def tan


def asin(op: MathNode) -> MathNode:
    """
    Element-wise inverse sine of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\arcsin(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.asin(tensor("x", 0.0)).eval()
    array(0.)
    """
    return _unary("asin", op, "asin({op.name})")
# end def asin


def acos(op: MathNode) -> MathNode:
    """
    Element-wise inverse cosine of ``op``.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.acos(tensor("x", 1.0)).eval()
    array(0.)
    """
    return _unary("acos", op, "acos({op.name})")
# end def acos


def atan(op: MathNode) -> MathNode:
    """
    Element-wise inverse tangent of ``op``.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.atan(tensor("x", 1.0)).eval()
    array(0.78539816)
    """
    return _unary("atan", op, "atan({op.name})")
# end def atan


def atan2(op1: MathNode, op2: MathNode) -> MathNode:
    """
    Element-wise :func:`numpy.arctan2` of ``op1`` and ``op2``.

    Parameters
    ----------
    op1 : MathNode
        Numerator operand (``y``).
    op2 : MathNode
        Denominator operand (``x``).

    Returns
    -------
    MathNode
        Expression representing :math:`\\operatorname{atan2}(op1, op2)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> y = tensor("y", 1.0)
    >>> x = tensor("x", 1.0)
    >>> trigo.atan2(y, x).eval()
    array(0.78539816)
    """
    return _binary("atan2", op1, op2, "atan2({lhs.name}, {rhs.name})")
# end def atan2


def sec(op: MathNode) -> MathNode:
    """
    Element-wise secant of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`1/\\cos(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.sec(tensor("theta", 0.0)).eval()
    array(1.)
    """
    return _unary("sec", op, "sec({op.name})")
# end def sec


def csc(op: MathNode) -> MathNode:
    """
    Element-wise cosecant of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`1/\\sin(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.csc(tensor("theta", 1.0)).eval()
    array(1.18839511)
    """
    return _unary("csc", op, "csc({op.name})")
# end def csc


def cot(op: MathNode) -> MathNode:
    """
    Element-wise cotangent of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`1/\\tan(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.cot(tensor("theta", 0.78539816339)).eval()
    array(1.)
    """
    return _unary("cot", op, "cot({op.name})")
# end def cot


def sinh(op: MathNode) -> MathNode:
    """
    Element-wise hyperbolic sine of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\sinh(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.sinh(tensor("x", 0.0)).eval()
    array(0.)
    """
    return _unary("sinh", op, "sinh({op.name})")
# end def sinh


def cosh(op: MathNode) -> MathNode:
    """
    Element-wise hyperbolic cosine of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\cosh(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.cosh(tensor("x", 0.0)).eval()
    array(1.)
    """
    return _unary("cosh", op, "cosh({op.name})")
# end def cosh


def tanh(op: MathNode) -> MathNode:
    """
    Element-wise hyperbolic tangent of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\tanh(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.tanh(tensor("x", 0.0)).eval()
    array(0.)
    """
    return _unary("tanh", op, "tanh({op.name})")
# end def tanh


def asinh(op: MathNode) -> MathNode:
    """
    Element-wise inverse hyperbolic sine of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\operatorname{asinh}(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.asinh(tensor("x", 0.0)).eval()
    array(0.)
    """
    return _unary("asinh", op, "asinh({op.name})")
# end def asinh


def acosh(op: MathNode) -> MathNode:
    """
    Element-wise inverse hyperbolic cosine of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\operatorname{acosh}(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.acosh(tensor("x", 1.5)).eval()
    array(0.96242365)
    """
    return _unary("acosh", op, "acosh({op.name})")
# end def acosh


def atanh(op: MathNode) -> MathNode:
    """
    Element-wise inverse hyperbolic tangent of ``op``.

    Parameters
    ----------
    op : MathNode
        Operand to transform.

    Returns
    -------
    MathNode
        Expression representing :math:`\\operatorname{atanh}(op)`.

    Examples
    --------
    >>> from pixelprism.math import tensor
    >>> from pixelprism.math.functional import trigo
    >>> trigo.atanh(tensor("x", 0.2)).eval()
    array(0.20273255)
    """
    return _unary("atanh", op, "atanh({op.name})")
# end def atanh
