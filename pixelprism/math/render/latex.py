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
Symbolic expression to LaTeX conversion helpers.

This module implements the first rendering layer which walks a
:class:`~pixelprism.math.math_expr.MathExpr` tree and produces a LaTeX math
string without evaluating or mutating the expression. The generated string can
then be passed to downstream presentation layers such as SVG rendering.
"""

# Imports
from __future__ import annotations
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple, List, Union, Optional
import numpy as np
from ..math_expr import MathNode, Constant, SliceExpr
from ..operators import Operator

__all__ = ["to_latex"]


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
}


def _escape_text(value: str) -> str:
    """
    Escape characters that carry special meaning in LaTeX.

    Parameters
    ----------
    value : str
        Source string that may contain characters such as ``_`` or ``%``.

    Returns
    -------
    str
        Safe LaTeX text with reserved characters replaced.

    Examples
    --------
    >>> _escape_text("x_i")
    'x\\_i'
    """
    result = "".join(_LATEX_ESCAPES.get(ch, ch) for ch in value)
    return result
# end def _escape_text


def _format_identifier(name: str) -> str:
    """
    Render a symbolic identifier using upright text.

    Parameters
    ----------
    name : str
        Identifier to format.

    Returns
    -------
    str
        Identifier wrapped in ``\\mathrm{}``.

    Examples
    --------
    >>> _format_identifier("weight")
    '\\\\mathrm{weight}'
    """
    return rf"\mathrm{{{_escape_text(name)}}}"
# end def _format_identifier


@dataclass(frozen=True)
class _OpRule:
    """
    Formatting rule describing precedence and rendering strategy.

    Attributes
    ----------
    precedence : int
        Numeric precedence used to determine when parentheses are required.
    formatter : Callable
        Callable that produces the operator's LaTeX string.
    """

    precedence: int
    formatter: Callable[["_LatexRenderer", MathNode, "_OpRule", Operator], str]
# end class _OpRule


class _LatexRenderer:
    """
    Internal helper that traverses :class:`MathExpr` nodes recursively.

    The renderer separates traversal logic from formatting helpers so the
    public entry point :func:`to_latex` stays lean.
    """

    _LEAF_PRECEDENCE = 100
    _FUNCTION_PRECEDENCE = 80

    def render(self, expr: MathNode) -> str:
        """
        Convert ``expr`` into a LaTeX math string.

        Parameters
        ----------
        expr : MathNode
            Expression tree root to render.

        Returns
        -------
        str
            LaTeX string for the provided expression.

        Examples
        --------
        >>> renderer = _LatexRenderer()
        >>> renderer.render(expr)  # doctest: +SKIP
        'x + y'
        """
        latex, _ = self._emit(expr)
        return latex
    # end def render

    def _emit(self, expr: MathNode) -> Tuple[str, int]:
        """
        Recursively render ``expr`` and return its precedence.

        Parameters
        ----------
        expr : MathNode
            Expression being formatted.

        Returns
        -------
        tuple[str, int]
            The LaTeX fragment and its precedence.
        """
        op = self._get_operator(expr)
        if op is None:
            return self._render_leaf(expr), self._LEAF_PRECEDENCE
        # end if

        op_name = op.name.lower()
        if op_name in _OP_RULES:
            rule = _OP_RULES[op_name]
            return rule.formatter(self, expr, rule, op), rule.precedence
        # end if

        if op_name in _FUNCTION_COMMANDS:
            latex = self._render_named_function(
                command=_FUNCTION_COMMANDS[op_name],
                operands=expr.children,
                operator=op
            )
            return latex, self._FUNCTION_PRECEDENCE
        # end if

        latex = self._render_generic_function(
            op_name,
            expr.children
        )
        return latex, self._FUNCTION_PRECEDENCE
    # end def _emit

    def _get_operator(self, expr: MathNode) -> Any:
        """
        Retrieve the operator descriptor attached to ``expr``.

        Parameters
        ----------
        expr : MathNode
            Expression whose operator should be returned.

        Returns
        -------
        Any | None
            Operator instance or ``None`` for leaves.
        """
        op = getattr(expr, "op", None)
        if op is None and hasattr(expr, "operator"):
            op = getattr(expr, "operator")
        # end if
        return op
    # end def _get_operator

    def _render_operand(
        self,
        expr: MathNode,
        parent_prec: int,
        *,
        allow_equal: bool,
    ) -> str:
        """
        Render a child expression, inserting parentheses when needed.

        Parameters
        ----------
        expr : MathNode
            Operand to render.
        parent_prec : int
            Precedence of the parent operator.
        allow_equal : bool
            Whether equal precedence is allowed without parentheses.

        Returns
        -------
        str
            Formatted operand string.
        """
        latex, prec = self._emit(expr)
        if prec < parent_prec or (not allow_equal and prec == parent_prec):
            return rf"\left({latex}\right)"
        # end if
        return latex
    # end def _render_operand

    def _render_group(self, expr: MathNode) -> str:
        """
        Render an expression without adjusting precedence.

        Parameters
        ----------
        expr : MathNode
            Expression to render.

        Returns
        -------
        str
            LaTeX fragment.
        """
        latex, _ = self._emit(expr)
        return latex
    # end def _render_group

    def _render_leaf(self, expr: MathNode) -> str:
        """
        Render a leaf node, preferring literal values when available.

        Parameters
        ----------
        expr : MathNode
            Leaf expression.

        Returns
        -------
        str
            Literal or identifier representation.
        """
        constant_literal = self._render_constant_leaf(expr)
        if constant_literal is not None:
            return constant_literal
        # end if

        literal = self._extract_scalar_name(expr)
        if literal is not None:
            return literal
        # end if

        label = getattr(expr, "name", None)
        if label:
            return _format_identifier(label)
        # end if
        return r"\mathrm{expr}"
    # end def _render_leaf

    def _render_constant_leaf(self, expr: MathNode) -> str | None:
        """
        Render the stored value for immutable leaves when feasible.
        """
        if not isinstance(expr, Constant):
            return None
        # end if

        data = self._extract_leaf_data(expr)
        if data is None:
            return None
        # end if

        array = self._array_from_data(data.value)
        if array is None:
            return None
        # end if

        shape = getattr(expr, "shape", None)
        rank = getattr(shape, "rank", array.ndim)
        if rank >= 3:
            return None
        # end if
        if rank == 0:
            return self._format_number(array.item())
        # end if
        if rank == 1:
            vector = np.atleast_1d(np.asarray(array)).reshape(-1)
            return self._format_vector(vector)
        # end if
        if rank == 2:
            matrix = np.atleast_2d(np.asarray(array))
            return self._format_matrix(matrix)
        # end if
        return None
    # end def _render_constant_leaf

    def _extract_leaf_data(self, expr: MathNode) -> Any | None:
        """
        Retrieve raw payload stored on a leaf expression.
        """
        for attr in ("value", "data", "_data"):
            if hasattr(expr, attr):
                data = getattr(expr, attr)
                if data is not None:
                    return data
                # end if
            # end if
        # end for
        return None
    # end def _extract_leaf_data

    def _extract_scalar_name(self, expr: MathNode) -> str | None:
        """
        Attempt to extract a scalar name from ``expr``.

        Parameters
        ----------
        expr : MathNode
            Expression to inspect.

        Returns
        -------
        str | None
            Literal string or ``None`` when no literal is available.
        """
        if self._get_operator(expr) is not None:
            return None
        # end if

        # Return scalar name
        name = expr.name
        if not isinstance(name, str) or not name:
            return name
        # end if

        if name.startswith("\\"):
            return name
        # end if

        latex_map = {
            # Greek letters (lowercase)
            "alpha": r"\alpha",
            "beta": r"\beta",
            "gamma": r"\gamma",
            "delta": r"\delta",
            "epsilon": r"\epsilon",
            "zeta": r"\zeta",
            "eta": r"\eta",
            "theta": r"\theta",
            "iota": r"\iota",
            "kappa": r"\kappa",
            "lambda": r"\lambda",
            "mu": r"\mu",
            "nu": r"\nu",
            "xi": r"\xi",
            "omicron": r"\omicron",
            "pi": r"\pi",
            "rho": r"\rho",
            "sigma": r"\sigma",
            "tau": r"\tau",
            "upsilon": r"\upsilon",
            "phi": r"\phi",
            "chi": r"\chi",
            "psi": r"\psi",
            "omega": r"\omega",
            # Greek letters (uppercase)
            "Alpha": r"\Alpha",
            "Beta": r"\Beta",
            "Gamma": r"\Gamma",
            "Delta": r"\Delta",
            "Epsilon": r"\Epsilon",
            "Zeta": r"\Zeta",
            "Eta": r"\Eta",
            "Theta": r"\Theta",
            "Iota": r"\Iota",
            "Kappa": r"\Kappa",
            "Lambda": r"\Lambda",
            "Mu": r"\Mu",
            "Nu": r"\Nu",
            "Xi": r"\Xi",
            "Omicron": r"\Omicron",
            "Pi": r"\Pi",
            "Rho": r"\Rho",
            "Sigma": r"\Sigma",
            "Tau": r"\Tau",
            "Upsilon": r"\Upsilon",
            "Phi": r"\Phi",
            "Chi": r"\Chi",
            "Psi": r"\Psi",
            "Omega": r"\Omega",
            # Common mathematical symbols
            "infty": r"\infty",
            "infinity": r"\infty",
            "inf": r"\infty",
            "partial": r"\partial",
            "nabla": r"\nabla",
            "sum": r"\sum",
            "prod": r"\prod",
            "int": r"\int",
            "sqrt": r"\sqrt",
        }
        return latex_map.get(name, name)
    # end def _extract_scalar_literal

    def _coerce_scalar(self, expr: MathNode) -> numbers.Number | None:
        """
        Convert an expression's stored data into a scalar.

        Parameters
        ----------
        expr : MathNode
            Leaf expression containing potential data.

        Returns
        -------
        numbers.Number | None
            Scalar value or ``None`` if conversion fails.
        """
        for attr in ("value", "data", "_data"):
            if hasattr(expr, attr):
                candidate = getattr(expr, attr)
                as_scalar = self._scalar_from_data(candidate)
                if as_scalar is not None:
                    return as_scalar
                # end if
            # end if
        # end for
        return None
    # end def _coerce_scalar

    def _scalar_from_data(self, data: Any) -> numbers.Number | None:
        """
        Extract a scalar number from ``data``.

        Parameters
        ----------
        data : Any
            Arbitrary payload.

        Returns
        -------
        numbers.Number | None
            Scalar result when ``data`` represents a 0-D tensor.
        """
        if isinstance(data, numbers.Number):
            return data
        # end if

        try:
            array = np.asarray(data)
        except Exception:
            return None
        # end try

        if array.ndim == 0:
            return array.item()
        # end if
        return None
    # end def _scalar_from_data

    def _array_from_data(self, data: Any) -> np.ndarray | None:
        """
        Convert stored payload into a NumPy array when possible.
        """
        if isinstance(data, np.ndarray):
            return data
        # end if
        try:
            return np.asarray(data)
        except Exception:
            return None
        # end try
    # end def _array_from_data

    def _format_number(self, value: numbers.Number) -> str:
        """
        Convert a Python or NumPy scalar into a LaTeX number literal.

        Parameters
        ----------
        value : numbers.Number
            Scalar to convert.

        Returns
        -------
        str
            LaTeX literal for the number.
        """
        if isinstance(value, numbers.Integral):
            return str(int(value))
        # end if

        if isinstance(value, numbers.Real):
            real = float(value)
            if math.isnan(real):
                return r"\mathrm{nan}"
            # end if
            if math.isinf(real):
                return r"\infty" if real > 0 else r"-\infty"
            # end if
            return f"{real:.6g}"
        # end if

        if isinstance(value, numbers.Complex):
            real = self._format_number(value.real)  # type: ignore[arg-type]
            imag = self._format_number(abs(value.imag))  # type: ignore[arg-type]
            sign = "-" if value.imag < 0 else "+"
            return f"{real} {sign} {imag}i"
        # end if

        return _escape_text(str(value))
    # end def _format_number

    def _format_vector(self, data: np.ndarray) -> str:
        """
        Format a rank-1 tensor as a column vector.
        """
        entries = [self._format_number(item) for item in data.tolist()]
        body = r" \\ ".join(entries)
        return rf"\begin{{bmatrix}}{body}\end{{bmatrix}}"
    # end def _format_vector

    def _format_matrix(self, data: np.ndarray) -> str:
        """
        Format a rank-2 tensor as a matrix literal.
        """
        rows = []
        for row in data.tolist():
            row_entries = [self._format_number(item) for item in row]
            rows.append(" & ".join(row_entries))
        # end for
        body = r" \\ ".join(rows)
        return rf"\begin{{bmatrix}}{body}\end{{bmatrix}}"
    # end def _format_matrix

    def _render_named_function(
            self,
            command: str,
            operands: Sequence[MathNode],
            operator: Operator
    ) -> str:
        """
        Render a known math function such as ``sin`` or ``sqrt``.

        Parameters
        ----------
        command : str
            LaTeX command to emit.
        operands : Sequence[MathNode]
            Operands to render as arguments.

        Returns
        -------
        str
            Function call in LaTeX syntax.
        """
        if command == r"\sqrt":
            if len(operands) != 1:
                raise ValueError("sqrt expects a single operand.")
            # end if
            inner = self._render_group(operands[0])
            return rf"\sqrt{{{inner}}}"
        # end if

        if command == r"\exp":
            if len(operands) != 1:
                raise ValueError("exp expects a single operand.")
            # end if
            inner = self._render_group(operands[0])
            return rf"e^{{{inner}}}"
        # end if

        if command == r"\sqrt[3]":
            if len(operands) != 1:
                raise ValueError("sqrt[3] expects a single operand.")
            # end if
            inner = self._render_group(operands[0])
            return rf"\sqrt[3]{{{inner}}}"
        # end if

        args = ", ".join(self._render_group(op) for op in operands)
        if len(operands) == 1:
            return rf"{command}\left({args}\right)"
        # end if
        return rf"{command}\left({args}\right)"
    # end def _render_named_function

    def _render_generic_function(
        self,
        name: str,
        operands: Sequence[MathNode],
    ) -> str:
        """
        Render an operator using ``\\operatorname{}``.

        Parameters
        ----------
        name : str
            Operator name.
        operands : Sequence[MathNode]
            Child expressions.

        Returns
        -------
        str
            LaTeX function call.
        """
        op_name = _escape_text(name)
        args = ", ".join(self._render_group(op) for op in operands)
        return rf"\operatorname{{{op_name}}}\left({args}\right)"
    # end def _render_generic_function
# end class _LatexRenderer


def _format_add(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format addition expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Addition node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    terms = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return " + ".join(terms)
# end def _format_add


def _format_sub(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format subtraction expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Subtraction node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    if len(expr.children) != 2:
        raise ValueError("sub expects exactly two operands.")
    # end if
    left = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=True)
    right = renderer._render_operand(expr.children[1], rule.precedence, allow_equal=False)
    return f"{left} - {right}"
# end def _format_sub


def _format_eq(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format equality comparisons with the \\equiv symbol.
    """
    if len(expr.children) != 2:
        raise ValueError("eq expects exactly two operands.")
    # end if
    left = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=True)
    right = renderer._render_operand(expr.children[1], rule.precedence, allow_equal=True)
    return rf"{left} \equiv {right}"
# end def _format_eq


def _format_neg(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format negation expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Negation node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    if len(expr.children) != 1:
        raise ValueError("neg expects a single operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return f"-{operand}"
# end def _format_neg


def _format_mul(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format multiplication expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Multiplication node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    factors = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r" \cdot ".join(factors)
# end def _format_mul


def _format_outer(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 2:
        raise ValueError("outer expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r" \otimes ".join(operands)
# end def _format_outer


def _format_mean(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("mean expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\operatorname{mean}(" + str(operands[0]) + r")"
# end def _format_mean


def _format_sum(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("sum expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\operatorname{sum}(" + str(operands[0]) + r")"
# end def _format_sum

def _format_summation(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("summation expects exactly one operand.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    bounded_name = expr.op.bounded_var
    lower = renderer._render_operand(expr.op.lower, rule.precedence, allow_equal=True)
    upper = renderer._render_operand(expr.op.upper, rule.precedence, allow_equal=True)
    return r"\sum_{" + bounded_name + "=" + lower + "}^{" + upper + "}{" + operands[0] + "}"
# end def _format_sum


def _format_product(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator):
     """Format product expressions."""
     if len(expr.children) != 1:
         raise ValueError("product expects exactly two operand.")
     # end if
     operands = [
         renderer._render_operand(child, rule.precedence, allow_equal=True)
         for child in expr.children
     ]
     bounded_name = expr.op.bounded_var
     lower = renderer._render_operand(expr.op.lower, rule.precedence, allow_equal=True)
     upper = renderer._render_operand(expr.op.upper, rule.precedence, allow_equal=True)
     return r"\prod_{" + bounded_name + "=" + lower + "}^{" + upper + "}{" + operands[0] + "}"
# end def _format_product


def _format_std(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("std expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\sigma(" + str(operands[0]) + r")"
# end def _format_std


def _format_q1(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Q1"""
    if len(expr.children) != 1:
        raise ValueError("q1 expects exactly two operands.")
    # end if
    inner = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=True)
    return rf"\operatorname{{q_1}}({{{inner}}})"
# end def _format_q1


def _format_q3(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Q3"""
    if len(expr.children) != 1:
        raise ValueError("q3 expects exactly two operands.")
    # end if
    inner = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=True)
    return rf"\operatorname{{q_3}}({{{inner}}})"
# end def _format_q3


def _format_div(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format division expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Division node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    if len(expr.children) != 2:
        raise ValueError("div expects exactly two operands.")
    # end if
    numerator = renderer._render_group(expr.children[0])
    denominator = renderer._render_group(expr.children[1])
    return rf"\frac{{{numerator}}}{{{denominator}}}"
# end def _format_div


def _format_pow(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format exponentiation expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Power node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    if len(expr.children) != 2:
        raise ValueError("pow expects exactly two operands.")
    # end if
    base = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    exponent = renderer._render_group(expr.children[1])
    return rf"{base}^{{{exponent}}}"
# end def _format_pow


def _format_square(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("square expects exactly 1 operand.")
    # end if
    base = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"{base}^2"
# end def _format_pow


def _format_exp2(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format 2-base exponential expressions."""
    if len(expr.children) != 1:
        raise ValueError("exp2 expects exactly one operand.")
    # end if
    base = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"2^{{{base}}}"
# end def _format_exp2

def _format_log2(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format 2-base logarithm expressions."""
    if len(expr.children) != 1:
        raise ValueError("log2 expects exactly one operand.")
    # end if
    inner = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    base = 2
    return rf"\log_{{{base}}}({{{inner}}})"
# end def _format_log2

def _format_log10(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format 10-base logarithm expressions."""
    if len(expr.children) != 1:
        raise ValueError("log10 expects exactly one operand.")
    # end if
    inner = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    base = 10
    return rf"\log_{{{base}}}({{{inner}}})"
# end def _format_log10


def _format_reciprocal(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format reciprocal expressions."""
    if len(expr.children) != 1:
        raise ValueError("reciprocal expects exactly one operand.")
    # end if
    base = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"\frac{{1}}{{{base}}}"
# end def _format_reciprocal


def _format_abs(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format absolute value expressions."""
    if len(expr.children) != 1:
        raise ValueError("abs expects exactly one operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"\left|{{{operand}}} \right|"
# end def _format_abs


#
# Discretization operators
#

def _format_sign(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format sign expressions."""
    if len(expr.children) != 1:
        raise ValueError("sign expects exactly one operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"\mathrm{{sgn}}({{{operand}}})"
# end def _format_sign


def _format_floor(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format floor expressions."""
    if len(expr.children) != 1:
        raise ValueError("floor expects exactly one operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"\lfloor{{{operand}}} \rfloor"
# end def _format_floor


def _format_ceil(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format ceiling expressions."""
    if len(expr.children) != 1:
        raise ValueError("ceil expects exactly one operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"\lceil{{{operand}}} \rceil"
# end def _format_ceil


def _format_round(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format rounding expressions."""
    if len(expr.children) != 1:
        raise ValueError("round expects exactly 1 operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)

    decimals = op.get_parameter('decimals')
    if decimals is not None and isinstance(decimals, MathNode):
        decimals = renderer._render_operand(op.get_parameter('decimals'), rule.precedence, allow_equal=False)
    elif isinstance(decimals, int):
        decimals = str(decimals)
    else:
        decimals = 0
    # end if

    return rf"\operatorname{{round}}({{{operand}}}, {{{decimals}}})"
# end def _format_round


def _format_clip(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format clipping expressions."""
    if len(expr.children) != 1:
        raise ValueError("clip expects exactly 1 operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)

    lower = op.get_parameter('lower')
    if lower is not None and isinstance(lower, MathNode):
        lower = renderer._render_operand(op.get_parameter('lower'), rule.precedence, allow_equal=False)
    elif isinstance(lower, int):
        lower = str(lower)
    else:
        lower = "-\infty"
    # end if

    upper = op.get_parameter('upper')
    if upper is not None and isinstance(upper, MathNode):
        upper = renderer._render_operand(op.get_parameter('upper'), rule.precedence, allow_equal=False)
    elif isinstance(upper, int):
        upper = str(upper)
    else:
        upper = "\infty"
    # end if

    return rf"\operatorname{{clip}}({{{operand}}}, {{{lower}}}, {{{upper}}})"
# end def _format_clip


#
# Linear algebra
#


def _format_transpose(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format transpose expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Transpose node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    if len(expr.children) != 1:
        raise ValueError("transpose expects a single operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    return rf"{operand}^{{\mathsf{{T}}}}"
# end def _format_transpose


def _format_matmul(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    Format matrix multiplication expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathNode
        Matrix multiplication node.
    rule : _OpRule
        Formatting rule used for precedence.

    Returns
    -------
    str
        Formatted LaTeX fragment.
    """
    if len(expr.children) < 2:
        raise ValueError("matmul expects at least two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"".join(operands)
# end def _format_matmul


def _format_dot(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """
    """
    if len(expr.children) != 2:
        raise ValueError("dot expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r" \cdot ".join(operands)
# end def _format_dot


#
# Structure
#


def _format_getitem(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    """Format getitem expressions."""
    if len(expr.children) != 1:
        raise ValueError("getitem expects exactly one operand.")
    # end if
    operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    indices: List[Union[SliceExpr, int]] = op.get_parameter('indices')
    indices_str = ",".join([str(ind) for ind in indices])
    if indices_str[:2] == "0:":
        indices_str = indices_str[1:]
    # end if
    return rf"{{{operand}}}_{{{indices_str}}}"
# end def _format_getitem


def _format_structure_unary(
        renderer: _LatexRenderer,
        expr: MathNode,
        rule: _OpRule, *,
        op: Operator,
        name: str,
        num_operands: int = 1,
) -> str:
    if not op.is_variadic and len(expr.children) != num_operands:
        raise ValueError(f"{name} expects exactly {num_operands} operands.")
    # end if
    if op.arity <= 0:
        operand = ""
    elif op.arity == 1:
        operand = renderer._render_operand(expr.children[0], rule.precedence, allow_equal=False)
    elif op.arity >= 2:
        operand = "(" + ",".join([
            renderer._render_operand(c, rule.precedence, allow_equal=False)
            for c in expr.children
        ]) + ")"
    # end if
    axes_suffix = _format_axes_suffix(op.get_parameter('axes')) or _format_axis_suffix(op.get_parameter('axis'))
    command = rf"\operatorname{{{name}}}"
    if axes_suffix:
        command = rf"{command}"
    # end if
    if axes_suffix:
        return rf"{command}({operand};\, {axes_suffix})"
    else:
        return rf"{command}({{{operand}}})"
    # end if
# end def _format_structure_unary


def _format_axis_suffix(axis: Optional[int]) -> Optional[str]:
    if axis is None:
        return None
    else:
        return rf"\mathrm{{axis={axis}}}"
    # end if
# end def _format_axis_suffis


def _format_axes_suffix(axes: Optional[Sequence[int]]) -> str:
    if axes is None:
        return ""
    # end if
    if len(axes) == 0:
        return ""
    # end if
    formatted = ", ".join(str(axis) for axis in axes)
    return rf"{formatted}"
# end def _format_axes_suffix


def _format_squeeze(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    return _format_structure_unary(renderer, expr, rule, op=op, name="squeeze")
# end def _format_squeeze


def _format_unsqueeze(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    return _format_structure_unary(renderer, expr, rule, op=op, name="unsqueeze")
# end def _format_unsqueeze


def _format_concat(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    return _format_structure_unary(renderer, expr, rule, op=op, name="concat", num_operands=-1)
# end def _format_concat


def _format_hstack(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    if len(expr.children) <= 1:
        raise ValueError("hstack expects at least two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"[" + r"\;".join(operands) + r"]"
# end def _format_hstack


def _format_vstack(renderer: _LatexRenderer, expr: MathNode, rule: _OpRule, op: Operator) -> str:
    if len(expr.children) <= 1:
        raise ValueError("vstack expects at least two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\begin{bmatrix}" + r" \\ ".join(operands) + r"\end{bmatrix}"
# end def _format_vstack


# Base
_ADD_RULE = _OpRule(precedence=10, formatter=_format_add)
_SUB_RULE = _OpRule(precedence=10, formatter=_format_sub)
_EQ_RULE = _OpRule(precedence=5, formatter=_format_eq)
_NEG_RULE = _OpRule(precedence=30, formatter=_format_neg)
_MUL_RULE = _OpRule(precedence=20, formatter=_format_mul)
_DIV_RULE = _OpRule(precedence=20, formatter=_format_div)
_POW_RULE = _OpRule(precedence=40, formatter=_format_pow)
_RECIPROCAL_RULE = _OpRule(precedence=40, formatter=_format_reciprocal)
_ABS_RULE = _OpRule(precedence=40, formatter=_format_abs)
_SQUARE_RULE = _OpRule(precedence=40, formatter=_format_square)
_EXP2_RULE = _OpRule(precedence=40, formatter=_format_exp2)
_LOG2_RULE = _OpRule(precedence=40, formatter=_format_log2)
_LOG10_RULE = _OpRule(precedence=40, formatter=_format_log10)

# Discretization
_SIGN_RULE = _OpRule(precedence=40, formatter=_format_sign)
_FLOOR_RULE = _OpRule(precedence=40, formatter=_format_floor)
_CEIL_RULE = _OpRule(precedence=40, formatter=_format_ceil)
_ROUND_RULE = _OpRule(precedence=40, formatter=_format_round)
_CLIP_RULE = _OpRule(precedence=40, formatter=_format_clip)

# Linear algebra
_TRANSPOSE_RULE = _OpRule(precedence=50, formatter=_format_transpose)
_MATMUL_RULE = _OpRule(precedence=20, formatter=_format_matmul)
_DOT_RULE = _OpRule(precedence=20, formatter=_format_dot)
_OUTER_RULE = _OpRule(precedence=20, formatter=_format_outer)

# Reduction
_SUM_RULE = _OpRule(precedence=100, formatter=_format_sum)
_MEAN_RULE = _OpRule(precedence=100, formatter=_format_mean)
_STD_RULE = _OpRule(precedence=100, formatter=_format_std)
_Q1_RULE = _OpRule(precedence=100, formatter=_format_q1)
_Q3_RULE = _OpRule(precedence=100, formatter=_format_q3)
_SUMMATION_RULE = _OpRule(precedence=100, formatter=_format_summation)
_PRODUCTS_RULE = _OpRule(precedence=100, formatter=_format_product)

# Structure
_GETITEM_RULE = _OpRule(precedence=50, formatter=_format_getitem)
_SQUEEZE_RULE = _OpRule(precedence=50, formatter=_format_squeeze)
_UNSQUEEZE_RULE = _OpRule(precedence=50, formatter=_format_unsqueeze)
_CONCAT_RULE = _OpRule(precedence=50, formatter=_format_concat)
_HSTACK_RULE = _OpRule(precedence=50, formatter=_format_hstack)
_VSTACK_RULE = _OpRule(precedence=50, formatter=_format_vstack)


_OP_RULES: Dict[str, _OpRule] = {
    "add": _ADD_RULE,
    "sub": _SUB_RULE,
    "subtract": _SUB_RULE,
    "eq": _EQ_RULE,
    "neg": _NEG_RULE,
    "negative": _NEG_RULE,
    "mul": _MUL_RULE,
    "multiply": _MUL_RULE,
    "div": _DIV_RULE,
    "divide": _DIV_RULE,
    "pow": _POW_RULE,
    "power": _POW_RULE,
    "reciprocal": _RECIPROCAL_RULE,
    "abs": _ABS_RULE,
    "absolute": _ABS_RULE,
    "square": _SQUARE_RULE,
    "exp2": _EXP2_RULE,
    "log2": _LOG2_RULE,
    "log10": _LOG10_RULE,
    # Discretization
    "sign": _SIGN_RULE,
    "floor": _FLOOR_RULE,
    "ceil": _CEIL_RULE,
    "round": _ROUND_RULE,
    "clip": _CLIP_RULE,
    # "product": _MUL_RULE,
    "matmul": _MATMUL_RULE,
    "matrix_multiply": _MATMUL_RULE,
    "dot": _DOT_RULE,
    "outer": _OUTER_RULE,
    "transpose": _TRANSPOSE_RULE,
    # Reduction
    "mean": _MEAN_RULE,
    "sum": _SUM_RULE,
    "std": _STD_RULE,
    "q1": _Q1_RULE,
    "q3": _Q3_RULE,
    "summation": _SUMMATION_RULE,
    "product": _PRODUCTS_RULE,
    # Structure
    "getitem": _GETITEM_RULE,
    "squeeze": _SQUEEZE_RULE,
    "unsqueeze": _UNSQUEEZE_RULE,
    "concatenate": _CONCAT_RULE,
    "hstack": _HSTACK_RULE,
    "vstack": _VSTACK_RULE,
}

_FUNCTION_COMMANDS: Dict[str, str] = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "exp": r"\exp",
    "log": r"\log",
    "sqrt": r"\sqrt",
    "cbrt": r"\sqrt[3]"
}


def to_latex(expr: MathNode) -> str:
    """
    Convert a :class:`MathExpr` into LaTeX math code.

    Parameters
    ----------
    expr : MathNode
        Expression tree to convert. The expression is treated as immutable
        input; it will not be evaluated or mutated.

    Returns
    -------
    str
        LaTeX math code suitable for downstream rendering layers.

    Examples
    --------
    >>> from pixelprism.math.render.latex import to_latex
    >>> latex = to_latex(expr)  # doctest: +SKIP
    >>> print(latex)  # doctest: +SKIP
    x + y
    """
    renderer = _LatexRenderer()
    return renderer.render(expr)
# end def to_latex
