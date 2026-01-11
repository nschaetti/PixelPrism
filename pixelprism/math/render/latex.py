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
from typing import Any, Callable, Dict, Sequence, Tuple
import numpy as np
from ..math_expr import MathExpr, Constant

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
    formatter: Callable[["_LatexRenderer", MathExpr, "_OpRule"], str]
# end class _OpRule


class _LatexRenderer:
    """
    Internal helper that traverses :class:`MathExpr` nodes recursively.

    The renderer separates traversal logic from formatting helpers so the
    public entry point :func:`to_latex` stays lean.
    """

    _LEAF_PRECEDENCE = 100
    _FUNCTION_PRECEDENCE = 80

    def render(self, expr: MathExpr) -> str:
        """
        Convert ``expr`` into a LaTeX math string.

        Parameters
        ----------
        expr : MathExpr
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

    def _emit(self, expr: MathExpr) -> Tuple[str, int]:
        """
        Recursively render ``expr`` and return its precedence.

        Parameters
        ----------
        expr : MathExpr
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
            return rule.formatter(self, expr, rule), rule.precedence
        # end if

        if op_name in _FUNCTION_COMMANDS:
            latex = self._render_named_function(_FUNCTION_COMMANDS[op_name], expr.children)
            return latex, self._FUNCTION_PRECEDENCE
        # end if

        latex = self._render_generic_function(op_name, expr.children)
        return latex, self._FUNCTION_PRECEDENCE
    # end def _emit

    def _get_operator(self, expr: MathExpr) -> Any:
        """
        Retrieve the operator descriptor attached to ``expr``.

        Parameters
        ----------
        expr : MathExpr
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
        expr: MathExpr,
        parent_prec: int,
        *,
        allow_equal: bool,
    ) -> str:
        """
        Render a child expression, inserting parentheses when needed.

        Parameters
        ----------
        expr : MathExpr
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

    def _render_group(self, expr: MathExpr) -> str:
        """
        Render an expression without adjusting precedence.

        Parameters
        ----------
        expr : MathExpr
            Expression to render.

        Returns
        -------
        str
            LaTeX fragment.
        """
        latex, _ = self._emit(expr)
        return latex
    # end def _render_group

    def _render_leaf(self, expr: MathExpr) -> str:
        """
        Render a leaf node, preferring literal values when available.

        Parameters
        ----------
        expr : MathExpr
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

    def _render_constant_leaf(self, expr: MathExpr) -> str | None:
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

    def _extract_leaf_data(self, expr: MathExpr) -> Any | None:
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

    def _extract_scalar_name(self, expr: MathExpr) -> str | None:
        """
        Attempt to extract a scalar name from ``expr``.

        Parameters
        ----------
        expr : MathExpr
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

    def _coerce_scalar(self, expr: MathExpr) -> numbers.Number | None:
        """
        Convert an expression's stored data into a scalar.

        Parameters
        ----------
        expr : MathExpr
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
        operands: Sequence[MathExpr],
    ) -> str:
        """
        Render a known math function such as ``sin`` or ``sqrt``.

        Parameters
        ----------
        command : str
            LaTeX command to emit.
        operands : Sequence[MathExpr]
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

        args = ", ".join(self._render_group(op) for op in operands)
        if len(operands) == 1:
            return rf"{command}\left({args}\right)"
        # end if
        return rf"{command}\left({args}\right)"
    # end def _render_named_function

    def _render_generic_function(
        self,
        name: str,
        operands: Sequence[MathExpr],
    ) -> str:
        """
        Render an operator using ``\\operatorname{}``.

        Parameters
        ----------
        name : str
            Operator name.
        operands : Sequence[MathExpr]
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


def _format_add(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format addition expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


def _format_sub(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format subtraction expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


def _format_neg(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format negation expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


def _format_mul(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format multiplication expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


def _format_matmul(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format matrix multiplication expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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
    return r" \times ".join(operands)
# end def _format_matmul


def _format_dot(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    """
    if len(expr.children) != 2:
        raise ValueError("dot expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\cdot".join(operands)
# end def _format_dot


def _format_outer(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    """
    if len(expr.children) != 2:
        raise ValueError("outer expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\otimes".join(operands)
# end def _format_outer


def _format_mean(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
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


def _format_sum(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("sum expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    return r"\sum" + str(operands[0])
# end def _format_sum

def _format_summation(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    """
    if len(expr.children) != 1:
        raise ValueError("sum expects exactly two operands.")
    # end if
    operands = [
        renderer._render_operand(child, rule.precedence, allow_equal=True)
        for child in expr.children
    ]
    bounded_name = expr.op.bounded_variable.name
    lower = renderer._render_operand(expr.op.lower, rule.precedence, allow_equal=True)
    upper = renderer._render_operand(expr.op.upper, rule.precedence, allow_equal=True)
    return r"\sum_{" + bounded_name + "=" + lower + "}^{" + upper + "}{" + operands[0] + "}"
 # end def _format_sum


def _format_std(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
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


def _format_div(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format division expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


def _format_pow(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format exponentiation expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


def _format_transpose(renderer: _LatexRenderer, expr: MathExpr, rule: _OpRule) -> str:
    """
    Format transpose expressions.

    Parameters
    ----------
    renderer : _LatexRenderer
        Renderer instance driving the traversal.
    expr : MathExpr
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


_ADD_RULE = _OpRule(precedence=10, formatter=_format_add)
_SUB_RULE = _OpRule(precedence=10, formatter=_format_sub)
_NEG_RULE = _OpRule(precedence=30, formatter=_format_neg)
_MUL_RULE = _OpRule(precedence=20, formatter=_format_mul)
_MATMUL_RULE = _OpRule(precedence=20, formatter=_format_matmul)
_DOT_RULE = _OpRule(precedence=20, formatter=_format_dot)
_OUTER_RULE = _OpRule(precedence=20, formatter=_format_outer)
_DIV_RULE = _OpRule(precedence=20, formatter=_format_div)
_POW_RULE = _OpRule(precedence=40, formatter=_format_pow)
_TRANSPOSE_RULE = _OpRule(precedence=50, formatter=_format_transpose)

_SUM_RULE = _OpRule(precedence=100, formatter=_format_sum)
_MEAN_RULE = _OpRule(precedence=100, formatter=_format_mean)
_STD_RULE = _OpRule(precedence=100, formatter=_format_std)
_SUMMATION_RULE = _OpRule(precedence=100, formatter=_format_summation)


_OP_RULES: Dict[str, _OpRule] = {
    "add": _ADD_RULE,
    "sub": _SUB_RULE,
    "subtract": _SUB_RULE,
    "neg": _NEG_RULE,
    "negative": _NEG_RULE,
    "mul": _MUL_RULE,
    "multiply": _MUL_RULE,
    "product": _MUL_RULE,
    "matmul": _MATMUL_RULE,
    "matrix_multiply": _MATMUL_RULE,
    "dot": _DOT_RULE,
    "outer": _OUTER_RULE,
    "div": _DIV_RULE,
    "divide": _DIV_RULE,
    "pow": _POW_RULE,
    "power": _POW_RULE,
    "transpose": _TRANSPOSE_RULE,
    # Reduction
    "mean": _MEAN_RULE,
    "sum": _SUM_RULE,
    "std": _STD_RULE,
    "summation": _SUMMATION_RULE,
}

_FUNCTION_COMMANDS: Dict[str, str] = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "exp": r"\exp",
    "log": r"\log",
    "sqrt": r"\sqrt",
}


def to_latex(expr: MathExpr) -> str:
    """
    Convert a :class:`MathExpr` into LaTeX math code.

    Parameters
    ----------
    expr : MathExpr
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
