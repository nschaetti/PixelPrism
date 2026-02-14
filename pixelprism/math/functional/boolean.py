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

from __future__ import annotations

from ..build import as_expr
from .helpers import apply_operator
from ..math_node import MathNode

__all__ = [
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "logical_not",
    "any",
    "all",
    "logical_and",
    "logical_or",
    "logical_xor",
]


def _comparison_display(lhs: MathNode, rhs: MathNode, symbol: str) -> str:
    return f"{lhs.name} {symbol} {rhs.name}"
# end def _comparison_display


def _binary_comparison(op_name: str, lhs, rhs, symbol: str) -> MathNode:
    lhs_expr = as_expr(lhs)
    rhs_expr = as_expr(rhs)
    return apply_operator(
        op_name,
        (lhs_expr, rhs_expr),
        _comparison_display(lhs_expr, rhs_expr, symbol),
    )
# end def _binary_comparison


def eq(lhs, rhs) -> MathNode:
    return _binary_comparison("eq", lhs, rhs, "≡")
# end def eq


def ne(lhs, rhs) -> MathNode:
    return _binary_comparison("ne", lhs, rhs, "≠")
# end def ne


def lt(lhs, rhs) -> MathNode:
    return _binary_comparison("lt", lhs, rhs, "<")
# end def lt


def le(lhs, rhs) -> MathNode:
    return _binary_comparison("le", lhs, rhs, "≤")
# end def le


def gt(lhs, rhs) -> MathNode:
    return _binary_comparison("gt", lhs, rhs, ">")
# end def gt


def ge(lhs, rhs) -> MathNode:
    return _binary_comparison("ge", lhs, rhs, "≥")
# end def ge


def logical_not(value) -> MathNode:
    operand = as_expr(value)
    return apply_operator(
        "not",
        (operand,),
        f"¬{operand.name}",
    )
# end def logical_not


def any(value) -> MathNode:
    operand = as_expr(value)
    return apply_operator(
        "any",
        (operand,),
        f"any({operand.name})",
    )
# end def any


def all(value) -> MathNode:
    operand = as_expr(value)
    return apply_operator(
        "all",
        (operand,),
        f"all({operand.name})",
    )
# end def all


def logical_and(lhs, rhs) -> MathNode:
    return _binary_comparison("and", lhs, rhs, "∧")
# end def logical_and


def logical_or(lhs, rhs) -> MathNode:
    return _binary_comparison("or", lhs, rhs, "∨")
# end def logical_or


def logical_xor(lhs, rhs) -> MathNode:
    return _binary_comparison("xor", lhs, rhs, "⊕")
# end def logical_xor
