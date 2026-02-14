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

from ..math_node import MathNode
from ..build import as_expr
from .helpers import apply_operator

__all__ = [
    "where",
    "if_",
]


def where(cond: MathNode, x: MathNode, y: MathNode) -> MathNode:
    """Elementwise conditional selection."""
    cond_expr = as_expr(cond)
    x_expr = as_expr(x)
    y_expr = as_expr(y)
    return apply_operator(
        op_name="where",
        operands=(cond_expr, x_expr, y_expr),
        display_name=f"where({cond_expr.name}, {x_expr.name}, {y_expr.name})",
    )
# end def where


def if_(cond: MathNode, then_expr: MathNode, else_expr: MathNode) -> MathNode:
    cond_expr = as_expr(cond)
    then_expr = as_expr(then_expr)
    else_expr = as_expr(else_expr)
    return apply_operator(
        op_name="if",
        operands=(then_expr, else_expr),
        display_name=f"if({cond_expr.name})",
        cond=cond_expr,
    )
# end def if_
