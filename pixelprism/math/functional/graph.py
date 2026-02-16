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
# Copyright (C) 2026 Pixel Prism
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

from typing import Optional

from ..build import as_expr
from ..dtype import DType
from ..typing import MathExpr
from .helpers import apply_operator


__all__ = [
    "degree",
    "in_degree",
    "out_degree",
    "laplacian",
    "is_cyclic",
    "topological_sort",
]


def degree(adjacency: MathExpr, directed: Optional[bool] = None, mode: str = "total") -> MathExpr:
    adj_expr = as_expr(adjacency)
    return apply_operator(
        op_name="degree",
        operands=(adj_expr,),
        display_name=f"degree({adj_expr.name})",
        directed=directed,
        mode=mode,
    )
# end def degree


def in_degree(adjacency: MathExpr, directed: Optional[bool] = True) -> MathExpr:
    adj_expr = as_expr(adjacency)
    return apply_operator(
        op_name="in_degree",
        operands=(adj_expr,),
        display_name=f"in_degree({adj_expr.name})",
        directed=directed,
    )
# end def in_degree


def out_degree(adjacency: MathExpr, directed: Optional[bool] = True) -> MathExpr:
    adj_expr = as_expr(adjacency)
    return apply_operator(
        op_name="out_degree",
        operands=(adj_expr,),
        display_name=f"out_degree({adj_expr.name})",
        directed=directed,
    )
# end def out_degree


def laplacian(
        adjacency: MathExpr,
        directed: Optional[bool] = None,
        normalized: bool = False,
        dtype: DType = DType.R,
) -> MathExpr:
    adj_expr = as_expr(adjacency)
    return apply_operator(
        op_name="laplacian",
        operands=(adj_expr,),
        display_name=f"laplacian({adj_expr.name})",
        directed=directed,
        normalized=normalized,
        dtype=dtype,
    )
# end def laplacian


def is_cyclic(adjacency: MathExpr, directed: Optional[bool] = None) -> MathExpr:
    adj_expr = as_expr(adjacency)
    return apply_operator(
        op_name="is_cyclic",
        operands=(adj_expr,),
        display_name=f"is_cyclic({adj_expr.name})",
        directed=directed,
    )
# end def is_cyclic


def topological_sort(adjacency: MathExpr, directed: Optional[bool] = True) -> MathExpr:
    adj_expr = as_expr(adjacency)
    return apply_operator(
        op_name="topological_sort",
        operands=(adj_expr,),
        display_name=f"topological_sort({adj_expr.name})",
        directed=directed,
    )
# end def topological_sort
