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

from typing import Any, Dict, Optional

from .. import as_expr
from ..dtype import DType
from ..operators.algorithmic import register_algorithm as _register_algorithm
from ..operators.algorithmic import has_algorithm, get_algorithm
from ..shape import Shape, ShapeLike
from ..typing import MathExpr
from .helpers import apply_operator


__all__ = [
    "register_algorithm",
    "has_algorithm",
    "get_algorithm",
    "algorithm",
]


def register_algorithm(name: str, fn):
    _register_algorithm(name, fn)
# end def register_algorithm


def algorithm(
        algorithm_name: str,
        *operands: MathExpr,
        out_shape: ShapeLike,
        out_dtype: DType = DType.R,
        params: Optional[Dict[str, Any]] = None,
) -> MathExpr:
    shape = Shape.create(out_shape)
    ops = tuple(as_expr(op) for op in operands)
    return apply_operator(
        op_name="algorithm",
        operands=ops,
        display_name=f"algorithm({algorithm_name})",
        algorithm_name=algorithm_name,
        out_shape=shape,
        out_dtype=out_dtype,
        params=params or {},
    )
# end def algorithm
