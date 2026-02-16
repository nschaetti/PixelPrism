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

from typing import Optional, Sequence, Union

from ..build import as_expr
from ..dtype import DType
from ..math_node import MathNode
from ..shape import Shape
from .helpers import apply_operator


ScalarLikeExpr = Union[MathNode, int, float]


__all__ = [
    "normal",
    "uniform",
    "randint",
    "poisson",
    "bernoulli",
    "cov",
    "corr",
    "zscore",
]


def normal(
        shape: Union[Shape, Sequence[int], int],
        loc: ScalarLikeExpr = 0.0,
        scale: ScalarLikeExpr = 1.0,
        dtype: DType = DType.R
) -> MathNode:
    tensor_shape = Shape.create(shape)
    loc_expr = as_expr(loc)
    scale_expr = as_expr(scale)
    return apply_operator(
        op_name="normal",
        operands=(),
        display_name=f"normal(shape={tensor_shape.dims}, loc={loc_expr.name}, scale={scale_expr.name})",
        shape=tensor_shape,
        loc=loc_expr,
        scale=scale_expr,
        dtype=dtype,
    )
# end def normal


def uniform(
        shape: Union[Shape, Sequence[int], int],
        low: ScalarLikeExpr = 0.0,
        high: ScalarLikeExpr = 1.0,
        dtype: DType = DType.R
) -> MathNode:
    tensor_shape = Shape.create(shape)
    low_expr = as_expr(low)
    high_expr = as_expr(high)
    return apply_operator(
        op_name="uniform",
        operands=(),
        display_name=f"uniform(shape={tensor_shape.dims}, low={low_expr.name}, high={high_expr.name})",
        shape=tensor_shape,
        low=low_expr,
        high=high_expr,
        dtype=dtype,
    )
# end def uniform


def randint(
        shape: Union[Shape, Sequence[int], int],
        low: ScalarLikeExpr,
        high: Optional[ScalarLikeExpr] = None,
        dtype: DType = DType.Z
) -> MathNode:
    tensor_shape = Shape.create(shape)
    low_expr = as_expr(low)
    high_expr = as_expr(high) if high is not None else None
    return apply_operator(
        op_name="randint",
        operands=(),
        display_name=f"randint(shape={tensor_shape.dims}, low={low_expr.name}, high={high_expr.name if high_expr is not None else None})",
        shape=tensor_shape,
        low=low_expr,
        high=high_expr,
        dtype=dtype,
    )
# end def randint


def poisson(
        shape: Union[Shape, Sequence[int], int],
        lam: ScalarLikeExpr = 1.0,
        dtype: DType = DType.Z
) -> MathNode:
    tensor_shape = Shape.create(shape)
    lam_expr = as_expr(lam)
    return apply_operator(
        op_name="poisson",
        operands=(),
        display_name=f"poisson(shape={tensor_shape.dims}, lam={lam_expr.name})",
        shape=tensor_shape,
        lam=lam_expr,
        dtype=dtype,
    )
# end def poisson


def bernoulli(
        shape: Union[Shape, Sequence[int], int],
        p: ScalarLikeExpr = 0.5,
        dtype: DType = DType.Z
) -> MathNode:
    tensor_shape = Shape.create(shape)
    p_expr = as_expr(p)
    return apply_operator(
        op_name="bernoulli",
        operands=(),
        display_name=f"bernoulli(shape={tensor_shape.dims}, p={p_expr.name})",
        shape=tensor_shape,
        p=p_expr,
        dtype=dtype,
    )
# end def bernoulli


def cov(
        x: MathNode,
        y: Optional[MathNode] = None,
        rowvar: bool = False,
        bias: bool = False,
        ddof: Optional[ScalarLikeExpr] = None,
        dtype: DType = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    y_expr = as_expr(y) if y is not None else None
    operands = (x_expr,) if y_expr is None else (x_expr, y_expr)
    ddof_expr = as_expr(ddof) if ddof is not None else None
    name = f"cov({x_expr.name})" if y_expr is None else f"cov({x_expr.name}, {y_expr.name})"
    return apply_operator(
        op_name="cov",
        operands=operands,
        display_name=name,
        rowvar=rowvar,
        bias=bias,
        ddof=ddof_expr,
        dtype=dtype,
    )
# end def cov


def corr(
        x: MathNode,
        y: Optional[MathNode] = None,
        rowvar: bool = False,
        dtype: DType = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    y_expr = as_expr(y) if y is not None else None
    operands = (x_expr,) if y_expr is None else (x_expr, y_expr)
    name = f"corr({x_expr.name})" if y_expr is None else f"corr({x_expr.name}, {y_expr.name})"
    return apply_operator(
        op_name="corr",
        operands=operands,
        display_name=name,
        rowvar=rowvar,
        dtype=dtype,
    )
# end def corr


def zscore(
        x: MathNode,
        axis: Optional[int] = None,
        ddof: ScalarLikeExpr = 0,
        eps: ScalarLikeExpr = 1e-8,
        dtype: Optional[DType] = None,
) -> MathNode:
    x_expr = as_expr(x)
    ddof_expr = as_expr(ddof)
    eps_expr = as_expr(eps)
    return apply_operator(
        op_name="zscore",
        operands=(x_expr,),
        display_name=f"zscore({x_expr.name})",
        axis=axis,
        ddof=ddof_expr,
        eps=eps_expr,
        dtype=dtype,
    )
# end def zscore
