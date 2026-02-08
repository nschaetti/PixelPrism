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

from typing import Callable, Sequence, Union, Optional
import numpy as np

from ..math_base import MathNode, Variable
from ..build import as_expr
from ..shape import Shape
from ..dtype import DType
from .helpers import apply_operator


__all__ = [
    "build_tensor",
    "vector",
    "matrix",
    "full",
    "zeros",
    "ones",
    "eye",
    "diag",
    "identity",
    "concatenate",
    "hstack",
    "vstack",
    "from_function",
    "sparse_coo",
    "linspace",
    "logspace",
    "map_",
]


def _scalar_constant(value: Union[int, float, bool], dtype: DType) -> MathNode:
    if dtype is DType.R:
        data = float(value)
    elif dtype is DType.Z:
        data = int(value)
    elif dtype is DType.B:
        data = bool(value)
    elif dtype is DType.C:
        data = complex(value)
    else:
        data = value
    return as_expr(data, dtype=dtype)
# end def _scalar_constant


def build_tensor(
        elements: Sequence[Union[MathNode, int, float]],
        shape: Optional[Union[Sequence[int], Shape]] = None,
) -> MathNode:
    operands = tuple(as_expr(o) for o in elements)
    tensor_shape = Shape.create(shape) if shape is not None else None
    element_names = ", ".join(op.name for op in operands)
    display_shape = f" @ {tensor_shape.dims}" if tensor_shape is not None else ""
    return apply_operator(
        op_name="build_tensor",
        operands=operands,
        display_name=f"tensor([{element_names}]){display_shape}",
        input_shape=tensor_shape
    )
# end def build_tensor


def vector(elements: Sequence[Union[MathNode, int, float]]) -> MathNode:
    operands = tuple(as_expr(elem) for elem in elements)
    if not operands:
        raise ValueError("vector requires at least one element.")
    names = ", ".join(op.name for op in operands)
    return apply_operator(
        op_name="vector",
        operands=operands,
        display_name=f"vector([{names}])"
    )
# end def vector


def matrix(rows: Sequence[Sequence[Union[MathNode, int, float]]]) -> MathNode:
    if not rows or not rows[0]:
        raise ValueError("matrix requires at least one row and one column.")
    row_count = len(rows)
    col_count = len(rows[0])
    operands: list[MathNode] = []
    for row in rows:
        if len(row) != col_count:
            raise ValueError("matrix rows must share the same length.")
        operands.extend(as_expr(value) for value in row)
    operand_tuple = tuple(operands)
    return apply_operator(
        op_name="matrix",
        operands=operand_tuple,
        display_name=f"matrix({row_count}x{col_count})",
        rows=row_count,
        cols=col_count
    )
# end def matrix


def full(
        fill_value: Union[MathNode, int, float],
        shape: Union[Shape, Sequence[int], int]
) -> MathNode:
    operand = as_expr(fill_value)
    tensor_shape = Shape.create(shape)
    return apply_operator(
        op_name="full",
        operands=(operand,),
        display_name=f"full({operand.name}, shape={tensor_shape.dims})",
        shape=tensor_shape
    )
# end def full


def zeros(
        shape: Union[Shape, Sequence[int], int],
        dtype: DType = DType.R
) -> MathNode:
    tensor_shape = Shape.create(shape)
    zero_expr = _scalar_constant(0, dtype)
    return apply_operator(
        op_name="full",
        operands=(zero_expr,),
        display_name=f"zeros({tensor_shape.dims})",
        shape=tensor_shape
    )
# end def zeros


def ones(
        shape: Union[Shape, Sequence[int], int],
        dtype: DType = DType.R
) -> MathNode:
    tensor_shape = Shape.create(shape)
    one_expr = _scalar_constant(1, dtype)
    return apply_operator(
        op_name="full",
        operands=(one_expr,),
        display_name=f"ones({tensor_shape.dims})",
        shape=tensor_shape
    )
# end def ones


def eye(rows: int, cols: Optional[int] = None, dtype: DType = DType.R) -> MathNode:
    if rows <= 0:
        raise ValueError("eye requires rows > 0.")
    if cols is not None and cols <= 0:
        raise ValueError("eye requires cols > 0.")
    return apply_operator(
        op_name="eye",
        operands=(),
        display_name=f"eye({rows}x{cols or rows})",
        rows=rows,
        cols=cols,
        dtype=dtype
    )
# end def eye


def diag(vector_expr: Union[MathNode, int, float]) -> MathNode:
    operand = as_expr(vector_expr)
    return apply_operator(
        op_name="diag",
        operands=(operand,),
        display_name=f"diag({operand.name})"
    )
# end def diag


def identity(size: int, dtype: DType = DType.R) -> MathNode:
    if size <= 0:
        raise ValueError("identity requires size > 0.")
    return eye(size, size, dtype=dtype)
# end def identity


def concatenate(operands: Sequence[MathNode], axis: Optional[int] = 0) -> MathNode:
    exprs = tuple(as_expr(op) for op in operands)
    if not exprs:
        raise ValueError("concatenate requires at least one operand.")
    names = ", ".join(expr.name for expr in exprs)
    return apply_operator(
        op_name="concatenate",
        operands=exprs,
        display_name=f"concatenate({names})",
        axis=axis
    )
# end def concatenate


def hstack(operands: Sequence[MathNode]) -> MathNode:
    exprs = tuple(as_expr(op) for op in operands)
    if not exprs:
        raise ValueError("hstack requires at least one operand.")
    names = ", ".join(expr.name for expr in exprs)
    return apply_operator(
        op_name="hstack",
        operands=exprs,
        display_name=f"hstack({names})",
    )
# end def hstack


def vstack(operands: Sequence[MathNode]) -> MathNode:
    exprs = tuple(as_expr(op) for op in operands)
    if not exprs:
        raise ValueError("vstack requires at least one operand.")
    names = ", ".join(expr.name for expr in exprs)
    return apply_operator(
        op_name="vstack",
        operands=exprs,
        display_name=f"vstack({names})",
    )
# end def vstack


def _build_index_vars(rank: int) -> tuple[Variable, ...]:
    default_names = ("i", "j", "k", "l", "m", "n")
    vars_list: list[Variable] = []
    for idx in range(rank):
        name = default_names[idx] if idx < len(default_names) else f"idx_{idx}"
        vars_list.append(
            Variable.create(
                name=name,
                dtype=DType.Z,
                shape=Shape.scalar()
            )
        )
    return tuple(vars_list)
# end def _build_index_vars


def from_function(
        shape: Union[Shape, Sequence[int], int],
        body: Union[MathNode, Callable[..., MathNode]],
        *,
        index_vars: Optional[Sequence[Variable]] = None
) -> MathNode:
    tensor_shape = Shape.create(shape)
    rank = tensor_shape.rank
    if index_vars is None:
        resolved_vars = _build_index_vars(rank)
    else:
        if len(index_vars) != rank:
            raise ValueError(
                f"Expected {rank} index variables, got {len(index_vars)}"
            )
        resolved_vars = tuple(index_vars)
    if callable(body):
        expr = body(*resolved_vars)
    else:
        expr = body
    expr = as_expr(expr)
    return apply_operator(
        op_name="from_function",
        operands=(expr,),
        display_name=f"from_function(shape={tensor_shape.dims})",
        shape=tensor_shape,
        index_vars=tuple(var.name for var in resolved_vars)
    )
# end def from_function


def sparse_coo(
        shape: Union[Shape, Sequence[int], int],
        indices: Sequence[Sequence[int]],
        values: Sequence[Union[MathNode, int, float]]
) -> MathNode:
    tensor_shape = Shape.create(shape)
    coords = tuple(tuple(int(dim) for dim in coord) for coord in indices)
    if len(coords) != len(values):
        raise ValueError(
            f"sparse_coo expects {len(coords)} values, got {len(values)}"
        )
    rank = tensor_shape.rank
    for coord in coords:
        if len(coord) != rank:
            raise ValueError(
                f"Index {coord} does not match tensor rank {rank}."
            )
    exprs = tuple(as_expr(value) for value in values)
    return apply_operator(
        op_name="sparse_coo",
        operands=exprs,
        display_name=f"sparse_coo(shape={tensor_shape.dims}, nnz={len(coords)})",
        shape=tensor_shape,
        indices=coords
    )
# end def sparse_coo


def _constant_int(expr: MathNode, name: str) -> int:
    if not expr.is_constant():
        raise ValueError(f"{name} must be a constant integer.")
    value_array = np.asarray(expr.eval().value)
    if value_array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    value = float(value_array.item())
    if not float(value).is_integer():
        raise ValueError(f"{name} must be an integer, got {value}.")
    return int(value)
# end def _constant_int


def linspace(
        start: MathNode | int | float,
        stop: MathNode | int | float,
        num: MathNode | int
) -> MathNode:
    start_expr = as_expr(start)
    stop_expr = as_expr(stop)
    num_expr = as_expr(num)
    length = _constant_int(num_expr, "linspace num")
    if length <= 0:
        raise ValueError("linspace requires num > 0.")
    return apply_operator(
        op_name="linspace",
        operands=(),
        display_name=f"linspace({start_expr.name}, {stop_expr.name}, {length})",
        start=start_expr,
        stop=stop_expr,
        num=num_expr
    )
# end def linspace


def logspace(
        start: MathNode | int | float,
        stop: MathNode | int | float,
        num: MathNode | int,
        base: MathNode | int | float = 10
) -> MathNode:
    start_expr = as_expr(start)
    stop_expr = as_expr(stop)
    num_expr = as_expr(num)
    base_expr = as_expr(base)
    length = _constant_int(num_expr, "logspace num")
    if length <= 0:
        raise ValueError("logspace requires num > 0.")
    # end if
    return apply_operator(
        op_name="logspace",
        operands=(),
        display_name=f"logspace({start_expr.name}, {stop_expr.name}, {length})",
        start=start_expr,
        stop=stop_expr,
        num=num_expr,
        base=base_expr
    )
# end def logspace


def map_(tensor: MathNode, var_name: str, body: MathNode) -> MathNode:
    tensor_expr = as_expr(tensor)
    body_expr = as_expr(body)
    return apply_operator(
        op_name="map",
        operands=(tensor_expr,),
        display_name=f"map({tensor_expr.name}, var={var_name})",
        var_name=var_name,
        body=body_expr
    )
# end def map_
