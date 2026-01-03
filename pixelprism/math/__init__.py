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
"""Unified symbolic math package with split class modules."""

from .build import as_expr
from .dtype import DType, AnyDType, NumericType, NestedListType, DataType, ScalarType
from .helpers import (
    as_sequence,
    build_from_flat,
    concat_python,
    flatten_simple,
    infer_dims_from_data,
    is_sequence_like,
    num_elements,
    ravel_index,
    reshape_python,
    select_ops,
    stack_python,
    transpose_python,
    unravel_index,
)
from .math_expr import MathExpr, MathLeaf
from .operators import Operator, Add, Sub, Mul, Div, operator_registry
from .shape import Dim, Dims, Shape, AnyShape
from .tensor import Tensor
from .utils import (
    tensor,
    scalar,
    vector,
    matrix,
    empty,
    zeros,
    ones,
    full,
    nan,
    I,
    diag,
    eye_like,
    zeros_like,
    ones_like,
)

__all__ = [
    # Build
    "as_expr",
    # DType
    "DType",
    "AnyDType",
    "NumericType",
    "ScalarType",
    "DataType",
    "NestedListType",
    # Math Expr
    "MathExpr",
    "MathLeaf",
    # Operators
    "Operator",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "operator_registry",
    # Shape
    "Shape",
    "Dim",
    "Dims",
    # Helpers
    "is_sequence_like",
    "as_sequence",
    "infer_dims_from_data",
    "num_elements",
    "flatten_simple",
    "build_from_flat",
    "reshape_python",
    "concat_python",
    "stack_python",
    "unravel_index",
    "ravel_index",
    "transpose_python",
    "select_ops",
    # Tensor
    "Tensor",
    # Utils
    "tensor",
    "scalar",
    "vector",
    "matrix",
    "empty",
    "zeros",
    "ones",
    "full",
    "nan",
    "I",
    "diag",
    "eye_like",
    "zeros_like",
    "ones_like",
]
