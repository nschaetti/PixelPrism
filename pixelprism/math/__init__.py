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

from .context import (
    Context,
    root_context,
    context,
    new_context,
    push_context,
    pop_context,
    root,
    get_value,
    set_value,
    lookup,
    create_variable,
    remove_variable,
    remove_deep,
    snapshot_context_stack,
    restore_context_stack
)

from .dtype import (
    DType,
    TypeLike,
    NumberLike,
    NumberListLike,
    ScalarLike,
    Z_DTYPE,
    R_DTYPE,
    to_numpy,
    convert_numpy,
    copy,
    create,
    promote,
    from_numpy,
)

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

from .math_exceptions import (
    SymbolicMathError,
    SymbolicMathOperatorError,
    SymbolicMathNotImplementedError,
    SymbolicMathRuntimeError,
    SymbolicMathValidationError,
    SymbolicMathLookupError,
    SymbolicMathTypeError,
    SymbolicMathShapeError,
)

from .math_base import MathBase

from .math_node import MathNode

from .math_leaves import MathLeaf, Variable, Constant

from .mixins import EvaluableMixin, DifferentiableMixin, PredicateMixin

from .operators import Operator, Add, Sub, Mul, Div, Pow, Log, Log2, Log10, operator_registry

from .shape import Dim, Dims, Shape, ShapeLike

from .tensor import (
    Tensor,
    TensorLike,
    # Structure
    t_concatenate,
    t_hstack,
    t_vstack,
    # Shapes
    ts_scalar,
    ts_vector,
    ts_matrix,
    # Creators
    t_zeros,
    t_ones,
    t_full,
    # Elementwise
    t_pow,
    t_square,
    t_sqrt,
    t_cbrt,
    t_reciprocal,
    t_exp,
    t_exp2,
    t_expm1,
    t_log,
    t_log2,
    t_log10,
    t_log1p,
    t_sin,
    t_cos,
    t_tan,
    t_arcsin,
    t_arccos,
    t_arctan,
    t_sinh,
    t_cosh,
    t_tanh,
    t_arcsinh,
    t_arccosh,
    t_arctanh,
    t_deg2rad,
    t_rad2deg,
    t_absolute,
    t_abs,
    t_sign,
    t_floor,
    t_ceil,
    t_trunc,
    t_rint,
    t_round,
    t_clip,
    t_equal,
    t_not_equal,
    t_less_equal,
    t_less,
    t_greater_equal,
    t_greater,
)

from .typing import (
    NumberLike,
    ScalarLike,
    NumberListLike,
    Index,
    DimExpr,
    MathExpr,
    TensorLike,
    TensorDim,
    TensorDims
)

from .utils import (
    var,
    const,
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

from .random import random_const_name, rand_name


Z = DType.Z
R = DType.R
C = DType.C
B = DType.B


__all__ = [
    # Build
    "as_expr",

    # Context
    "Context",
    "root_context",
    "context",
    "root",
    "set_value",
    "get_value",
    "new_context",
    "push_context",
    "remove_variable",
    "create_variable",
    "remove_deep",
    "snapshot_context_stack",
    "restore_context_stack",

    # DType
    "DType",
    "Z",
    "R",
    "C",
    "B",
    "TypeLike",
    "NumberLike",
    "ScalarLike",
    "NumberListLike",
    "Z_DTYPE",
    "R_DTYPE",
    "to_numpy",
    "convert_numpy",
    "copy",
    "create",
    "promote",
    "from_numpy",

    # Exceptions
    "SymbolicMathError",
    "SymbolicMathOperatorError",
    "SymbolicMathNotImplementedError",
    "SymbolicMathRuntimeError",
    "SymbolicMathValidationError",
    "SymbolicMathLookupError",

    # Math Expr
    "MathBase",

    # MathNode
    "MathNode",

    # MathLeaf
    "MathLeaf",
    "Variable",
    "Constant",

    # Mixins
    "EvaluableMixin",
    "DifferentiableMixin",
    "PredicateMixin",

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
    "TensorLike",
    "ts_scalar",
    "ts_vector",
    "ts_matrix",
    "t_full",
    "t_zeros",
    "t_ones",
    "t_concatenate",
    "t_hstack",
    "t_vstack",
    "t_pow",
    "t_square",
    "t_sqrt",
    "t_cbrt",
    "t_reciprocal",
    "t_exp",
    "t_exp2",
    "t_expm1",
    "t_log",
    "t_log2",
    "t_log10",
    "t_log1p",
    "t_sin",
    "t_cos",
    "t_tan",
    "t_arcsin",
    "t_arccos",
    "t_arctan",
    "t_sinh",
    "t_cosh",
    "t_tanh",
    "t_arcsinh",
    "t_arccosh",
    "t_arctanh",
    "t_deg2rad",
    "t_rad2deg",
    "t_absolute",
    "t_abs",
    "t_sign",
    "t_floor",
    "t_ceil",
    "t_trunc",
    "t_rint",
    "t_round",
    "t_clip",
    # "einsum",

    # Typing
    "NumberLike",
    "ScalarLike",
    "NumberListLike",
    "Index",
    "DimExpr",
    "MathExpr",

    # Utils
    "var",
    "random_const_name",
    "const",
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
    "ones_like"
]
