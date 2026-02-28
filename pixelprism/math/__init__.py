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
from .typing_utils import as_expr

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

from .decorators import (
    rule,
)

from .dtype import (
    DType,
    TypeLike,
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
    SymbolicMathInvalidDimensionError
)

from .math_base import MathBase

from .math_node import MathNode

from .math_leaves import MathLeaf, Variable, Constant, SymbolicConstant, var, const

from . import constants

from .function import Function

from .mixins import ExpressionMixin, AlgebraicMixin, SimplifyRuleMixin

from .operators import (
    OperatorBase,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Log2,
    Log10,
    operator_registry,
    register_algorithm,
    has_algorithm,
    get_algorithm,
)

from .shape import (
    DimExpr,
    DimInt,
    DimLike,
    Shape,
    ShapeLike,
)

from .tensor import (
    Tensor,
    TensorLike,
    # Structure
    concatenate as t_concatenate,
    hstack as t_hstack,
    vstack as t_vstack,
    # Shapes
    scalar_shape,
    vector_shape,
    matrix_shape,
    # Creators
    tensor as t_tensor,
    scalar as t_scalar,
    zeros as t_zeros,
    zeros_like as t_zeros_like,
    ones as t_ones,
    ones_like as t_ones_like,
    full as t_full,
    normal as t_normal,
    uniform as t_uniform,
    randint as t_randint,
    poisson as t_poisson,
    bernoulli as t_bernoulli,
    # Elementwise
    pow as t_pow,
    square as t_square,
    sqrt as t_sqrt,
    cbrt as t_cbrt,
    reciprocal as t_reciprocal,
    exp as t_exp,
    exp2 as t_exp2,
    expm1 as t_expm1,
    log as t_log,
    log2 as t_log2,
    log10 as t_log10,
    log1p as t_log1p,
    sin as t_sin,
    cos as t_cos,
    tan as t_tan,
    arcsin as t_arcsin,
    arccos as t_arccos,
    arctan as t_arctan,
    sinh as t_sinh,
    cosh as t_cosh,
    tanh as t_tanh,
    arcsinh as t_arcsinh,
    arccosh as t_arccosh,
    arctanh as t_arctanh,
    deg2rad as t_deg2rad,
    rad2deg as t_rad2deg,
    # Discretization
    absolute as t_absolute,
    abs as t_abs,
    sign as t_sign,
    floor as t_floor,
    ceil as t_ceil,
    trunc as t_trunc,
    rint as t_rint,
    rounding as t_round,
    clip as t_clip,
    # Comparison
    equal as t_equal,
    not_equal as t_not_equal,
    less_equal as t_less_equal,
    less as t_less,
    greater_equal as t_greater_equal,
    greater as t_greater,
    any as t_any,
    all as t_all,
    # Linear algebra
    eye_like as t_eye_like,
    einsum as t_einsum,
    transpose as t_transpose,
    inverse as t_inverse,
    trace as t_trace,
    matmul as t_matmul,
)

from .typing import (
    ScalarLike,
    ScalarListLike,
    Index,
    ExprPattern,
    NodePattern,
    VariablePattern,
    ConstantPattern,
    AnyPattern,
    p_node,
    p_var,
    p_const,
    p_any,
    node_match,
    MathExpr,
    TensorLike,
    LeafKind,
    SimplifyOptions,
    SimplifyRule,
    SimplifyRuleType,
    OpSimplifyResult,
    OpConstruct,
    AlgebraicExpr,
    ExprLike,
    OpAssociativity
)

from .pattern import (
    P,
    patterns,
)

from .geometry import (
    Point1,
    Point2,
    Point3,
    Line2,
    Line3,
    Affine2,
    Affine3,
    Circle2,
    Ellipse2,
)


from .random import random_const_name, rand_name
from .ml import Perceptron, DecisionTreeClassifier, SVC
from .graph import (
    Graph,
    ExpressionGraph,
    graph,
    digraph,
    graph_from_edges,
    graph_from_adjacency,
    expression_to_graph,
    expression_to_adjacency,
    graph_to_dot,
)


Z = DType.Z
R = DType.R
C = DType.C
B = DType.B

CONSTANT = LeafKind.CONSTANT
VARIABLE = LeafKind.VARIABLE

tensor = t_tensor
scalar = t_scalar
zeros = t_zeros
ones = t_ones
full = t_full
normal = t_normal
uniform = t_uniform
randint = t_randint
poisson = t_poisson
bernoulli = t_bernoulli
eye_like = t_eye_like
einsum = t_einsum
transpose = t_transpose
inverse = t_inverse
trace = t_trace
matmul = t_matmul


class T:
    """Alias for Tensor creation functions."""
    tensor = staticmethod(t_tensor)
    scalar = staticmethod(t_scalar)
    ones = staticmethod(t_ones)
    zeros = staticmethod(t_zeros)
    full = staticmethod(t_full)
    normal = staticmethod(t_normal)
    uniform = staticmethod(t_uniform)
    randint = staticmethod(t_randint)
    poisson = staticmethod(t_poisson)
    bernoulli = staticmethod(t_bernoulli)

    pow = staticmethod(t_pow)
    square = staticmethod(t_square)
    sqrt = staticmethod(t_sqrt)
    cbrt = staticmethod(t_cbrt)
    reciprocal = staticmethod(t_reciprocal)
    exp = staticmethod(t_exp)
    exp2 = staticmethod(t_exp2)
    expm1 = staticmethod(t_expm1)
    log = staticmethod(t_log)
    log2 = staticmethod(t_log2)
    log10 = staticmethod(t_log10)

    sin = staticmethod(t_sin)
    cos = staticmethod(t_cos)
    tan = staticmethod(t_tan)
    arcsin = staticmethod(t_arcsin)
    arccos = staticmethod(t_arccos)
    arctan = staticmethod(t_arctan)
    sinh = staticmethod(t_sinh)
    cosh = staticmethod(t_cosh)
    tanh = staticmethod(t_tanh)
    arcsinh = staticmethod(t_arcsinh)
    arccosh = staticmethod(t_arccosh)
    arctanh = staticmethod(t_arctanh)
    deg2rad = staticmethod(t_deg2rad)
    rad2deg = staticmethod(t_rad2deg)

    absolute = staticmethod(t_absolute)
    abs = staticmethod(t_abs)
    sign = staticmethod(t_sign)
    floor = staticmethod(t_floor)
    ceil = staticmethod(t_ceil)
    trunc = staticmethod(t_trunc)
    rint = staticmethod(t_rint)
    round = staticmethod(t_round)
    clip = staticmethod(t_clip)

    hstack = staticmethod(t_hstack)
    vstack = staticmethod(t_vstack)
    concatenate = staticmethod(t_concatenate)

    eye_like = staticmethod(t_eye_like)
    einsum = staticmethod(t_einsum)
    transpose = staticmethod(t_transpose)
    inverse = staticmethod(t_inverse)
    trace = staticmethod(t_trace)
    matmul = staticmethod(t_matmul)

    equal = staticmethod(t_equal)
    not_equal = staticmethod(t_not_equal)
    less_equal = staticmethod(t_less_equal)
    less = staticmethod(t_less)
    greater_equal = staticmethod(t_greater_equal)
    greater = staticmethod(t_greater)
# end class T


class S:
    """Alias for TensorShape creation functions."""
    scalar = staticmethod(scalar_shape)
    vector = staticmethod(vector_shape)
    matrix = staticmethod(matrix_shape)
# end class S


__all__ = [
    # Alias
    "T",
    "S",
    "P",
    "patterns",
    "VARIABLE",
    "CONSTANT",

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

    # Decorators
    "rule",

    # DType
    "DType",
    "Z",
    "R",
    "C",
    "B",
    "TypeLike",
    "ScalarLike",
    "ScalarListLike",
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
    "SymbolicConstant",
    "constants",
    "var",
    "const",

    # Function
    "Function",
    "Perceptron",
    "DecisionTreeClassifier",
    "SVC",

    # Mixins
    "ExpressionMixin",
    "AlgebraicMixin",
    "SimplifyRuleMixin",

    # Operators
    "OperatorBase",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "operator_registry",
    "register_algorithm",
    "has_algorithm",
    "get_algorithm",

    # Shape
    "Shape",
    "ShapeLike",
    "DimExpr",
    "DimInt",
    "DimLike",

    # Geometry
    "Point1",
    "Point2",
    "Point3",
    "Line2",
    "Line3",
    "Affine2",
    "Affine3",
    "Circle2",
    "Ellipse2",

    # Graph
    "Graph",
    "ExpressionGraph",
    "graph",
    "digraph",
    "graph_from_edges",
    "graph_from_adjacency",
    "expression_to_graph",
    "expression_to_adjacency",
    "graph_to_dot",

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
    "tensor",
    "scalar",
    "zeros",
    "ones",
    "full",
    "normal",
    "uniform",
    "randint",
    "poisson",
    "bernoulli",
    "scalar_shape",
    "vector_shape",
    "matrix_shape",
    "t_full",
    "t_zeros",
    "t_zeros_like",
    "t_ones",
    "t_ones_like",
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
    "ExprPattern",
    "NodePattern",
    "VariablePattern",
    "ConstantPattern",
    "AnyPattern",
    "p_node",
    "p_var",
    "p_const",
    "p_any",
    "node_match",
    "ScalarLike",
    "ScalarListLike",
    "Index",
    "DimExpr",
    "MathExpr",
    "random_const_name",
    "rand_name",
    "LeafKind",
    "SimplifyOptions",
    "SimplifyRule",
    "SimplifyRuleType",
    "OpSimplifyResult",
    "OpConstruct",
    "AlgebraicExpr",
    "OpAssociativity",
]
