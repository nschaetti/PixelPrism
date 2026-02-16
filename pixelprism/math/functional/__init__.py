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


from .activations import (
    relu,
    leaky_relu,
    sigmoid,
    softplus,
    gelu
)

from .boolean import (
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    logical_not,
    any,
    all,
    logical_and,
    logical_or,
    logical_xor,
)

from .builders import (
    build_tensor,
    vector,
    matrix,
    full,
    zeros,
    ones,
    eye,
    diag,
    identity,
    concatenate,
    linspace,
    logspace,
    map_,
    hstack,
    vstack,
    from_function,
    sparse_coo
)
from .stats import (
    normal,
    uniform,
    randint,
    poisson,
    bernoulli,
    cov,
    corr,
    zscore,
)
from .statistical_learning import (
    linear_regression_fit,
    linear_regression_predict,
    polynomial_features,
    polynomial_regression_fit,
    polynomial_regression_predict,
    mse,
    rmse,
    mae,
    r2,
)
from .graph import (
    degree,
    in_degree,
    out_degree,
    laplacian,
    is_cyclic,
    topological_sort,
)
from .algorithmic import (
    register_algorithm,
    has_algorithm,
    get_algorithm,
    algorithm,
)
from .machine_learning import (
    fit,
    predict,
    decision_boundary,
    coefficients,
    intercept,
    classes,
    tree_fit,
    tree_predict,
    tree_classes,
    svm_fit,
    svm_predict,
    svm_decision_function,
    svm_classes,
)

from .calculus import (
    diff,
    nabla,
)

from .conditional import (
    where,
    if_
)

from .discretization import (
    sign,
    floor,
    ceil,
    trunc,
    rint,
    round,
    clip,
)

from .elementwise import (
    add,
    sub,
    mul,
    div,
    pow,
    log,
    log10,
    log2,
    log1p,
    exp,
    exp2,
    expm1,
    sqrt,
    square,
    cbrt,
    reciprocal,
    deg2rad,
    rad2deg,
    absolute,
    abs,
    neg,
)

from .activations import (
    relu,
    leaky_relu,
    sigmoid,
    softplus,
    gelu,
)
from .conditional import (
    where,
    if_,
)

from .linear_algebra import (
    matmul,
    dot,
    outer,
    trace,
    transpose,
    det,
    inverse,
    norm,
    infty_norm,
    frobenius_norm,
)

from .reduction import (
    sum,
    mean,
    std,
    median,
    max,
    min,
    q1,
    q3,
    summation,
    product
)

from .structure import (
    getitem,
    flatten,
    squeeze,
    unsqueeze,
    reshape
)

from .trigo import (
    sin,
    cos,
    acos,
    acosh,
    atan,
    cot,
    atan2,
    csc,
    sec,
    tan,
    asin,
    cosh,
    sinh,
    tanh,
    asinh,
    atanh
)


__all__ = [
    # Activation
    "relu",
    "leaky_relu",
    "gelu",
    "softplus",
    "sigmoid",

    # Boolean
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

    # Builders
    "build_tensor",
    "vector",
    "matrix",
    "full",
    "zeros",
    "ones",
    "normal",
    "uniform",
    "randint",
    "poisson",
    "bernoulli",
    "eye",
    "diag",
    "identity",
    "concatenate",
    "linspace",
    "logspace",
    "map_",
    "hstack",
    "vstack",
    "from_function",
    "sparse_coo",
    "cov",
    "corr",
    "zscore",

    # Statistical Learning
    "linear_regression_fit",
    "linear_regression_predict",
    "polynomial_features",
    "polynomial_regression_fit",
    "polynomial_regression_predict",
    "mse",
    "rmse",
    "mae",
    "r2",

    # Graph
    "degree",
    "in_degree",
    "out_degree",
    "laplacian",
    "is_cyclic",
    "topological_sort",

    # Algorithmic
    "register_algorithm",
    "has_algorithm",
    "get_algorithm",
    "algorithm",

    # Machine Learning
    "fit",
    "predict",
    "decision_boundary",
    "coefficients",
    "intercept",
    "classes",
    "tree_fit",
    "tree_predict",
    "tree_classes",
    "svm_fit",
    "svm_predict",
    "svm_decision_function",
    "svm_classes",

    # Activations
    "relu",
    "leaky_relu",
    "sigmoid",
    "softplus",
    "gelu",

    # Calculus
    "diff",
    "nabla",

    # Conditional
    "where",
    "if_",

    # Element wise
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log2",
    "log1p",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    "deg2rad",
    "rad2deg",
    "absolute",
    "abs",
    "neg",

    # Conditional
    "where",
    "if_",

    # Discretization
    "sign",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "clip",
    # Trigo
    "sin",
    "cos",
    "acos",
    "acosh",
    "atan",
    "cot",
    "atan2",
    "csc",
    "sec",
    "tan",
    "asin",
    "cosh",
    "sinh",
    "tanh",
    "asinh",
    "atanh",

    # Linear Algebra
    "matmul",
    "dot",
    "outer",
    "trace",
    "transpose",
    "det",
    "inverse",
    "norm",
    "infty_norm",
    "frobenius_norm",

    # Reduction
    "sum",
    "mean",
    "std",
    "median",
    "max",
    "min",
    "q1",
    "q3",
    "summation",
    "product",

    # Structure
    "getitem",
    "flatten",
    "squeeze",
    "unsqueeze",
    "reshape",

    # Conditional
    "where",
]
