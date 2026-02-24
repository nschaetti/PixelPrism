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
"""
Operator package exports.
"""

from .base import Operands, OperatorBase, OperatorRegistry, operator_registry
from .boolean import (
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Not,
    Any,
    All,
    And,
    Or,
    Xor,
)
from .builders import (
    BuildTensor,
    Vector,
    Matrix,
    Full,
    Concatenate,
    HStack,
    VStack,
    Linspace,
    Logspace,
    FromFunction,
    Diag,
    SparseCOO,
    Zeros,
    Ones,
    Eye,
    Identity,
    Map
)
from .stats import (
    Normal,
    Uniform,
    RandInt,
    Poisson,
    Bernoulli,
    Covariance,
    Correlation,
    ZScore,
)
from .elementwise import (
    Add,
    Div,
    ElementwiseOperator,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log1p,
    Log2,
    Log10,
    Mul,
    Neg,
    Pow,
    Absolute,
    Abs,
    Deg2rad,
    Rad2deg,
    Reciprocal,
    Cbrt,
    Square,
    Sqrt,
    Sub,
    UnaryElementwiseOperator,
)
from .discretization import (
    Sign,
    Floor,
    Ceil,
    Trunc,
    Rint,
    Round,
    Clip,
)
from .linear_algebra import (
    LinearAlgebraParametricOperator,
    LinearAlgebraOperator,
    MatMul,
    Dot,
    Outer,
    Trace,
    Transpose,
    Det,
    Inverse,
    Norm,
    InftyNorm,
    FrobeniusNorm,
)
from .reduction import (
    ReductionOperator,
    Sum,
    Mean,
    Std,
    Median,
    Max,
    Min,
    Q1,
    Q3,
    Summation,
    Product,
)
from .structure import (
    Getitem,
    Flatten,
    Squeeze,
    Unsqueeze,
    Reshape
)
from .trigo import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Cos,
    Cosh,
    Cot,
    Csc,
    Sec,
    Sin,
    Sinh,
    Tan,
    Tanh,
)
from .activations import (
    ReLU,
    LeakyReLU,
    Sigmoid,
    Softplus,
    GELU,
)
from .conditional import (
    Where,
)
from .algorithmic import (
    AlgorithmicOperator,
    register_algorithm,
    has_algorithm,
    get_algorithm,
)
from .machine_learning import (
    MachineLearningOperator,
    SKLEARN_AVAILABLE,
    PerceptronTrain,
    PerceptronPredict,
    PerceptronDecisionBoundary,
    PerceptronCoefficients,
    PerceptronIntercept,
    PerceptronClasses,
    DecisionTreeTrain,
    DecisionTreePredict,
    DecisionTreeClasses,
    SVMTrain,
    SVMPredict,
    SVMDecisionFunction,
    SVMClasses,
)
from .graph import (
    GraphOperator,
    Degree,
    InDegree,
    OutDegree,
    Laplacian,
    IsCyclic,
    TopologicalSort,
)
from .statistical_learning import (
    StatisticalLearningOperator,
    LinearRegressionFit,
    LinearRegressionPredict,
    PolynomialFeatures,
    PolynomialRegressionFit,
    PolynomialRegressionPredict,
    MSE,
    RMSE,
    MAE,
    R2,
)


# Boolean
BOOLEAN_OPERATORS = (Eq, Ne, Lt, Le, Gt, Ge, Not, Any, All, And, Or, Xor)
ALGEBRAIC_STRICT_OPERATORS = (Add, Sub, Mul, Div, Pow)
ALGEBRAIC_OPERATORS = (Add, Sub, Mul, Div, Pow, Absolute, Abs)
BUILDERS = (
    Vector,
    Matrix,
    Full,
    Concatenate,
    HStack,
    VStack,
    Zeros,
    Ones,
    Normal,
    Uniform,
    RandInt,
    Poisson,
    Bernoulli,
    Eye,
    Identity,
    Map,
    Linspace,
    Logspace,
    Diag,
    SparseCOO,
)
STATS_OPERATORS = (Covariance, Correlation, ZScore)
DISCRETIZATION_OPERATORS = (Sign, Floor, Ceil, Trunc, Rint, Round, Clip)
ELEMENTWISE_OPERATORS = (Neg, Exp, Exp2, Expm1, Log, Log1p, Log2, Log10, Square, Cbrt, Reciprocal, Abs, Where)
TRIGO_OPERATORS = (Acos, Acosh, Asin, Asinh, Atan, Atan2, Atanh, Cos, Cosh, Cot, Csc, Sec, Sin, Sinh, Tan, Tanh)
LINEAR_ALGEBRA_OPERATORS = (MatMul, Dot, Outer, Trace, Transpose, Det, Inverse, Norm, InftyNorm, FrobeniusNorm)
REDUCTION_OPERATORS = (Sum, Mean, Std, Median, Max, Min, Q1, Q3, Summation, Product)
STRUCTURE_OPERATORS = (Getitem, Flatten, Squeeze, Unsqueeze, Reshape)
ACTIVATIONS_OPERATORS = (ReLU, LeakyReLU, Sigmoid, Softplus, GELU)
CONDITIONAL_OPERATORS = (Where,)
GRAPH_OPERATORS = (Degree, InDegree, OutDegree, Laplacian, IsCyclic, TopologicalSort)
ALGORITHMIC_OPERATORS = (AlgorithmicOperator,)
MACHINE_LEARNING_OPERATORS = (
    PerceptronTrain,
    PerceptronPredict,
    PerceptronDecisionBoundary,
    PerceptronCoefficients,
    PerceptronIntercept,
    PerceptronClasses,
    DecisionTreeTrain,
    DecisionTreePredict,
    DecisionTreeClasses,
    SVMTrain,
    SVMPredict,
    SVMDecisionFunction,
    SVMClasses,
)
STATISTICAL_LEARNING_OPERATORS = (
    LinearRegressionFit,
    LinearRegressionPredict,
    PolynomialFeatures,
    PolynomialRegressionFit,
    PolynomialRegressionPredict,
    MSE,
    RMSE,
    MAE,
    R2,
)


__all__ = [
    # List
    "BOOLEAN_OPERATORS",
    "ALGEBRAIC_STRICT_OPERATORS",
    "ALGEBRAIC_OPERATORS",
    "BUILDERS",
    "STATS_OPERATORS",
    "DISCRETIZATION_OPERATORS",
    "ELEMENTWISE_OPERATORS",
    "TRIGO_OPERATORS",
    "LINEAR_ALGEBRA_OPERATORS",
    "REDUCTION_OPERATORS",
    "STRUCTURE_OPERATORS",
    "ACTIVATIONS_OPERATORS",
    "CONDITIONAL_OPERATORS",
    "GRAPH_OPERATORS",
    "ALGORITHMIC_OPERATORS",
    "MACHINE_LEARNING_OPERATORS",
    "STATISTICAL_LEARNING_OPERATORS",
    # Base
    "Operands",
    "OperatorBase",
    "OperatorRegistry",
    "operator_registry",
    "ElementwiseOperator",
    "UnaryElementwiseOperator",
    # Builders
    "BuildTensor",
    "Vector",
    "Matrix",
    "Full",
    "Concatenate",
    "HStack",
    "VStack",
    "Zeros",
    "Ones",
    "Normal",
    "Uniform",
    "RandInt",
    "Poisson",
    "Bernoulli",
    "Covariance",
    "Correlation",
    "ZScore",
    "Eye",
    "Identity",
    "Map",
    "Linspace",
    "Logspace",
    "FromFunction",
    "Diag",
    "SparseCOO",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Softplus",
    "GELU",
    # Boolean
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "Not",
    "Any",
    "All",
    "And",
    "Or",
    "Xor",
    # Element-wise
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Exp",
    "Exp2",
    "Expm1",
    "Log",
    "Log1p",
    "Log2",
    "Log10",
    "Sqrt",
    "Square",
    "Cbrt",
    "Reciprocal",
    "Deg2rad",
    "Rad2deg",
    "Absolute",
    "Abs",
    "Where",
    # Graph
    "GraphOperator",
    "Degree",
    "InDegree",
    "OutDegree",
    "Laplacian",
    "IsCyclic",
    "TopologicalSort",
    # Algorithmic
    "AlgorithmicOperator",
    "register_algorithm",
    "has_algorithm",
    "get_algorithm",
    # Machine Learning
    "MachineLearningOperator",
    "SKLEARN_AVAILABLE",
    "PerceptronTrain",
    "PerceptronPredict",
    "PerceptronDecisionBoundary",
    "PerceptronCoefficients",
    "PerceptronIntercept",
    "PerceptronClasses",
    "DecisionTreeTrain",
    "DecisionTreePredict",
    "DecisionTreeClasses",
    "SVMTrain",
    "SVMPredict",
    "SVMDecisionFunction",
    "SVMClasses",
    # Statistical Learning
    "StatisticalLearningOperator",
    "LinearRegressionFit",
    "LinearRegressionPredict",
    "PolynomialFeatures",
    "PolynomialRegressionFit",
    "PolynomialRegressionPredict",
    "MSE",
    "RMSE",
    "MAE",
    "R2",
    # Discrete
    "Sign",
    "Floor",
    "Ceil",
    "Trunc",
    "Rint",
    "Round",
    "Clip",
    "Neg",
    # Trigo
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Atan2",
    "Sec",
    "Csc",
    "Cot",
    "Sinh",
    "Cosh",
    "Tanh",
    "Asinh",
    "Acosh",
    "Atanh",
    # Linear Algebra
    "LinearAlgebraParametricOperator",
    "LinearAlgebraOperator",
    "MatMul",
    "Dot",
    "Outer",
    "Trace",
    "Transpose",
    "Det",
    "Inverse",
    "Norm",
    "InftyNorm",
    "FrobeniusNorm",
    # Reduction
    "Sum",
    "Mean",
    "Std",
    "Median",
    "Max",
    "Min",
    "Q1",
    "Q3",
    "Summation",
    "Product",
    # Structure
    "Getitem",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Reshape"
]
