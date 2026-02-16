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
"""Statistical learning operators (regression and feature maps)."""

from __future__ import annotations

from abc import ABC
from itertools import combinations, combinations_with_replacement
from typing import Optional, Sequence, Union

import numpy as np

from ..build import as_expr
from ..dtype import DType, to_numpy
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import Tensor
from ..typing import MathExpr
from .base import Operands, OperatorBase, ParametricOperator, operator_registry


ScalarParameter = Union[MathExpr, int, float]


__all__ = [
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
]


def _ensure_scalar_parameter(value: ScalarParameter, name: str) -> MathExpr:
    expr = as_expr(value)
    if expr.rank != 0:
        raise ValueError(f"{name} must be a scalar expression.")
    # end if
    return expr
# end def _ensure_scalar_parameter


def _eval_scalar_parameter(expr: MathExpr, name: str) -> float:
    value = np.asarray(expr.eval().value)
    if value.shape != ():
        raise ValueError(f"{name} must evaluate to a scalar.")
    # end if
    return float(value.item())
# end def _eval_scalar_parameter


def _eval_int_parameter(expr: MathExpr, name: str) -> int:
    value = _eval_scalar_parameter(expr, name)
    if not float(value).is_integer():
        raise ValueError(f"{name} must evaluate to an integer.")
    # end if
    return int(value)
# end def _eval_int_parameter


def _flatten_targets(y: np.ndarray, name: str) -> np.ndarray:
    if y.ndim == 1:
        return y
    # end if
    if y.ndim == 2 and y.shape[1] == 1:
        return y.reshape(-1)
    # end if
    raise ValueError(f"{name} expects y with shape (n,) or (n, 1), got {y.shape}")
# end def _flatten_targets


def _poly_design(x: np.ndarray, degree: int, include_bias: bool) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError(f"polynomial operators expect rank-1 x, got shape={x.shape}")
    # end if
    start = 0 if include_bias else 1
    powers = np.arange(start, degree + 1, dtype=np.int64)
    return np.stack([np.power(x, p) for p in powers], axis=1)
# end def _poly_design


def _poly_design_nd(
        x: np.ndarray,
        degree: int,
        include_bias: bool,
        interaction_only: bool = False,
) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    # end if
    if x.ndim != 2:
        raise ValueError(f"polynomial operators expect rank-1 or rank-2 x, got shape={x.shape}")
    # end if
    if degree < 1:
        raise ValueError("polynomial operators require degree >= 1.")
    # end if

    n_samples, n_features = x.shape
    columns = []

    if include_bias:
        columns.append(np.ones(n_samples, dtype=np.float64))
    # end if

    combo_fn = combinations if interaction_only else combinations_with_replacement
    for current_degree in range(1, degree + 1):
        for idx_tuple in combo_fn(range(n_features), current_degree):
            columns.append(np.prod(x[:, idx_tuple], axis=1))
        # end for
    # end for

    return np.stack(columns, axis=1)
# end def _poly_design_nd


def _n_poly_terms(n_features: int, degree: int, include_bias: bool, interaction_only: bool) -> int:
    combo_fn = combinations if interaction_only else combinations_with_replacement
    total = 1 if include_bias else 0
    for current_degree in range(1, degree + 1):
        total += sum(1 for _ in combo_fn(range(n_features), current_degree))
    # end for
    return total
# end def _n_poly_terms


class StatisticalLearningOperator(OperatorBase, ParametricOperator, ABC):
    """Base class for statistical learning operators."""

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.R
    # end def infer_dtype

# end class StatisticalLearningOperator


class LinearRegressionFit(StatisticalLearningOperator):
    """Fit linear regression coefficients with optional ridge regularization."""

    NAME = "linear_regression_fit"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(self, include_intercept: bool = True, ridge_alpha: ScalarParameter = 0.0, dtype: DType = DType.R):
        alpha_expr = _ensure_scalar_parameter(ridge_alpha, f"{self.NAME} ridge_alpha")
        super().__init__(include_intercept=include_intercept, ridge_alpha=alpha_expr, dtype=dtype)
        self._include_intercept = bool(include_intercept)
        self._ridge_alpha = alpha_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return self._ridge_alpha.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, y = operands
        if x.rank != 2:
            raise ValueError(f"{self.NAME} expects X rank-2, got rank={x.rank}")
        # end if
        if y.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects y rank-1 or rank-2, got rank={y.rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        n_features = x.shape[1]
        if self._include_intercept:
            return Shape((n_features + 1,))
        # end if
        return Shape((n_features,))
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x_tensor, y_tensor = operands
        x = np.asarray(x_tensor.eval().value, dtype=np.float64)
        y = _flatten_targets(np.asarray(y_tensor.eval().value, dtype=np.float64), self.NAME)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{self.NAME} expects matching sample counts, got {x.shape[0]} and {y.shape[0]}")
        # end if

        alpha = _eval_scalar_parameter(self._ridge_alpha, f"{self.NAME} ridge_alpha")
        if alpha < 0.0:
            raise ValueError("linear_regression_fit requires ridge_alpha >= 0.")
        # end if

        design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1) if self._include_intercept else x

        if alpha > 0.0:
            eye = np.eye(design.shape[1], dtype=np.float64)
            if self._include_intercept:
                eye[0, 0] = 0.0
            # end if
            lhs = design.T @ design + alpha * eye
            rhs = design.T @ y
            beta = np.linalg.pinv(lhs) @ rhs
        else:
            beta = np.linalg.pinv(design) @ y
        # end if

        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(beta, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("LinearRegressionFit does not support backward propagation.")
    # end def _backward

# end class LinearRegressionFit


class LinearRegressionPredict(StatisticalLearningOperator):
    """Predict from fitted linear regression coefficients."""

    NAME = "linear_regression_predict"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(self, include_intercept: bool = True, dtype: DType = DType.R):
        super().__init__(include_intercept=include_intercept, dtype=dtype)
        self._include_intercept = bool(include_intercept)
        self._dtype = dtype
    # end def __init__

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, beta = operands
        if x.rank != 2:
            raise ValueError(f"{self.NAME} expects X rank-2, got rank={x.rank}")
        # end if
        if beta.rank != 1:
            raise ValueError(f"{self.NAME} expects beta rank-1, got rank={beta.rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        return Shape((x.shape[0],))
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x_tensor, beta_tensor = operands
        x = np.asarray(x_tensor.eval().value, dtype=np.float64)
        beta = np.asarray(beta_tensor.eval().value, dtype=np.float64).reshape(-1)
        design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1) if self._include_intercept else x
        if design.shape[1] != beta.shape[0]:
            raise ValueError(f"{self.NAME} expects beta length {design.shape[1]}, got {beta.shape[0]}")
        # end if
        pred = design @ beta
        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(pred, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("LinearRegressionPredict does not support backward propagation.")
    # end def _backward

# end class LinearRegressionPredict


class PolynomialFeatures(StatisticalLearningOperator):
    """Build polynomial feature matrix from univariate or multivariate x."""

    NAME = "polynomial_features"
    ARITY = 1
    IS_VARIADIC = False

    def __init__(
            self,
            degree: ScalarParameter = 2,
            include_bias: bool = True,
            interaction_only: bool = False,
            dtype: DType = DType.R
    ):
        degree_expr = _ensure_scalar_parameter(degree, f"{self.NAME} degree")
        super().__init__(
            degree=degree_expr,
            include_bias=include_bias,
            interaction_only=interaction_only,
            dtype=dtype
        )
        self._degree = degree_expr
        self._include_bias = bool(include_bias)
        self._interaction_only = bool(interaction_only)
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return self._degree.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, = operands
        if x.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects rank-1 or rank-2 x, got rank={x.rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        degree = _eval_int_parameter(self._degree, f"{self.NAME} degree")
        if degree < 1:
            raise ValueError("polynomial_features requires degree >= 1.")
        # end if
        x, = operands
        if x.rank == 1:
            n_terms = degree + 1 if self._include_bias else degree
            return Shape((x.shape[0], n_terms))
        # end if
        n_features = int(x.shape[1])
        n_terms = _n_poly_terms(n_features, degree, self._include_bias, self._interaction_only)
        return Shape((x.shape[0], n_terms))
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        degree = _eval_int_parameter(self._degree, f"{self.NAME} degree")
        if degree < 1:
            raise ValueError("polynomial_features requires degree >= 1.")
        # end if
        x, = operands
        x_val = np.asarray(x.eval().value, dtype=np.float64)
        out = _poly_design_nd(
            x_val,
            degree=degree,
            include_bias=self._include_bias,
            interaction_only=self._interaction_only,
        )
        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(out, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PolynomialFeatures does not support backward propagation.")
    # end def _backward

# end class PolynomialFeatures


class PolynomialRegressionFit(StatisticalLearningOperator):
    """Fit polynomial regression coefficients for rank-1 or rank-2 x."""

    NAME = "polynomial_regression_fit"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(
            self,
            degree: ScalarParameter = 2,
            include_bias: bool = True,
            interaction_only: bool = False,
            ridge_alpha: ScalarParameter = 0.0,
            dtype: DType = DType.R
    ):
        degree_expr = _ensure_scalar_parameter(degree, f"{self.NAME} degree")
        alpha_expr = _ensure_scalar_parameter(ridge_alpha, f"{self.NAME} ridge_alpha")
        super().__init__(
            degree=degree_expr,
            include_bias=include_bias,
            interaction_only=interaction_only,
            ridge_alpha=alpha_expr,
            dtype=dtype
        )
        self._degree = degree_expr
        self._include_bias = bool(include_bias)
        self._interaction_only = bool(interaction_only)
        self._ridge_alpha = alpha_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._degree.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._ridge_alpha.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, y = operands
        if x.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects rank-1 or rank-2 x, got rank={x.rank}")
        # end if
        if y.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects y rank-1 or rank-2, got rank={y.rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        degree = _eval_int_parameter(self._degree, f"{self.NAME} degree")
        if degree < 1:
            raise ValueError("polynomial_regression_fit requires degree >= 1.")
        # end if
        x, = operands[:1]
        n_features = 1 if x.rank == 1 else int(x.shape[1])
        n_terms = _n_poly_terms(n_features, degree, self._include_bias, self._interaction_only)
        return Shape((n_terms,))
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        degree = _eval_int_parameter(self._degree, f"{self.NAME} degree")
        if degree < 1:
            raise ValueError("polynomial_regression_fit requires degree >= 1.")
        # end if
        alpha = _eval_scalar_parameter(self._ridge_alpha, f"{self.NAME} ridge_alpha")
        if alpha < 0.0:
            raise ValueError("polynomial_regression_fit requires ridge_alpha >= 0.")
        # end if

        x_tensor, y_tensor = operands
        x = np.asarray(x_tensor.eval().value, dtype=np.float64)
        y = _flatten_targets(np.asarray(y_tensor.eval().value, dtype=np.float64), self.NAME)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{self.NAME} expects matching sample counts, got {x.shape[0]} and {y.shape[0]}")
        # end if

        design = _poly_design_nd(
            x,
            degree=degree,
            include_bias=self._include_bias,
            interaction_only=self._interaction_only,
        )
        if alpha > 0.0:
            eye = np.eye(design.shape[1], dtype=np.float64)
            if self._include_bias:
                eye[0, 0] = 0.0
            # end if
            lhs = design.T @ design + alpha * eye
            rhs = design.T @ y
            beta = np.linalg.pinv(lhs) @ rhs
        else:
            beta = np.linalg.pinv(design) @ y
        # end if

        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(beta, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PolynomialRegressionFit does not support backward propagation.")
    # end def _backward

# end class PolynomialRegressionFit


class PolynomialRegressionPredict(StatisticalLearningOperator):
    """Predict values from polynomial coefficients."""

    NAME = "polynomial_regression_predict"
    ARITY = 2
    IS_VARIADIC = False

    def __init__(
            self,
            include_bias: bool = True,
            degree: Optional[ScalarParameter] = None,
            interaction_only: bool = False,
            dtype: DType = DType.R
    ):
        degree_expr = _ensure_scalar_parameter(degree, f"{self.NAME} degree") if degree is not None else None
        super().__init__(
            include_bias=include_bias,
            degree=degree_expr,
            interaction_only=interaction_only,
            dtype=dtype
        )
        self._include_bias = bool(include_bias)
        self._degree = degree_expr
        self._interaction_only = bool(interaction_only)
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        if self._degree is None:
            return False
        # end if
        return self._degree.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x, beta = operands
        if x.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects rank-1 or rank-2 x, got rank={x.rank}")
        # end if
        if beta.rank != 1:
            raise ValueError(f"{self.NAME} expects rank-1 beta, got rank={beta.rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        x, _ = operands
        return Shape((x.shape[0],))
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x_tensor, beta_tensor = operands
        x = np.asarray(x_tensor.eval().value, dtype=np.float64)
        beta = np.asarray(beta_tensor.eval().value, dtype=np.float64).reshape(-1)

        if self._degree is None:
            if x.ndim != 1:
                raise ValueError("polynomial_regression_predict requires degree for rank-2 inputs.")
            # end if
            start_power = 0 if self._include_bias else 1
            powers = np.arange(start_power, start_power + beta.shape[0], dtype=np.int64)
            design = np.stack([np.power(x, p) for p in powers], axis=1)
        else:
            degree = _eval_int_parameter(self._degree, f"{self.NAME} degree")
            if degree < 1:
                raise ValueError("polynomial_regression_predict requires degree >= 1.")
            # end if
            design = _poly_design_nd(
                x,
                degree=degree,
                include_bias=self._include_bias,
                interaction_only=self._interaction_only,
            )
            if design.shape[1] != beta.shape[0]:
                raise ValueError(
                    f"{self.NAME} expects beta length {design.shape[1]} for provided degree, got {beta.shape[0]}"
                )
            # end if
        # end if

        pred = design @ beta
        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(pred, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("PolynomialRegressionPredict does not support backward propagation.")
    # end def _backward

# end class PolynomialRegressionPredict


class RegressionMetricOperator(StatisticalLearningOperator, ABC):
    """Base class for scalar regression metrics."""

    ARITY = 2
    IS_VARIADIC = False

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 2
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        y_true, y_pred = operands
        if y_true.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects y_true rank-1 or rank-2, got rank={y_true.rank}")
        # end if
        if y_pred.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects y_pred rank-1 or rank-2, got rank={y_pred.rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.scalar()
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.R
    # end def infer_dtype

    @staticmethod
    def _flatten_pair(operands: Operands, name: str) -> tuple[np.ndarray, np.ndarray]:
        y_true = _flatten_targets(np.asarray(operands[0].eval().value, dtype=np.float64), f"{name} y_true")
        y_pred = _flatten_targets(np.asarray(operands[1].eval().value, dtype=np.float64), f"{name} y_pred")
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"{name} expects matching sample counts, got {y_true.shape[0]} and {y_pred.shape[0]}")
        # end if
        return y_true, y_pred
    # end def _flatten_pair

# end class RegressionMetricOperator


class MSE(RegressionMetricOperator):
    """Mean squared error metric."""

    NAME = "mse"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        y_true, y_pred = self._flatten_pair(operands, self.NAME)
        value = np.mean((y_true - y_pred) ** 2)
        return Tensor(data=np.asarray(value, dtype=np.float32), dtype=DType.R)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("MSE does not support backward propagation.")
    # end def _backward

# end class MSE


class RMSE(RegressionMetricOperator):
    """Root mean squared error metric."""

    NAME = "rmse"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        y_true, y_pred = self._flatten_pair(operands, self.NAME)
        value = np.sqrt(np.mean((y_true - y_pred) ** 2))
        return Tensor(data=np.asarray(value, dtype=np.float32), dtype=DType.R)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("RMSE does not support backward propagation.")
    # end def _backward

# end class RMSE


class MAE(RegressionMetricOperator):
    """Mean absolute error metric."""

    NAME = "mae"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        y_true, y_pred = self._flatten_pair(operands, self.NAME)
        value = np.mean(np.abs(y_true - y_pred))
        return Tensor(data=np.asarray(value, dtype=np.float32), dtype=DType.R)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("MAE does not support backward propagation.")
    # end def _backward

# end class MAE


class R2(RegressionMetricOperator):
    """Coefficient of determination (R-squared) metric."""

    NAME = "r2"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        y_true, y_pred = self._flatten_pair(operands, self.NAME)
        ss_res = np.sum((y_true - y_pred) ** 2)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        if ss_tot == 0.0:
            value = 1.0 if ss_res == 0.0 else 0.0
        else:
            value = 1.0 - (ss_res / ss_tot)
        # end if
        return Tensor(data=np.asarray(value, dtype=np.float32), dtype=DType.R)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("R2 does not support backward propagation.")
    # end def _backward

# end class R2


operator_registry.register(LinearRegressionFit)
operator_registry.register(LinearRegressionPredict)
operator_registry.register(PolynomialFeatures)
operator_registry.register(PolynomialRegressionFit)
operator_registry.register(PolynomialRegressionPredict)
operator_registry.register(MSE)
operator_registry.register(RMSE)
operator_registry.register(MAE)
operator_registry.register(R2)
