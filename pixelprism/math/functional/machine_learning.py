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

from typing import Optional, Union

from ..build import as_expr
from ..dtype import DType
from ..operators.machine_learning import SKLEARN_AVAILABLE
from ..typing import MathExpr
from .helpers import apply_operator


ScalarLikeExpr = Union[MathExpr, int, float]


__all__ = [
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
]


def _ensure_ml_available() -> None:
    if SKLEARN_AVAILABLE:
        return
    # end if
    raise ImportError("scikit-learn is required for basic ML functions in pixelprism.math.functional.machine_learning")
# end def _ensure_ml_available


def fit(
        x: MathExpr,
        y: MathExpr,
        *,
        max_iter: ScalarLikeExpr = 1000,
        learning_rate: ScalarLikeExpr = 1.0,
        shuffle: bool = True,
        seed: Optional[int] = None,
        fit_intercept: bool = True,
        dtype: DType = DType.R,
) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    y_expr = as_expr(y)
    return apply_operator(
        op_name="perceptron_train",
        operands=(x_expr, y_expr),
        display_name=f"fit({x_expr.name}, {y_expr.name})",
        max_iter=as_expr(max_iter),
        learning_rate=as_expr(learning_rate),
        shuffle=shuffle,
        seed=seed,
        fit_intercept=fit_intercept,
        dtype=dtype,
    )
# end def fit


def predict(x: MathExpr, theta: MathExpr, *, dtype: DType = DType.Z) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    theta_expr = as_expr(theta)
    return apply_operator(
        op_name="perceptron_predict",
        operands=(x_expr, theta_expr),
        display_name=f"predict({x_expr.name}, {theta_expr.name})",
        dtype=dtype,
    )
# end def predict


def decision_boundary(theta: MathExpr, *, dtype: DType = DType.R) -> MathExpr:
    _ensure_ml_available()
    theta_expr = as_expr(theta)
    return apply_operator(
        op_name="perceptron_decision_boundary",
        operands=(theta_expr,),
        display_name=f"decision_boundary({theta_expr.name})",
        dtype=dtype,
    )
# end def decision_boundary


def coefficients(theta: MathExpr, *, dtype: DType = DType.R) -> MathExpr:
    _ensure_ml_available()
    theta_expr = as_expr(theta)
    return apply_operator(
        op_name="perceptron_coefficients",
        operands=(theta_expr,),
        display_name=f"coefficients({theta_expr.name})",
        dtype=dtype,
    )
# end def coefficients


def intercept(theta: MathExpr, *, dtype: DType = DType.R) -> MathExpr:
    _ensure_ml_available()
    theta_expr = as_expr(theta)
    return apply_operator(
        op_name="perceptron_intercept",
        operands=(theta_expr,),
        display_name=f"intercept({theta_expr.name})",
        dtype=dtype,
    )
# end def intercept


def classes(*, dtype: DType = DType.Z) -> MathExpr:
    _ensure_ml_available()
    return apply_operator(
        op_name="perceptron_classes",
        operands=(),
        display_name="classes()",
        dtype=dtype,
    )
# end def classes


def tree_fit(
        x: MathExpr,
        y: MathExpr,
        *,
        max_depth: ScalarLikeExpr = 3,
        min_samples_split: ScalarLikeExpr = 2,
        min_samples_leaf: ScalarLikeExpr = 1,
        criterion: str = "gini",
        dtype: DType = DType.R,
) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    y_expr = as_expr(y)
    return apply_operator(
        op_name="decision_tree_train",
        operands=(x_expr, y_expr),
        display_name=f"tree_fit({x_expr.name}, {y_expr.name})",
        max_depth=as_expr(max_depth),
        min_samples_split=as_expr(min_samples_split),
        min_samples_leaf=as_expr(min_samples_leaf),
        criterion=criterion,
        dtype=dtype,
    )
# end def tree_fit


def tree_predict(x: MathExpr, tree: MathExpr, *, dtype: DType = DType.Z) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    tree_expr = as_expr(tree)
    return apply_operator(
        op_name="decision_tree_predict",
        operands=(x_expr, tree_expr),
        display_name=f"tree_predict({x_expr.name}, {tree_expr.name})",
        dtype=dtype,
    )
# end def tree_predict


def tree_classes(y: MathExpr, *, dtype: DType = DType.Z) -> MathExpr:
    _ensure_ml_available()
    y_expr = as_expr(y)
    return apply_operator(
        op_name="decision_tree_classes",
        operands=(y_expr,),
        display_name=f"tree_classes({y_expr.name})",
        dtype=dtype,
    )
# end def tree_classes


def svm_fit(
        x: MathExpr,
        y: MathExpr,
        *,
        c: ScalarLikeExpr = 1.0,
        max_iter: ScalarLikeExpr = 1000,
        tol: ScalarLikeExpr = 1e-4,
        fit_intercept: bool = True,
        seed: Optional[int] = None,
        dtype: DType = DType.R,
) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    y_expr = as_expr(y)
    return apply_operator(
        op_name="svm_train",
        operands=(x_expr, y_expr),
        display_name=f"svm_fit({x_expr.name}, {y_expr.name})",
        c=as_expr(c),
        max_iter=as_expr(max_iter),
        tol=as_expr(tol),
        fit_intercept=fit_intercept,
        seed=seed,
        dtype=dtype,
    )
# end def svm_fit


def svm_predict(x: MathExpr, theta: MathExpr, *, dtype: DType = DType.Z) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    theta_expr = as_expr(theta)
    return apply_operator(
        op_name="svm_predict",
        operands=(x_expr, theta_expr),
        display_name=f"svm_predict({x_expr.name}, {theta_expr.name})",
        dtype=dtype,
    )
# end def svm_predict


def svm_decision_function(x: MathExpr, theta: MathExpr, *, dtype: DType = DType.R) -> MathExpr:
    _ensure_ml_available()
    x_expr = as_expr(x)
    theta_expr = as_expr(theta)
    return apply_operator(
        op_name="svm_decision_function",
        operands=(x_expr, theta_expr),
        display_name=f"svm_decision_function({x_expr.name}, {theta_expr.name})",
        dtype=dtype,
    )
# end def svm_decision_function


def svm_classes(*, dtype: DType = DType.Z) -> MathExpr:
    _ensure_ml_available()
    return apply_operator(
        op_name="svm_classes",
        operands=(),
        display_name="svm_classes()",
        dtype=dtype,
    )
# end def svm_classes
