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

from typing import Optional, Union

from .. import as_expr
from ..dtype import DType
from ..math_node import MathNode
from .helpers import apply_operator


ScalarLikeExpr = Union[MathNode, int, float]


__all__ = [
    "linear_regression_fit",
    "linear_regression_predict",
    "polynomial_features",
    "polynomial_regression_fit",
    "polynomial_regression_predict",
    "mse",
    "rmse",
    "mae",
    "r2",
]


def linear_regression_fit(
        x: MathNode,
        y: MathNode,
        include_intercept: bool = True,
        ridge_alpha: ScalarLikeExpr = 0.0,
        dtype: DType = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    y_expr = as_expr(y)
    alpha_expr = as_expr(ridge_alpha)
    return apply_operator(
        op_name="linear_regression_fit",
        operands=(x_expr, y_expr),
        display_name=f"linreg_fit({x_expr.name}, {y_expr.name})",
        include_intercept=include_intercept,
        ridge_alpha=alpha_expr,
        dtype=dtype,
    )
# end def linear_regression_fit


def linear_regression_predict(
        x: MathNode,
        beta: MathNode,
        include_intercept: bool = True,
        dtype: DType = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    beta_expr = as_expr(beta)
    return apply_operator(
        op_name="linear_regression_predict",
        operands=(x_expr, beta_expr),
        display_name=f"linreg_predict({x_expr.name}, {beta_expr.name})",
        include_intercept=include_intercept,
        dtype=dtype,
    )
# end def linear_regression_predict


def polynomial_features(
        x: MathNode,
        degree: ScalarLikeExpr = 2,
        include_bias: bool = True,
        interaction_only: bool = False,
        dtype: DType = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    degree_expr = as_expr(degree)
    return apply_operator(
        op_name="polynomial_features",
        operands=(x_expr,),
        display_name=f"poly_features({x_expr.name})",
        degree=degree_expr,
        include_bias=include_bias,
        interaction_only=interaction_only,
        dtype=dtype,
    )
# end def polynomial_features


def polynomial_regression_fit(
        x: MathNode,
        y: MathNode,
        degree: ScalarLikeExpr = 2,
        include_bias: bool = True,
        interaction_only: bool = False,
        ridge_alpha: ScalarLikeExpr = 0.0,
        dtype: DType = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    y_expr = as_expr(y)
    degree_expr = as_expr(degree)
    alpha_expr = as_expr(ridge_alpha)
    return apply_operator(
        op_name="polynomial_regression_fit",
        operands=(x_expr, y_expr),
        display_name=f"polyreg_fit({x_expr.name}, {y_expr.name})",
        degree=degree_expr,
        include_bias=include_bias,
        interaction_only=interaction_only,
        ridge_alpha=alpha_expr,
        dtype=dtype,
    )
# end def polynomial_regression_fit


def polynomial_regression_predict(
        x: MathNode,
        beta: MathNode,
        include_bias: bool = True,
        degree: Optional[ScalarLikeExpr] = None,
        interaction_only: bool = False,
        dtype: Optional[DType] = DType.R,
) -> MathNode:
    x_expr = as_expr(x)
    beta_expr = as_expr(beta)
    degree_expr = as_expr(degree) if degree is not None else None
    return apply_operator(
        op_name="polynomial_regression_predict",
        operands=(x_expr, beta_expr),
        display_name=f"polyreg_predict({x_expr.name}, {beta_expr.name})",
        include_bias=include_bias,
        degree=degree_expr,
        interaction_only=interaction_only,
        dtype=dtype,
    )
# end def polynomial_regression_predict


def mse(y_true: MathNode, y_pred: MathNode) -> MathNode:
    true_expr = as_expr(y_true)
    pred_expr = as_expr(y_pred)
    return apply_operator(
        op_name="mse",
        operands=(true_expr, pred_expr),
        display_name=f"mse({true_expr.name}, {pred_expr.name})",
    )
# end def mse


def rmse(y_true: MathNode, y_pred: MathNode) -> MathNode:
    true_expr = as_expr(y_true)
    pred_expr = as_expr(y_pred)
    return apply_operator(
        op_name="rmse",
        operands=(true_expr, pred_expr),
        display_name=f"rmse({true_expr.name}, {pred_expr.name})",
    )
# end def rmse


def mae(y_true: MathNode, y_pred: MathNode) -> MathNode:
    true_expr = as_expr(y_true)
    pred_expr = as_expr(y_pred)
    return apply_operator(
        op_name="mae",
        operands=(true_expr, pred_expr),
        display_name=f"mae({true_expr.name}, {pred_expr.name})",
    )
# end def mae


def r2(y_true: MathNode, y_pred: MathNode) -> MathNode:
    true_expr = as_expr(y_true)
    pred_expr = as_expr(y_pred)
    return apply_operator(
        op_name="r2",
        operands=(true_expr, pred_expr),
        display_name=f"r2({true_expr.name}, {pred_expr.name})",
    )
# end def r2
