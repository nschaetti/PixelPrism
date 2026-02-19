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

import numpy as np

import pixelprism.math as pm
from pixelprism.math.functional import statistical_learning as SL


def test_linear_regression_fit_and_predict_match_numpy_pinv():
    x_np = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 3.0],
        ],
        dtype=np.float32,
    )
    true_beta = np.array([1.5, -0.75])
    intercept = 2.0
    y_np = intercept + (x_np @ true_beta)

    x = pm.const("linreg_x", data=x_np, dtype=pm.DType.R)
    y = pm.const("linreg_y", data=y_np, dtype=pm.DType.R)

    fit_expr = SL.linear_regression_fit(x, y, include_intercept=True)
    beta = fit_expr.eval().value

    design = np.concatenate([np.ones((x_np.shape[0], 1)), x_np], axis=1)
    expected_beta = np.linalg.pinv(design) @ y_np
    np.testing.assert_allclose(beta, expected_beta.astype(np.float32), rtol=1e-5, atol=1e-6)

    pred_expr = SL.linear_regression_predict(x, fit_expr, include_intercept=True)
    expected_pred = design @ expected_beta
    np.testing.assert_allclose(pred_expr.eval().value, expected_pred.astype(np.float32), rtol=1e-5, atol=1e-5)
# end test_linear_regression_fit_and_predict_match_numpy_pinv


def test_linear_regression_ridge_alpha_runtime_math_expr():
    x_np = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    y_np = np.array([0.0, 1.0, 2.1, 2.9, 4.2], dtype=np.float32)
    x = pm.const("ridge_x", data=x_np, dtype=pm.DType.R)
    y = pm.const("ridge_y", data=y_np, dtype=pm.DType.R)
    alpha = pm.var("ridge_alpha", dtype=pm.DType.R, shape=())

    fit_expr = SL.linear_regression_fit(x, y, include_intercept=True, ridge_alpha=alpha)
    with pm.new_context():
        pm.set_value("ridge_alpha", 0.5)
        beta = fit_expr.eval().value
    # end with

    design = np.concatenate([np.ones((x_np.shape[0], 1)), x_np], axis=1)
    eye = np.eye(design.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0
    expected_beta = np.linalg.pinv(design.T @ design + 0.5 * eye) @ (design.T @ y_np)
    np.testing.assert_allclose(beta, expected_beta.astype(np.float32), rtol=1e-5, atol=1e-5)
# end test_linear_regression_ridge_alpha_runtime_math_expr


def test_polynomial_features_univariate():
    x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = pm.const("poly_x", data=x_np, dtype=pm.DType.R)
    features_expr = SL.polynomial_features(x, degree=3, include_bias=True)
    expected = np.stack([np.ones_like(x_np), x_np, x_np ** 2, x_np ** 3], axis=1)
    np.testing.assert_allclose(features_expr.eval().value, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert features_expr.shape.dims == expected.shape
# end test_polynomial_features_univariate


def test_polynomial_regression_fit_and_predict():
    x_np = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
    y_np = 1.0 + (2.0 * x_np) + (0.5 * (x_np ** 2))
    x = pm.const("polyreg_x", data=x_np, dtype=pm.DType.R)
    y = pm.const("polyreg_y", data=y_np, dtype=pm.DType.R)

    fit_expr = SL.polynomial_regression_fit(x, y, degree=2, include_bias=True)
    beta = fit_expr.eval().value

    design = np.stack([np.ones_like(x_np), x_np, x_np ** 2], axis=1)
    expected_beta = np.linalg.pinv(design) @ y_np
    np.testing.assert_allclose(beta, expected_beta.astype(np.float32), rtol=1e-5, atol=1e-5)

    pred_expr = SL.polynomial_regression_predict(x, fit_expr, include_bias=True)
    expected_pred = design @ expected_beta
    np.testing.assert_allclose(pred_expr.eval().value, expected_pred.astype(np.float32), rtol=1e-5, atol=1e-5)
# end test_polynomial_regression_fit_and_predict


def test_polynomial_multivariate_features_and_regression():
    x_np = np.array(
        [
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
        ],
        dtype=np.float32,
    )
    x = pm.const("poly_multi_x", data=x_np, dtype=pm.DType.R)

    features_expr = SL.polynomial_features(x, degree=2, include_bias=True)
    features = features_expr.eval().value
    expected_features = np.stack(
        [
            np.ones(x_np.shape[0]),
            x_np[:, 0],
            x_np[:, 1],
            x_np[:, 0] ** 2,
            x_np[:, 0] * x_np[:, 1],
            x_np[:, 1] ** 2,
        ],
        axis=1,
    )
    np.testing.assert_allclose(features, expected_features.astype(np.float32), rtol=1e-5, atol=1e-5)

    beta_true = np.array([1.0, 2.0, -0.5, 0.2, 0.1, -0.3], dtype=np.float64)
    y_np = (expected_features @ beta_true).astype(np.float32)
    y = pm.const("poly_multi_y", data=y_np, dtype=pm.DType.R)

    fit_expr = SL.polynomial_regression_fit(x, y, degree=2, include_bias=True)
    beta = fit_expr.eval().value
    expected_beta = np.linalg.pinv(expected_features) @ y_np
    np.testing.assert_allclose(beta, expected_beta.astype(np.float32), rtol=1e-4, atol=1e-4)

    pred_expr = SL.polynomial_regression_predict(x, fit_expr, include_bias=True, degree=2)
    pred = pred_expr.eval().value
    expected_pred = expected_features @ expected_beta
    np.testing.assert_allclose(pred, expected_pred.astype(np.float32), rtol=1e-4, atol=1e-4)
# end test_polynomial_multivariate_features_and_regression


def test_regression_metrics_match_numpy():
    y_true_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_pred_np = np.array([1.2, 1.8, 2.9, 4.3], dtype=np.float32)
    y_true = pm.const("metric_y_true", data=y_true_np, dtype=pm.DType.R)
    y_pred = pm.const("metric_y_pred", data=y_pred_np, dtype=pm.DType.R)

    mse_expr = SL.mse(y_true, y_pred)
    rmse_expr = SL.rmse(y_true, y_pred)
    mae_expr = SL.mae(y_true, y_pred)
    r2_expr = SL.r2(y_true, y_pred)

    mse_expected = np.mean((y_true_np - y_pred_np) ** 2)
    rmse_expected = np.sqrt(mse_expected)
    mae_expected = np.mean(np.abs(y_true_np - y_pred_np))
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2_expected = 1.0 - (ss_res / ss_tot)

    np.testing.assert_allclose(mse_expr.eval().value, np.asarray(mse_expected, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rmse_expr.eval().value, np.asarray(rmse_expected, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mae_expr.eval().value, np.asarray(mae_expected, dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(r2_expr.eval().value, np.asarray(r2_expected, dtype=np.float32), rtol=1e-6, atol=1e-6)
# end test_regression_metrics_match_numpy
