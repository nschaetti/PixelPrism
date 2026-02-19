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
import pytest

import pixelprism.math as pm
from pixelprism.math.functional import stats as ST


def test_covariance_matches_numpy_matrix_mode():
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 5.0, 7.0],
            [3.0, 1.0, 0.0],
            [4.0, 4.0, 2.0],
        ],
        dtype=np.float32,
    )
    expr = pm.const("cov_matrix", data=values, dtype=pm.DType.R)
    cov_expr = ST.cov(expr, rowvar=False, ddof=1)
    expected = np.cov(values, rowvar=False, ddof=1)
    np.testing.assert_allclose(cov_expr.eval().value, expected.astype(np.float32), rtol=1e-5, atol=1e-6)
    assert cov_expr.shape.dims == expected.shape
    assert cov_expr.dtype == pm.DType.R
# end test_covariance_matches_numpy_matrix_mode


def test_correlation_matches_numpy_vector_pair():
    x_vals = np.array([1.0, 3.0, 2.0, 5.0, 4.0], dtype=np.float32)
    y_vals = np.array([2.0, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    x = pm.const("corr_x", data=x_vals, dtype=pm.DType.R)
    y = pm.const("corr_y", data=y_vals, dtype=pm.DType.R)
    corr_expr = ST.corr(x, y)
    expected = np.corrcoef(x_vals, y_vals)
    np.testing.assert_allclose(corr_expr.eval().value, expected.astype(np.float32), rtol=1e-5, atol=1e-6)
    assert corr_expr.shape.dims == (2, 2)
# end test_correlation_matches_numpy_vector_pair


def test_zscore_matches_numpy_and_runtime_params():
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    x = pm.const("zscore_x", data=data, dtype=pm.DType.R)
    ddof = pm.var("zscore_ddof", dtype=pm.DType.Z, shape=())
    eps = pm.var("zscore_eps", dtype=pm.DType.R, shape=())
    expr = ST.zscore(x, axis=0, ddof=ddof, eps=eps)

    with pm.new_context():
        pm.set_value("zscore_ddof", 1)
        pm.set_value("zscore_eps", 1e-8)
        expected = (data - np.mean(data, axis=0, keepdims=True)) / (
            np.std(data, axis=0, ddof=1, keepdims=True) + 1e-8
        )
        np.testing.assert_allclose(expr.eval().value, expected.astype(np.float32), rtol=1e-5, atol=1e-6)
    # end with

    with pm.new_context():
        pm.set_value("zscore_ddof", 0)
        pm.set_value("zscore_eps", 0.0)
        with pytest.raises(ValueError):
            expr.eval()
        # end with
    # end with

    assert expr.shape.dims == data.shape
# end test_zscore_matches_numpy_and_runtime_params
