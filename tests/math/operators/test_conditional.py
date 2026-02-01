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
from pixelprism.math.functional.conditional import where, if_


def _make_const(name, values, dtype):
    array = np.asarray(values, dtype=dtype.to_numpy())
    return pm.const(name=name, data=array, dtype=dtype), array
# end def _make_const


def test_where_scalar_case():
    cond = pm.const("cond_scalar", data=True, dtype=pm.DType.BOOL)
    x = pm.const("x_scalar", data=1.5, dtype=pm.DType.FLOAT32)
    y = pm.const("y_scalar", data=-2.0, dtype=pm.DType.FLOAT32)
    expr = where(cond, x, y)
    np.testing.assert_allclose(expr.eval().value, np.array(1.5, dtype=np.float32))
    assert expr.shape.dims == ()
    assert expr.dtype == pm.DType.FLOAT32
# end test_where_scalar_case


@pytest.mark.parametrize(
    "cond_vals, x_vals, y_vals",
    [
        ([True, False, True], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        ([False, False], [-1.0, -2.0], [4.0, 5.0]),
    ]
)
def test_where_vector_case(cond_vals, x_vals, y_vals):
    cond, cond_np = _make_const("cond_vec", cond_vals, pm.DType.BOOL)
    x, x_np = _make_const("x_vec", x_vals, pm.DType.FLOAT32)
    y, y_np = _make_const("y_vec", y_vals, pm.DType.FLOAT32)
    expr = where(cond, x, y)
    expected = np.where(cond_np, x_np, y_np)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == cond.shape.dims
# end test_where_vector_case


def test_where_matrix_case():
    cond_values = [[True, False], [False, True]]
    x_values = [[1, 2], [3, 4]]
    y_values = [[-1, -2], [-3, -4]]
    cond, cond_np = _make_const("cond_mat", cond_values, pm.DType.BOOL)
    x, x_np = _make_const("x_mat", x_values, pm.DType.INT32)
    y, y_np = _make_const("y_mat", y_values, pm.DType.INT32)
    expr = where(cond, x, y)
    expected = np.where(cond_np, x_np, y_np)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (2, 2)
    assert expr.dtype == pm.DType.INT32
# end test_where_matrix_case


def test_where_dtype_promotion():
    cond = pm.const("cond_promote", data=[True, False], dtype=pm.DType.BOOL)
    x = pm.const("x_promote", data=[1.0, 2.0], dtype=pm.DType.FLOAT32)
    y = pm.const("y_promote", data=[1, 2], dtype=pm.DType.INT32)
    expr = where(cond, x, y)
    assert expr.dtype == pm.DType.FLOAT32
    expected = np.where(cond.eval().value, x.eval().value, y.eval().value)
    np.testing.assert_allclose(expr.eval().value, expected)
# end test_where_dtype_promotion


def test_where_cond_not_bool_raises():
    cond = pm.const("cond_bad", data=1, dtype=pm.DType.INT32)
    x = pm.const("x_bad", data=1.0, dtype=pm.DType.FLOAT32)
    y = pm.const("y_bad", data=-1.0, dtype=pm.DType.FLOAT32)
    with pytest.raises(TypeError):
        where(cond, x, y)
# end test_where_cond_not_bool_raises


def test_where_shape_mismatch_raises():
    cond = pm.const("cond_shape", data=[True, False], dtype=pm.DType.BOOL)
    x = pm.const("x_shape", data=[[1.0, 2.0], [3.0, 4.0]], dtype=pm.DType.FLOAT32)
    y = pm.const("y_shape", data=[[0.0, 0.0], [0.0, 0.0]], dtype=pm.DType.FLOAT32)
    with pytest.raises(ValueError):
        where(cond, x, y)
# end test_where_shape_mismatch_raises


def test_if_selects_true_branch():
    cond = pm.const("if_cond_true", data=True, dtype=pm.DType.BOOL)
    then_expr = pm.const("if_then", data=2.0, dtype=pm.DType.FLOAT32)
    else_expr = pm.var("if_else_unbound", dtype=pm.DType.FLOAT32, shape=())
    expr = if_(cond, then_expr, else_expr)
    np.testing.assert_allclose(expr.eval().value, np.array(2.0, dtype=np.float32))
    assert expr.shape.dims == ()
    assert expr.dtype == pm.DType.FLOAT32
# end test_if_selects_true_branch


def test_if_selects_false_branch_without_evaluating_true():
    cond = pm.const("if_cond_false", data=False, dtype=pm.DType.BOOL)
    then_expr = pm.var("if_then_unbound", dtype=pm.DType.FLOAT32, shape=())
    else_expr = pm.const("if_else", data=-5.0, dtype=pm.DType.FLOAT32)
    expr = if_(cond, then_expr, else_expr)
    np.testing.assert_allclose(expr.eval().value, np.array(-5.0, dtype=np.float32))
# end test_if_selects_false_branch_without_evaluating_true


def test_if_requires_matching_shapes():
    cond = pm.const("if_cond", data=True, dtype=pm.DType.BOOL)
    then_expr = pm.const("if_then_shape", data=[1.0, 2.0], dtype=pm.DType.FLOAT32)
    else_expr = pm.const("if_else_shape", data=3.0, dtype=pm.DType.FLOAT32)
    with pytest.raises(ValueError):
        if_(cond, then_expr, else_expr)
# end test_if_requires_matching_shapes
