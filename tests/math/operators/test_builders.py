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
from pixelprism.math.functional import builders as FB


def _make_const(name, values, dtype):
    arr = np.asarray(values, dtype=dtype.to_numpy())
    return pm.const(name=name, data=arr, dtype=dtype), arr
# end def _make_const


def test_concatenate_matches_numpy():
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_left = pm.const("concat_left", data=left.copy(), dtype=pm.DType.FLOAT32)
    expr_right = pm.const("concat_right", data=right.copy(), dtype=pm.DType.FLOAT32)
    expr = FB.concatenate((expr_left, expr_right), axis=0)
    expected = np.concatenate([left, right], axis=0)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == expected.shape
# end test_concatenate_matches_numpy


def test_hstack_concatenates_along_second_axis():
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_left = pm.const("hstack_left", data=left.copy(), dtype=pm.DType.FLOAT32)
    expr_right = pm.const("hstack_right", data=right.copy(), dtype=pm.DType.FLOAT32)
    expr = FB.hstack((expr_left, expr_right))
    expected = np.hstack([left, right])
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == expected.shape
# end test_hstack_concatenates_along_second_axis


def test_vstack_concatenates_along_first_axis():
    upper = np.arange(6, dtype=np.float32).reshape(2, 3)
    lower = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_upper = pm.const("vstack_upper", data=upper.copy(), dtype=pm.DType.FLOAT32)
    expr_lower = pm.const("vstack_lower", data=lower.copy(), dtype=pm.DType.FLOAT32)
    expr = FB.vstack((expr_upper, expr_lower))
    expected = np.vstack([upper, lower])
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == expected.shape
# end test_vstack_concatenates_along_first_axis


def test_from_function_vector_default_indices():
    idx = pm.var("from_fun_i", dtype=pm.DType.INT32, shape=())
    body = idx + pm.const("from_fun_offset", data=1, dtype=pm.DType.INT32)
    expr = FB.from_function(shape=(4,), body=body, index_vars=[idx])
    expected = np.arange(4, dtype=np.int32) + 1
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (4,)
# end test_from_function_vector_default_indices


def test_linspace_basic_values():
    start = pm.const("lin_start", data=0.0, dtype=pm.DType.FLOAT32)
    stop = pm.const("lin_stop", data=1.0, dtype=pm.DType.FLOAT32)
    expr = FB.linspace(start, stop, 5)
    expected = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (5,)
    assert expr.children == tuple()
# end test_linspace_basic_values


def test_linspace_requires_scalar_operands():
    start = pm.const("lin_bad_start", data=[0.0, 1.0], dtype=pm.DType.FLOAT32)
    stop = pm.const("lin_stop", data=1.0, dtype=pm.DType.FLOAT32)
    with pytest.raises(ValueError):
        FB.linspace(start, stop, 3)
# end test_linspace_requires_scalar_operands


def test_linspace_num_must_be_constant_integer():
    start = pm.const("lin_start_int", data=0.0, dtype=pm.DType.FLOAT32)
    stop = pm.const("lin_stop_int", data=1.0, dtype=pm.DType.FLOAT32)
    num_var = pm.var("lin_num_var", dtype=pm.DType.INT32, shape=())
    with pytest.raises(ValueError):
        FB.linspace(start, stop, num_var)
    with pytest.raises(ValueError):
        FB.linspace(start, stop, 2.5)
# end test_linspace_num_must_be_constant_integer


def test_logspace_default_base():
    start = pm.const("log_start", data=0.0, dtype=pm.DType.FLOAT32)
    stop = pm.const("log_stop", data=1.0, dtype=pm.DType.FLOAT32)
    expr = FB.logspace(start, stop, 4)
    expected = np.logspace(0.0, 1.0, 4, base=10.0, dtype=np.float32)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (4,)
    assert expr.children == tuple()
# end test_logspace_default_base


def test_logspace_custom_base():
    start = pm.const("log2_start", data=0.0, dtype=pm.DType.FLOAT32)
    stop = pm.const("log2_stop", data=3.0, dtype=pm.DType.FLOAT32)
    expr = FB.logspace(start, stop, 4, base=2)
    expected = np.logspace(0.0, 3.0, 4, base=2.0, dtype=np.float32)
    np.testing.assert_allclose(expr.eval().value, expected)
# end test_logspace_custom_base


def test_logspace_base_must_be_constant():
    start = pm.const("log_bad_start", data=0.0, dtype=pm.DType.FLOAT32)
    stop = pm.const("log_bad_stop", data=1.0, dtype=pm.DType.FLOAT32)
    base = pm.var("log_base", dtype=pm.DType.INT32, shape=())
    with pytest.raises(ValueError):
        FB.logspace(start, stop, 3, base=base)
# end test_logspace_base_must_be_constant


def test_map_simple_case():
    tensor = pm.const("map_tensor", data=[-1.0, 2.0], dtype=pm.DType.FLOAT32)
    var = pm.var("map_value", dtype=pm.DType.FLOAT32, shape=())
    body = var * var
    expr = FB.map_(tensor, var.name, body)
    expected = np.square(tensor.eval().value)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == tensor.shape.dims
    assert expr.children == (tensor,)
# end test_map_simple_case
