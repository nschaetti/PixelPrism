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
from pixelprism.math.dtype import to_numpy
from pixelprism.math.functional import builders as FB


def _make_const(name, values, dtype):
    arr = np.asarray(values, dtype=to_numpy(dtype))
    return pm.const(name=name, data=arr, dtype=dtype), arr
# end def _make_const


def test_concatenate_matches_numpy():
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_left = pm.const("concat_left", data=left.copy(), dtype=pm.DType.R)
    expr_right = pm.const("concat_right", data=right.copy(), dtype=pm.DType.R)
    expr = FB.concatenate((expr_left, expr_right), axis=0)
    expected = np.concatenate([left, right], axis=0)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == expected.shape
# end test_concatenate_matches_numpy


def test_hstack_concatenates_along_second_axis():
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_left = pm.const("hstack_left", data=left.copy(), dtype=pm.DType.R)
    expr_right = pm.const("hstack_right", data=right.copy(), dtype=pm.DType.R)
    expr = FB.hstack((expr_left, expr_right))
    expected = np.hstack([left, right])
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == expected.shape
# end test_hstack_concatenates_along_second_axis


def test_vstack_concatenates_along_first_axis():
    upper = np.arange(6, dtype=np.float32).reshape(2, 3)
    lower = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_upper = pm.const("vstack_upper", data=upper.copy(), dtype=pm.DType.R)
    expr_lower = pm.const("vstack_lower", data=lower.copy(), dtype=pm.DType.R)
    expr = FB.vstack((expr_upper, expr_lower))
    expected = np.vstack([upper, lower])
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == expected.shape
# end test_vstack_concatenates_along_first_axis


def test_from_function_vector_default_indices():
    idx = pm.var("from_fun_i", dtype=pm.DType.Z, shape=())
    body = idx + pm.const("from_fun_offset", data=1, dtype=pm.DType.Z)
    expr = FB.from_function(shape=(4,), body=body, index_vars=[idx])
    expected = np.arange(4, dtype=np.int32) + 1
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (4,)
# end test_from_function_vector_default_indices


def test_linspace_basic_values():
    start = pm.const("lin_start", data=0.0, dtype=pm.DType.R)
    stop = pm.const("lin_stop", data=1.0, dtype=pm.DType.R)
    expr = FB.linspace(start, stop, 5)
    expected = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (5,)
    assert expr.children == tuple()
# end test_linspace_basic_values


def test_linspace_requires_scalar_operands():
    start = pm.const("lin_bad_start", data=[0.0, 1.0], dtype=pm.DType.R)
    stop = pm.const("lin_stop", data=1.0, dtype=pm.DType.R)
    with pytest.raises(ValueError):
        FB.linspace(start, stop, 3)
# end test_linspace_requires_scalar_operands


def test_linspace_num_must_be_constant_integer():
    start = pm.const("lin_start_int", data=0.0, dtype=pm.DType.R)
    stop = pm.const("lin_stop_int", data=1.0, dtype=pm.DType.R)
    num_var = pm.var("lin_num_var", dtype=pm.DType.Z, shape=())
    with pytest.raises(ValueError):
        FB.linspace(start, stop, num_var)
    with pytest.raises(ValueError):
        FB.linspace(start, stop, 2.5)
# end test_linspace_num_must_be_constant_integer


def test_logspace_default_base():
    start = pm.const("log_start", data=0.0, dtype=pm.DType.R)
    stop = pm.const("log_stop", data=1.0, dtype=pm.DType.R)
    expr = FB.logspace(start, stop, 4)
    expected = np.logspace(0.0, 1.0, 4, base=10.0, dtype=np.float32)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (4,)
    assert expr.children == tuple()
# end test_logspace_default_base


def test_logspace_custom_base():
    start = pm.const("log2_start", data=0.0, dtype=pm.DType.R)
    stop = pm.const("log2_stop", data=3.0, dtype=pm.DType.R)
    expr = FB.logspace(start, stop, 4, base=2)
    expected = np.logspace(0.0, 3.0, 4, base=2.0, dtype=np.float32)
    np.testing.assert_allclose(expr.eval().value, expected)
# end test_logspace_custom_base


def test_logspace_base_must_be_constant():
    start = pm.const("log_bad_start", data=0.0, dtype=pm.DType.R)
    stop = pm.const("log_bad_stop", data=1.0, dtype=pm.DType.R)
    base = pm.var("log_base", dtype=pm.DType.Z, shape=())
    with pytest.raises(ValueError):
        FB.logspace(start, stop, 3, base=base)
# end test_logspace_base_must_be_constant


def test_map_simple_case():
    tensor = pm.const("map_tensor", data=[-1.0, 2.0], dtype=pm.DType.R)
    var = pm.var("map_value", dtype=pm.DType.R, shape=())
    body = var * var
    expr = FB.map_(tensor, var.name, body)
    expected = np.square(tensor.eval().value)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == tensor.shape.dims
    assert expr.children == (tensor,)
# end test_map_simple_case


def test_random_builders_shape_dtype_and_ranges():
    normal_expr = FB.normal(shape=(4, 3), loc=0.0, scale=1.0, dtype=pm.DType.R)
    uniform_expr = FB.uniform(shape=(4, 3), low=-2.0, high=5.0, dtype=pm.DType.R)
    randint_expr = FB.randint(shape=(4, 3), low=2, high=7, dtype=pm.DType.Z)
    poisson_expr = FB.poisson(shape=(4, 3), lam=3.0, dtype=pm.DType.Z)
    bernoulli_expr = FB.bernoulli(shape=(4, 3), p=0.25)

    normal_val = normal_expr.eval().value
    uniform_val = uniform_expr.eval().value
    randint_val = randint_expr.eval().value
    poisson_val = poisson_expr.eval().value
    bernoulli_val = bernoulli_expr.eval().value

    assert tuple(normal_expr.shape.dims) == (4, 3)
    assert tuple(uniform_expr.shape.dims) == (4, 3)
    assert tuple(randint_expr.shape.dims) == (4, 3)
    assert tuple(poisson_expr.shape.dims) == (4, 3)
    assert tuple(bernoulli_expr.shape.dims) == (4, 3)

    assert normal_expr.dtype == pm.DType.R
    assert uniform_expr.dtype == pm.DType.R
    assert randint_expr.dtype == pm.DType.Z
    assert poisson_expr.dtype == pm.DType.Z
    assert bernoulli_expr.dtype == pm.DType.Z

    assert np.all(uniform_val >= -2.0)
    assert np.all(uniform_val < 5.0)
    assert np.all(randint_val >= 2)
    assert np.all(randint_val < 7)
    assert np.all(np.isin(bernoulli_val, [0, 1]))
# end test_random_builders_shape_dtype_and_ranges


def test_random_builders_parameters_are_evaluated_on_the_fly():
    loc = pm.var("normal_loc", dtype=pm.DType.R, shape=())
    scale = pm.var("normal_scale", dtype=pm.DType.R, shape=())
    low = pm.var("uniform_low", dtype=pm.DType.R, shape=())
    high = pm.var("uniform_high", dtype=pm.DType.R, shape=())
    randint_low = pm.var("randint_low", dtype=pm.DType.Z, shape=())
    randint_high = pm.var("randint_high", dtype=pm.DType.Z, shape=())
    lam = pm.var("poisson_lam", dtype=pm.DType.R, shape=())
    p = pm.var("bernoulli_p", dtype=pm.DType.R, shape=())

    normal_expr = FB.normal(shape=(32, 16), loc=loc, scale=scale, dtype=pm.DType.R)
    uniform_expr = FB.uniform(shape=(32, 16), low=low, high=high, dtype=pm.DType.R)
    randint_expr = FB.randint(shape=(32, 16), low=randint_low, high=randint_high, dtype=pm.DType.Z)
    poisson_expr = FB.poisson(shape=(32, 16), lam=lam, dtype=pm.DType.Z)
    bernoulli_expr = FB.bernoulli(shape=(32, 16), p=p, dtype=pm.DType.Z)

    with pm.new_context():
        pm.set_value("normal_loc", 3.0)
        pm.set_value("normal_scale", 1e-9)
        pm.set_value("uniform_low", 2.0)
        pm.set_value("uniform_high", 2.1)
        pm.set_value("randint_low", 5)
        pm.set_value("randint_high", 6)
        pm.set_value("poisson_lam", 4.0)
        pm.set_value("bernoulli_p", 1.0)

        normal_val = normal_expr.eval().value
        uniform_val = uniform_expr.eval().value
        randint_val = randint_expr.eval().value
        poisson_val = poisson_expr.eval().value
        bernoulli_val = bernoulli_expr.eval().value

    assert np.allclose(normal_val, np.full((32, 16), 3.0, dtype=np.float32), atol=1e-4)
    assert np.all(uniform_val >= 2.0)
    assert np.all(uniform_val < 2.1)
    assert np.all(randint_val == 5)
    assert np.all(poisson_val >= 0)
    assert np.all(bernoulli_val == 1)
# end test_random_builders_parameters_are_evaluated_on_the_fly


def test_random_builders_validation_errors():
    normal_expr = FB.normal(shape=(2, 2), scale=-1.0)
    with pytest.raises(ValueError):
        normal_expr.eval()
    # end with

    uniform_expr = FB.uniform(shape=(2, 2), low=1.0, high=1.0)
    with pytest.raises(ValueError):
        uniform_expr.eval()
    # end with

    poisson_expr = FB.poisson(shape=(2, 2), lam=-0.1)
    with pytest.raises(ValueError):
        poisson_expr.eval()
    # end with

    bernoulli_expr = FB.bernoulli(shape=(2, 2), p=1.1)
    with pytest.raises(ValueError):
        bernoulli_expr.eval()
    # end with

    randint_expr = FB.randint(shape=(2, 2), low=0)
    with pytest.raises(ValueError):
        randint_expr.eval()
    # end with

    randint_bounds_expr = FB.randint(shape=(2, 2), low=5, high=5)
    with pytest.raises(ValueError):
        randint_bounds_expr.eval()
    # end with
# end test_random_builders_validation_errors
