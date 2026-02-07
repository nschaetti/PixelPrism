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

# Imports
import numpy as np
import pytest

import pixelprism.math as pm
from pixelprism.math import utils, DType, render
from pixelprism.math.dtype import to_numpy
from pixelprism.math.functional import reduction as R


def _vector_operand():
    return utils.vector(
        value=[1.0, -2.0, 3.0, 0.5],
        dtype=DType.R
    )
# end def _vector_operand


def _matrix_operand():
    return utils.matrix(
        value=[[1.0, 2.5, -3.0], [4.0, 0.0, 1.5]],
        dtype=DType.R
    )
# end def _matrix_operand


def _tensor_operand():
    return utils.tensor(
        data=[[[1.0, -1.0], [2.0, 3.0]], [[-0.5, 0.5], [4.0, -2.0]]],
        dtype=DType.R
    )
# end def _tensor_operand


def _const_expr(name, values, dtype):
    """Create a constant expression and its numpy payload."""
    arr = np.asarray(values, dtype=to_numpy(dtype))
    return pm.const(name=name, data=arr, dtype=dtype), arr
# end def _const_expr


OPERAND_FACTORIES = (
    ("vector", _vector_operand),
    ("matrix", _matrix_operand),
    ("tensor", _tensor_operand),
)


REDUCTION_CASES = (
    ("sum", R.sum, np.sum),
    ("mean", R.mean, np.mean),
    ("std", R.std, np.std),
    ("median", R.median, np.median),
    ("max", R.max, np.max),
    ("min", R.min, np.min),
)


@pytest.mark.parametrize("op_name, op_func, np_func", REDUCTION_CASES)
@pytest.mark.parametrize("operand_name, operand_factory", OPERAND_FACTORIES)
def test_reduction_scalar_result(op_name, op_func, np_func, operand_name, operand_factory):
    """
    Each reduction should collapse arbitrary tensor ranks to a scalar.
    """
    tensor_data = operand_factory()
    operand_expr = pm.const(
        name=f"{operand_name}_const",
        data=tensor_data.value.copy(),
        dtype=tensor_data.dtype
    )
    expr = op_func(operand_expr)
    operand_values = tensor_data.value
    expected = np.array(np_func(operand_values), dtype=operand_values.dtype)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == ()
    assert expr.dtype == tensor_data.dtype
# end test_reduction_scalar_result


def test_reduction_expr_op_matrix():
    # Variables
    x = pm.var("x", dtype=pm.DType.R, shape=(2, 2))
    y = pm.var("y", dtype=pm.DType.R, shape=(2, 2))
    n = pm.var("n", dtype=pm.DType.Z, shape=())
    i = pm.var("i", dtype=pm.DType.Z, shape=())

    # Math equations
    z1 = R.sum(x, axis=0)
    z2 = R.sum(y, axis=1)
    z3 = z1 + z2
    z4 = R.summation(i * n, 1, 10, "i")

    # Set value and evaluate
    with pm.new_context():
        pm.set_value("x", [[1.0, 2.0], [3.0, 4.0]])
        pm.set_value("y", [[5.0, 6.0], [7.0, 8.0]])
        pm.set_value("n", 2)
        np.testing.assert_allclose(z1.eval().value, np.array([4.0, 6.0], dtype=np.float32))
        np.testing.assert_allclose(z2.eval().value, np.array([11.0, 15.0], dtype=np.float32))
        np.testing.assert_allclose(z3.eval().value, np.array([15.0, 21.0], dtype=np.float32))
        np.testing.assert_allclose(z4.eval().value, np.array(110.0, dtype=np.float32))
    # end with
# end test test_reduction_expr_op_matrix


AXIS_REDUCTION_CASES = (
    ("sum", R.sum, np.sum),
    ("mean", R.mean, np.mean),
    ("std", R.std, np.std),
    ("median", R.median, np.median),
    ("max", R.max, np.max),
    ("min", R.min, np.min),
)


AXIS_INPUT_CASES = (
    (
        "float32_matrix_axis0",
        [[-1.0, 2.0, 0.5], [3.5, -4.0, 1.0]],
        pm.DType.R,
        0,
    ),
    (
        "float32_matrix_axis1",
        [[1.0, 3.0], [-2.5, 4.5], [0.0, 1.0]],
        pm.DType.R,
        1,
    ),
    (
        "float64_tensor_axis2",
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        pm.DType.R,
        2,
    ),
)


@pytest.mark.parametrize("op_name, op_func, np_func", AXIS_REDUCTION_CASES)
@pytest.mark.parametrize("case_name, values, dtype, axis", AXIS_INPUT_CASES)
def test_reduction_axis_results(op_name, op_func, np_func, case_name, values, dtype, axis):
    expr, np_values = _const_expr(f"{op_name}_{case_name}", values, dtype)
    result = op_func(expr, axis=axis)
    expected = np_func(np_values, axis=axis)
    np.testing.assert_allclose(result.eval().value, expected)
    assert result.dtype == dtype
    assert result.shape.dims == tuple(np.asarray(expected).shape)
# end test_reduction_axis_results


PERCENTILE_CASES = (
    ("q1", R.q1, 25.0),
    ("q3", R.q3, 75.0),
)


@pytest.mark.parametrize("op_name, op_func, percentile", PERCENTILE_CASES)
@pytest.mark.parametrize("case_name, values, dtype, axis", AXIS_INPUT_CASES)
def test_percentile_axis_results(op_name, op_func, percentile, case_name, values, dtype, axis):
    expr, np_values = _const_expr(f"{op_name}_{case_name}", values, dtype)
    result = op_func(expr, axis=axis)
    expected = np.percentile(np_values, percentile, axis=axis)
    np.testing.assert_allclose(result.eval().value, expected)
    assert result.dtype == dtype
    assert result.shape.dims == tuple(np.asarray(expected).shape)
# end test_percentile_axis_results


def test_percentile_scalar_results():
    expr, values = _const_expr("percentile_scalar", [1.0, 2.0, 4.0, 8.0], pm.DType.R)
    q1_expr = R.q1(expr)
    q3_expr = R.q3(expr)
    np.testing.assert_allclose(q1_expr.eval().value, np.percentile(values, 25))
    np.testing.assert_allclose(q3_expr.eval().value, np.percentile(values, 75))
    assert q1_expr.shape.dims == ()
    assert q3_expr.shape.dims == ()
# end test_percentile_scalar_results


@pytest.mark.parametrize("op_func", (R.sum, R.mean, R.std, R.median, R.max, R.min, R.q1, R.q3))
def test_reduction_invalid_positive_axis(op_func):
    expr, _ = _const_expr("invalid_axis_tensor", np.arange(8).reshape(2, 2, 2), pm.DType.R)
    with pytest.raises(ValueError):
        op = op_func(expr, axis=5)
        op.eval()
    # end with
# end test_reduction_invalid_positive_axis


def test_reduction_negative_axis_rejected():
    expr, _ = _const_expr("negative_axis_tensor", [[1.0, -1.0], [2.0, 3.5]], pm.DType.R)
    with pytest.raises(ValueError):
        R.sum(expr, axis=-1)
    # end with
# end test_reduction_negative_axis_rejected


def test_summation_polynomial_expression():
    idx = pm.var("sum_poly_idx", dtype=pm.DType.Z, shape=())
    coeff = pm.var("sum_poly_coeff", dtype=pm.DType.Z, shape=())
    bias = pm.const("poly_bias", data=5, dtype=pm.DType.Z)
    poly_body = idx * idx + coeff * idx + bias
    expr = R.summation(poly_body, lower=-2, upper=2, i="sum_poly_idx")
    with pm.new_context():
        pm.set_value("sum_poly_coeff", 3)
        expected = sum((i * i) + (3 * i) + 5 for i in range(-2, 3))
        np.testing.assert_allclose(expr.eval().value, np.array(expected, dtype=np.int32))
    # end with
# end test_summation_polynomial_expression


def test_nested_summations_with_dependent_bounds():
    outer_idx = pm.var("outer_idx", dtype=pm.DType.Z, shape=())
    inner_idx = pm.var("inner_idx", dtype=pm.DType.Z, shape=())
    weight = pm.var("nested_weight", dtype=pm.DType.R, shape=())
    bias = pm.const("nested_bias", data=1.0, dtype=pm.DType.R)

    inner_body = (inner_idx + outer_idx) * weight + bias
    inner_sum = R.summation(
        inner_body,
        lower=outer_idx,
        upper=outer_idx + pm.const("inner_span", data=2, dtype=pm.DType.Z),
        i="inner_idx"
    )

    nested_expr = R.summation(
        inner_sum * (outer_idx + pm.const("outer_offset", data=1, dtype=pm.DType.Z)),
        lower=1,
        upper=3,
        i="outer_idx"
    )

    with pm.new_context():
        pm.set_value("nested_weight", 0.5)
        expected = 0.0
        for outer in range(1, 4):
            inner_total = 0.0
            for inner in range(outer, outer + 3):
                inner_total += (inner + outer) * 0.5 + 1.0
            # end for
            expected += inner_total * (outer + 1)
        # end for
        np.testing.assert_allclose(nested_expr.eval().value, np.array(expected, dtype=np.float32))
    # end with
# end test_nested_summations_with_dependent_bounds


def test_product_linear_body_matches_python():
    idx = pm.var("prod_idx", dtype=pm.DType.Z, shape=())
    scale = pm.var("prod_scale", dtype=pm.DType.R, shape=())
    offset = pm.const("prod_offset", data=1.5, dtype=pm.DType.R)
    shift = pm.const("prod_shift", data=1, dtype=pm.DType.Z)

    body = (idx + shift) * scale + offset
    expr = R.product(body, lower=1, upper=3, i="prod_idx")

    with pm.new_context():
        pm.set_value("prod_scale", 0.5)
        expected = 1.0
        for i in range(1, 4):
            expected *= ((i + 1) * 0.5 + 1.5)
        np.testing.assert_allclose(expr.eval().value, np.array(expected, dtype=np.float32))
    # end with
# end test_product_linear_body_matches_python


def test_product_empty_range_returns_identity():
    idx = pm.var("prod_idx_empty", dtype=pm.DType.Z, shape=())
    term = idx + pm.const("prod_term_shift", data=2, dtype=pm.DType.Z)
    lower = pm.const("prod_empty_lower", data=3, dtype=pm.DType.Z)
    upper = pm.const("prod_empty_upper", data=2, dtype=pm.DType.Z)

    expr = R.product(term, lower=lower, upper=upper, i="prod_idx_empty")
    np.testing.assert_allclose(expr.eval().value, np.array(1, dtype=np.int32))
# end test_product_empty_range_returns_identity
