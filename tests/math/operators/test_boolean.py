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
from pixelprism.math.operators import Eq, Ne, Lt, Le, Gt, Ge, Not, Any, All, And, Or, Xor


def _const_from_values(values, dtype: pm.DType):
    """Create a constant tensor along with the numpy payload it wraps."""
    arr = np.asarray(values, dtype=dtype.to_numpy())
    expr = pm.const(
        name=pm.random_const_name("cmp_operand"),
        data=arr.copy(),
        dtype=dtype,
    )
    return expr, arr
# end def _const_from_values


COMPARISON_OPERATORS = (
    ("eq", Eq, np.equal),
    ("ne", Ne, np.not_equal),
    ("lt", Lt, np.less),
    ("le", Le, np.less_equal),
    ("gt", Gt, np.greater),
    ("ge", Ge, np.greater_equal),
)


COMPARISON_OPERAND_CASES = (
    pytest.param(
        np.array(3.5, dtype=np.float32),
        pm.DType.FLOAT32,
        np.array(3.5, dtype=np.float32),
        pm.DType.FLOAT32,
        id="scalar_float32",
    ),
    pytest.param(
        np.arange(4, dtype=np.int32),
        pm.DType.INT32,
        np.array([0, 2, 1, 3], dtype=np.int32),
        pm.DType.INT32,
        id="vector_int32",
    ),
    pytest.param(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        pm.DType.FLOAT64,
        np.array([[1, 0, 3], [4, 7, 6]], dtype=np.int32),
        pm.DType.INT32,
        id="matrix_mixed_dtypes",
    ),
    pytest.param(
        np.array(
            [[[True, False, True], [False, True, False]],
             [[False, False, True], [True, True, False]]],
            dtype=bool,
        ),
        pm.DType.BOOL,
        np.array(
            [[[True, True, True], [False, False, False]],
             [[False, True, False], [True, False, False]]],
            dtype=bool,
        ),
        pm.DType.BOOL,
        id="tensor3d_bool",
    ),
    pytest.param(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3),
        pm.DType.FLOAT32,
        (np.arange(18, dtype=np.float32).reshape(3, 2, 3) * 0.5),
        pm.DType.FLOAT32,
        id="tensor3d_float32",
    ),
)

NOT_OPERAND_CASES = (
    pytest.param(np.array(True, dtype=bool), id="scalar_true"),
    pytest.param(np.array(False, dtype=bool), id="scalar_false"),
    pytest.param(np.array([True, False, True, False], dtype=bool), id="vector_bool"),
    pytest.param(np.array([[True, False], [False, True]], dtype=bool), id="matrix_bool"),
    pytest.param(np.zeros((2, 1, 3), dtype=bool), id="tensor3d_bool"),
)

BOOLEAN_REDUCTION_CASES = (
    pytest.param(np.array(True, dtype=bool), True, True, id="scalar_true"),
    pytest.param(np.array(False, dtype=bool), False, False, id="scalar_false"),
    pytest.param(np.array([True, False, False], dtype=bool), True, False, id="vector_mixed"),
    pytest.param(np.array([[False, False], [False, False]], dtype=bool), False, False, id="matrix_all_false"),
    pytest.param(np.array([[True, True], [True, True]], dtype=bool), True, True, id="matrix_all_true"),
    pytest.param(
        np.array([[[True], [False]], [[False], [False]]], dtype=bool),
        True,
        False,
        id="tensor3d",
    ),
)

BOOLEAN_BINARY_OPERATORS = (
    ("and", And, np.logical_and),
    ("or", Or, np.logical_or),
    ("xor", Xor, np.logical_xor),
)

BOOLEAN_BINARY_CASES = (
    pytest.param(
        np.array(True, dtype=bool),
        np.array(False, dtype=bool),
        id="scalar_true_false",
    ),
    pytest.param(
        np.array([True, False, True], dtype=bool),
        np.array([False, False, True], dtype=bool),
        id="vector",
    ),
    pytest.param(
        np.array([[True, True], [False, False]], dtype=bool),
        np.array([[True, False], [True, False]], dtype=bool),
        id="matrix",
    ),
    pytest.param(
        np.random.default_rng(0).random((2, 2, 3)) > 0.5,
        np.random.default_rng(1).random((2, 2, 3)) > 0.5,
        id="tensor3d_random",
    ),
)


@pytest.mark.parametrize("op_name, op_cls, np_func", COMPARISON_OPERATORS)
@pytest.mark.parametrize("lhs_values, lhs_dtype, rhs_values, rhs_dtype", COMPARISON_OPERAND_CASES)
def test_comparison_operator_matches_numpy(op_name, op_cls, np_func, lhs_values, lhs_dtype, rhs_values, rhs_dtype):
    """Comparison operators should mirror numpy semantics for varied inputs."""
    lhs_expr, lhs_arr = _const_from_values(lhs_values, lhs_dtype)
    rhs_expr, rhs_arr = _const_from_values(rhs_values, rhs_dtype)
    operator = op_cls()

    assert operator.check_operands([lhs_expr, rhs_expr])
    assert operator.check_shapes([lhs_expr, rhs_expr])

    result = operator.eval([lhs_expr, rhs_expr])
    expected = np_func(lhs_arr, rhs_arr)

    np.testing.assert_array_equal(result.value, expected, err_msg=f"{op_name} mismatch")
    assert result.dtype == pm.DType.BOOL
    assert operator.infer_dtype([lhs_expr, rhs_expr]) == pm.DType.BOOL

    expected_shape = np.asarray(expected).shape
    assert result.shape.dims == expected_shape
    assert operator.infer_shape([lhs_expr, rhs_expr]).dims == expected_shape
# end test_comparison_operator_matches_numpy


@pytest.mark.parametrize("op_cls", [Eq, Ne, Lt, Le, Gt, Ge])
def test_comparison_operator_rejects_mismatched_shapes(op_cls):
    """Every comparison operator must refuse incompatible operand shapes."""
    lhs_expr, _ = _const_from_values(np.zeros((2, 2), dtype=np.float32), pm.DType.FLOAT32)
    rhs_expr, _ = _const_from_values(np.zeros(3, dtype=np.float32), pm.DType.FLOAT32)
    operator = op_cls()

    assert operator.check_operands([lhs_expr, rhs_expr]) is False
    assert operator.check_shapes([lhs_expr, rhs_expr]) is False
# end test_comparison_operator_rejects_mismatched_shapes


@pytest.mark.parametrize("op_name, op_cls, np_func", BOOLEAN_BINARY_OPERATORS)
@pytest.mark.parametrize("lhs_values, rhs_values", BOOLEAN_BINARY_CASES)
def test_boolean_binary_operator_matches_numpy(op_name, op_cls, np_func, lhs_values, rhs_values):
    """Boolean binary operators follow numpy semantics."""
    lhs_expr, lhs_arr = _const_from_values(lhs_values, pm.DType.BOOL)
    rhs_expr, rhs_arr = _const_from_values(rhs_values, pm.DType.BOOL)
    operator = op_cls()

    assert operator.check_operands([lhs_expr, rhs_expr])
    assert operator.check_shapes([lhs_expr, rhs_expr])

    result = operator.eval([lhs_expr, rhs_expr])
    expected = np_func(lhs_arr, rhs_arr)

    np.testing.assert_array_equal(result.value, expected, err_msg=f"{op_name} mismatch")
    assert result.dtype == pm.DType.BOOL
    assert result.shape.dims == expected.shape
    assert operator.infer_shape([lhs_expr, rhs_expr]).dims == expected.shape
    assert operator.infer_dtype([lhs_expr, rhs_expr]) == pm.DType.BOOL
# end test_boolean_binary_operator_matches_numpy


@pytest.mark.parametrize("op_cls", [And, Or, Xor])
def test_boolean_binary_operator_rejects_non_bool(op_cls):
    """Boolean binary operators must reject non-boolean operands."""
    lhs_expr, _ = _const_from_values(np.array([1, 0, 1], dtype=np.int32), pm.DType.INT32)
    rhs_expr, _ = _const_from_values(np.array([True, False, True], dtype=bool), pm.DType.BOOL)
    operator = op_cls()

    assert operator.check_operands([lhs_expr, rhs_expr]) is False
    with pytest.raises(TypeError):
        operator.eval([lhs_expr, rhs_expr])
# end test_boolean_binary_operator_rejects_non_bool


@pytest.mark.parametrize("values", NOT_OPERAND_CASES)
def test_not_operator_matches_numpy(values):
    """Logical not mirrors numpy for boolean tensors."""
    expr, arr = _const_from_values(values, pm.DType.BOOL)
    operator = Not()

    assert operator.check_operands([expr])
    result = operator.eval([expr])

    expected = np.logical_not(arr)
    np.testing.assert_array_equal(result.value, expected)
    assert result.dtype == pm.DType.BOOL
    assert result.shape.dims == expected.shape
    assert operator.infer_shape([expr]).dims == expected.shape
    assert operator.infer_dtype([expr]) == pm.DType.BOOL
# end test_not_operator_matches_numpy


def test_not_operator_rejects_non_bool_operands():
    """Logical not must reject tensors that are not boolean."""
    expr, _ = _const_from_values(np.array([1, 0], dtype=np.int32), pm.DType.INT32)
    operator = Not()

    assert operator.check_operands([expr]) is False
    with pytest.raises(TypeError):
        operator.eval([expr])
# end test_not_operator_rejects_non_bool_operands


@pytest.mark.parametrize("values, expected_any, expected_all", BOOLEAN_REDUCTION_CASES)
def test_any_all_operators(values, expected_any, expected_all):
    """Any/All reductions should match numpy semantics."""
    expr, _ = _const_from_values(values, pm.DType.BOOL)

    any_op = Any()
    all_op = All()

    assert any_op.check_operands([expr])
    assert all_op.check_operands([expr])

    any_result = any_op.eval([expr])
    all_result = all_op.eval([expr])

    assert any_result.dtype == pm.DType.BOOL
    assert all_result.dtype == pm.DType.BOOL
    assert any_result.shape.dims == ()
    assert all_result.shape.dims == ()
    assert any_op.infer_shape([expr]).dims == ()
    assert all_op.infer_shape([expr]).dims == ()
    assert any_result.value.item() is expected_any
    assert all_result.value.item() is expected_all
# end test_any_all_operators


@pytest.mark.parametrize("op_cls", [Any, All])
def test_boolean_reduction_rejects_non_bool(op_cls):
    """Any/All must reject non-boolean tensors."""
    expr, _ = _const_from_values(np.ones((2, 2), dtype=np.float32), pm.DType.FLOAT32)
    operator = op_cls()

    assert operator.check_operands([expr]) is False
    with pytest.raises(TypeError):
        operator.eval([expr])
# end test_boolean_reduction_rejects_non_bool
