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
from pixelprism.math.operators import Eq, Ne, Lt, Le, Gt, Ge


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
