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
from pixelprism.math.operators import Eq


def _const_from_values(values, dtype: pm.DType):
    """Create a constant tensor along with the numpy payload it wraps."""
    arr = np.asarray(values, dtype=dtype.to_numpy())
    expr = pm.const(
        name=pm.random_const_name("eq_operand"),
        data=arr.copy(),
        dtype=dtype,
    )
    return expr, arr
# end def _const_from_values


EQ_OPERAND_CASES = (
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
        np.array([0, 1, 0, 3], dtype=np.int32),
        pm.DType.INT32,
        id="vector_int32",
    ),
    pytest.param(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        pm.DType.FLOAT64,
        np.array([[1, 2, 0], [4, 0, 6]], dtype=np.int32),
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
             [[False, True, True], [True, True, False]]],
            dtype=bool,
        ),
        pm.DType.BOOL,
        id="tensor3d_bool",
    ),
)


@pytest.mark.parametrize(
    "lhs_values, lhs_dtype, rhs_values, rhs_dtype",
    EQ_OPERAND_CASES,
)
def test_eq_operator_matches_numpy(lhs_values, lhs_dtype, rhs_values, rhs_dtype):
    """Eq should emit a boolean tensor matching numpy.equal for varied inputs."""
    lhs_expr, lhs_arr = _const_from_values(lhs_values, lhs_dtype)
    rhs_expr, rhs_arr = _const_from_values(rhs_values, rhs_dtype)
    operator = Eq()

    assert operator.check_operands([lhs_expr, rhs_expr])
    assert operator.check_shapes([lhs_expr, rhs_expr])

    result = operator.eval([lhs_expr, rhs_expr])
    expected = np.equal(lhs_arr, rhs_arr)

    np.testing.assert_array_equal(result.value, expected)
    assert result.dtype == pm.DType.BOOL
    assert operator.infer_dtype([lhs_expr, rhs_expr]) == pm.DType.BOOL

    expected_shape = np.asarray(expected).shape
    assert result.shape.dims == expected_shape
    assert operator.infer_shape([lhs_expr, rhs_expr]).dims == expected_shape
# end test_eq_operator_matches_numpy


def test_eq_operator_rejects_mismatched_shapes():
    """Eq should refuse operands with incompatible shapes."""
    lhs_expr, _ = _const_from_values(np.zeros((2, 2), dtype=np.float32), pm.DType.FLOAT32)
    rhs_expr, _ = _const_from_values(np.zeros(3, dtype=np.float32), pm.DType.FLOAT32)
    operator = Eq()

    assert operator.check_operands([lhs_expr, rhs_expr]) is False
    assert operator.check_shapes([lhs_expr, rhs_expr]) is False
# end test_eq_operator_rejects_mismatched_shapes
