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
from pixelprism.math.functional.builders import sparse_coo


def test_sparse_coo_vector_assignment():
    """SparseCOO should place values at the requested coordinates."""
    expr = sparse_coo(
        shape=(5,),
        indices=[(0,), (3,), (4,)],
        values=[
            pm.const("sc_vec_a", data=2, dtype=pm.DType.INT32),
            pm.const("sc_vec_b", data=-1, dtype=pm.DType.INT32),
            pm.const("sc_vec_c", data=5, dtype=pm.DType.INT32),
        ]
    )
    expected = np.array([2, 0, 0, -1, 5], dtype=np.int32)
    np.testing.assert_array_equal(expr.eval().value, expected)
    assert expr.shape.dims == (5,)
    assert expr.dtype == pm.DType.INT32
# end test_sparse_coo_vector_assignment


def test_sparse_coo_matrix_placement():
    """2-D sparse COO tensors should assign values into dense storage."""
    expr = sparse_coo(
        shape=(2, 3),
        indices=[(0, 1), (1, 2)],
        values=[
            pm.const("sc_mat_a", data=3.5, dtype=pm.DType.FLOAT32),
            pm.const("sc_mat_b", data=-2.0, dtype=pm.DType.FLOAT32),
        ]
    )
    result = expr.eval().value
    expected = np.zeros((2, 3), dtype=np.float32)
    expected[0, 1] = 3.5
    expected[1, 2] = -2.0
    np.testing.assert_allclose(result, expected)
    assert expr.shape.dims == (2, 3)
    assert expr.dtype == pm.DType.FLOAT32
# end test_sparse_coo_matrix_placement


def test_sparse_coo_promotes_dtypes():
    """Value dtypes should be promoted across entries."""
    expr = sparse_coo(
        shape=(2,),
        indices=[(0,), (1,)],
        values=[
            pm.const("sc_promote_a", data=1, dtype=pm.DType.INT32),
            pm.const("sc_promote_b", data=2.5, dtype=pm.DType.FLOAT64),
        ]
    )
    result = expr.eval()
    assert expr.dtype == pm.DType.FLOAT64
    np.testing.assert_allclose(result.value, np.array([1.0, 2.5], dtype=np.float64))
# end test_sparse_coo_promotes_dtypes


def test_sparse_coo_mismatched_lengths():
    """Indices and values length mismatch should raise an error."""
    with pytest.raises(ValueError):
        sparse_coo(
            shape=(3,),
            indices=[(0,), (1,)],
            values=[pm.const("sc_bad_a", data=1, dtype=pm.DType.INT32)]
        )
# end test_sparse_coo_mismatched_lengths


def test_sparse_coo_invalid_index_rank():
    """Indices of the wrong rank should raise during construction."""
    with pytest.raises(ValueError):
        sparse_coo(
            shape=(2, 2),
            indices=[(0,)],
            values=[pm.const("sc_bad_rank", data=1, dtype=pm.DType.INT32)]
        )
# end test_sparse_coo_invalid_index_rank
