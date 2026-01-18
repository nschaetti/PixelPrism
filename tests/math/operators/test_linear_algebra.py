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
from pixelprism.math import DType
from pixelprism.math.functional import linear_algebra as LA
from pixelprism.math.functional import elementwise as EL
from pixelprism.math.functional.helpers import apply_operator as _apply_operator
from ._helpers import assert_expr_allclose as _assert_expr_allclose


def _make_const(name, values, dtype):
    """Create a constant expression alongside its NumPy array."""
    array = np.asarray(values, dtype=dtype.to_numpy())
    return pm.const(name=name, data=array, dtype=dtype), array
# end def _make_const


def _transpose(expr):
    """Apply the registered transpose operator."""
    return _apply_operator("transpose", (expr,), f"transpose({expr.name})")
# end def _transpose


def _det(expr):
    """Apply the registered determinant operator."""
    return _apply_operator("det", (expr,), f"det({expr.name})")
# end def _det


def _inverse(expr):
    """Apply the registered inverse operator."""
    return _apply_operator("inverse", (expr,), f"inverse({expr.name})")
# end def _inverse


def test_matmul_vector_matrix():
    """
    Ensure vector @ matrix produces the expected 1-D tensor.
    """
    vector, vector_np = _make_const("v", [1.0, 2.0, 3.0], DType.FLOAT32)
    matrix, matrix_np = _make_const(
        "M",
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        DType.FLOAT32,
    )

    expr = LA.matmul(vector, matrix)
    expected = np.matmul(vector_np, matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_vector_matrix


def test_matmul_matrix_vector():
    """
    Ensure matrix @ vector collapses to 1-D tensor.
    """
    matrix, matrix_np = _make_const(
        "M",
        [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]],
        DType.FLOAT64,
    )
    vector, vector_np = _make_const("v", [1.0, 0.0, 1.0], DType.FLOAT64)

    expr = LA.matmul(matrix, vector)
    expected = np.matmul(matrix_np, vector_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_matrix_vector


@pytest.mark.parametrize(
    "left,right",
    [
        (
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ),
        (
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
    ]
)
def test_matmul_matrix_matrix(left, right):
    """
    Validate matrix @ matrix multiplies with standard shape inference.
    """
    a, a_np = _make_const("A", left, DType.FLOAT32)
    b, b_np = _make_const("B", right, DType.FLOAT32)

    expr = LA.matmul(a, b)
    expected = np.matmul(a_np, b_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_matrix_matrix


def test_matmul_batched_matrix_matrix():
    """
    Confirm batched matmul supports leading dimensions.
    """
    lhs_data = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    rhs_data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    lhs, lhs_np = _make_const("lhs", lhs_data, DType.FLOAT32)
    rhs, rhs_np = _make_const("rhs", rhs_data, DType.FLOAT32)

    expr = LA.matmul(lhs, rhs)
    expected = np.matmul(lhs_np, rhs_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_batched_matrix_matrix


def test_dot_vector_vector():
    """
    Dot product of two simple vectors should match NumPy's implementation.
    """
    vec_left, left_np = _make_const("a", [1.0, 2.0, 3.0], DType.FLOAT32)
    vec_right, right_np = _make_const("b", [4.0, 5.0, 6.0], DType.FLOAT32)

    expr = LA.dot(vec_left, vec_right)
    expected = np.dot(left_np, right_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == ()
# end test_dot_vector_vector


def test_dot_batched_vectors():
    """
    Batched vectors should produce a batch of dot products.
    """
    lhs_values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    rhs_values = [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]
    lhs, lhs_np = _make_const("lhs", lhs_values, DType.FLOAT32)
    rhs, rhs_np = _make_const("rhs", rhs_values, DType.FLOAT32)

    expr = LA.dot(lhs, rhs)
    expected = np.sum(lhs_np * rhs_np, axis=-1)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_dot_batched_vectors


def test_outer_vector_vector():
    """
    Standard outer product should expand vectors into a matrix.
    """
    vec_left, lhs_np = _make_const("u", [1.0, 2.0], DType.FLOAT32)
    vec_right, rhs_np = _make_const("v", [3.0, 4.0, 5.0], DType.FLOAT32)

    expr = LA.outer(vec_left, vec_right)
    expected = np.outer(lhs_np, rhs_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_outer_vector_vector


def test_outer_batched_vectors():
    """
    Batched outer products should expand each batch entry independently.
    """
    lhs_data = [[1.0, 2.0], [3.0, 4.0]]
    rhs_data = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]

    lhs, lhs_np = _make_const("lhs", lhs_data, DType.FLOAT32)
    rhs, rhs_np = _make_const("rhs", rhs_data, DType.FLOAT32)

    expr = LA.outer(lhs, rhs)
    expected = lhs_np[:, :, None] * rhs_np[:, None, :]

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_outer_batched_vectors


def test_transpose_matrix_matches_numpy():
    """
    Transposing a 2-D tensor should swap axes like NumPy.
    """
    matrix, matrix_np = _make_const(
        "transpose_matrix",
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        DType.FLOAT32,
    )
    expr = _transpose(matrix)
    expected = np.transpose(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == matrix.dtype
# end test_transpose_matrix_matches_numpy


def test_transpose_batched_tensor_reverses_axes():
    """
    Batched tensors should transpose all axes.
    """
    data = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    tensor, tensor_np = _make_const("transpose_batched", data, DType.FLOAT64)

    expr = _transpose(tensor)
    expected = np.transpose(tensor_np)
    _assert_expr_allclose(expr, expected)
    assert tuple(expr.eval().shape.dims) == tuple(expected.shape)
    assert tuple(expr.shape.dims) == expected.shape
    assert expr.dtype == tensor.dtype
# end test_transpose_batched_tensor_reverses_axes


def test_transpose_is_involution():
    """
    Transpose applied twice should equal the original expression.
    """
    matrix, matrix_np = _make_const(
        "transpose_involution",
        [[0.0, 1.0], [2.0, 3.0]],
        DType.FLOAT32,
    )
    expr = _transpose(_transpose(matrix))

    _assert_expr_allclose(expr, matrix_np)
    assert expr.shape.dims == matrix.shape.dims
# end test_transpose_is_involution


def test_det_scalar_result_matches_numpy():
    """
    Determinant of a square matrix should match NumPy.
    """
    matrix, matrix_np = _make_const(
        "det_square",
        [[2.0, 0.0], [0.0, -4.0]],
        DType.FLOAT64,
    )
    expr = _det(matrix)
    expected = np.linalg.det(matrix_np)
    _assert_expr_allclose(expr, expected, atol=1e-12)
    assert expr.shape.dims == ()
    assert expr.dtype == matrix.dtype
# end test_det_scalar_result_matches_numpy


def test_det_higher_order_matrix():
    """
    Determinant should produce a scalar for larger matrices.
    """
    matrix, matrix_np = _make_const(
        "det_three",
        [[3.0, 1.0, -1.0], [2.0, 0.0, 1.0], [1.0, 4.0, 2.0]],
        DType.FLOAT32,
    )
    expr = _det(matrix)
    expected = np.linalg.det(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == ()
    assert expr.dtype == matrix.dtype
# end test_det_higher_order_matrix


def test_inverse_matrix_matches_numpy():
    """
    Inverting a full-rank matrix should match NumPy.
    """
    matrix, matrix_np = _make_const(
        "inv_square",
        [[1.0, 2.0], [3.0, 4.0]],
        DType.FLOAT64,
    )
    expr = _inverse(matrix)
    expected = np.linalg.inv(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == DType.FLOAT32
# end test_inverse_matrix_matches_numpy


def test_inverse_batched_matrices():
    """
    Batched inverse should operate per matrix in the batch.
    """
    data = np.array(
        [
            [[4.0, 7.0], [2.0, 6.0]],
            [[1.0, 2.0], [3.0, 5.0]],
        ],
        dtype=np.float32,
    )
    matrix, matrix_np = _make_const("inv_batched", data, DType.FLOAT32)
    expr = _inverse(matrix)
    expected = np.linalg.inv(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == DType.FLOAT32
# end test_inverse_batched_matrices


def test_trace_matrix_returns_scalar():
    """
    Trace of a single matrix should match NumPy and collapse to scalar.
    """
    matrix, matrix_np = _make_const(
        "T",
        [[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [5.0, 6.0, 0.0]],
        DType.FLOAT64,
    )

    expr = LA.trace(matrix)
    expected = np.trace(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == ()
    assert expr.dtype == matrix.dtype
# end test_trace_matrix_returns_scalar


def test_trace_batched_matrices():
    """
    Batched traces operate independently across leading dimensions.
    """
    batch_data = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    matrices, batch_np = _make_const("batched", batch_data, DType.FLOAT32)

    expr = LA.trace(matrices)
    expected = np.trace(batch_np, axis1=-2, axis2=-1)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == matrices.dtype
# end test_trace_batched_matrices


def test_linear_algebra_batched_mixed_dtype_chain():
    """
    Combine batched matmul and dot across mixed dtypes.
    """
    lhs, lhs_np = _make_const(
        "lhs_mixed",
        np.arange(2 * 3 * 2, dtype=np.float64).reshape(2, 3, 2),
        DType.FLOAT64,
    )
    rhs, rhs_np = _make_const(
        "rhs_mixed",
        np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4) * 0.5,
        DType.FLOAT32,
    )
    gather_data = np.broadcast_to(
        np.asarray([[1.0], [-0.5], [2.0], [-1.5]], dtype=np.float64),
        (2, 4, 1),
    )
    gather, gather_np = _make_const("gather_vec", gather_data, DType.FLOAT64)

    matmul_expr = LA.matmul(lhs, rhs)
    expr = LA.matmul(matmul_expr, gather)
    expected_matmul = np.matmul(lhs_np, rhs_np)
    expected = np.matmul(expected_matmul, gather_np)

    _assert_expr_allclose(expr, expected)
    assert expr.dtype == DType.FLOAT64
    assert expr.shape.dims == expected.shape
# end test_linear_algebra_batched_mixed_dtype_chain


def test_linear_algebra_with_elementwise_composition():
    """
    Compose linear algebra ops with elementwise transformations.
    """
    mat_a, mat_a_np = _make_const(
        "comp_a",
        [[1.0, 2.0], [3.0, 4.0]],
        DType.FLOAT32,
    )
    mat_b, mat_b_np = _make_const(
        "comp_b",
        [[0.5, -1.0], [1.5, 2.5]],
        DType.FLOAT32,
    )
    vec_c, vec_c_np = _make_const("comp_c", [2.0, -1.0], DType.FLOAT32)

    combined_expr = EL.sqrt(
        EL.absolute(
            LA.matmul(mat_a, mat_b)
            + EL.mul(
                LA.matmul(mat_b, mat_a),
                EL.neg(LA.outer(vec_c, vec_c))
            )
        )
    )
    matmul_np = np.matmul(mat_a_np, mat_b_np)
    outer_np = np.outer(vec_c_np, vec_c_np)
    matmul_flip = np.matmul(mat_b_np, mat_a_np)
    expected = np.sqrt(np.abs(matmul_np + (matmul_flip * -outer_np)))

    _assert_expr_allclose(combined_expr, expected)
    assert combined_expr.shape.dims == expected.shape
# end test_linear_algebra_with_elementwise_composition
