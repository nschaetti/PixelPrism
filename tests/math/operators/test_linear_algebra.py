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
from pixelprism.math.dtype import to_numpy
from pixelprism.math.functional import linear_algebra as LA
from pixelprism.math.functional import elementwise as EL
from pixelprism.math.functional.helpers import apply_operator as _apply_operator
from ._helpers import assert_expr_allclose as _assert_expr_allclose


def _make_const(name, values, dtype):
    """Create a constant expression alongside its NumPy array."""
    array = np.asarray(values, dtype=to_numpy(dtype))
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
    vector, vector_np = _make_const("v", [1.0, 2.0, 3.0], DType.R)
    matrix, matrix_np = _make_const(
        "M",
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        DType.R,
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
        DType.R,
    )
    vector, vector_np = _make_const("v", [1.0, 0.0, 1.0], DType.R)

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
    a, a_np = _make_const("A", left, DType.R)
    b, b_np = _make_const("B", right, DType.R)

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

    lhs, lhs_np = _make_const("lhs", lhs_data, DType.R)
    rhs, rhs_np = _make_const("rhs", rhs_data, DType.R)

    expr = LA.matmul(lhs, rhs)
    expected = np.matmul(lhs_np, rhs_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_batched_matrix_matrix


def test_dot_vector_vector():
    """
    Dot product of two simple vectors should match NumPy's implementation.
    """
    vec_left, left_np = _make_const("a", [1.0, 2.0, 3.0], DType.R)
    vec_right, right_np = _make_const("b", [4.0, 5.0, 6.0], DType.R)

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
    lhs, lhs_np = _make_const("lhs", lhs_values, DType.R)
    rhs, rhs_np = _make_const("rhs", rhs_values, DType.R)

    expr = LA.dot(lhs, rhs)
    expected = np.sum(lhs_np * rhs_np, axis=-1)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
# end test_dot_batched_vectors


def test_outer_vector_vector():
    """
    Standard outer product should expand vectors into a matrix.
    """
    vec_left, lhs_np = _make_const("u", [1.0, 2.0], DType.R)
    vec_right, rhs_np = _make_const("v", [3.0, 4.0, 5.0], DType.R)

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

    lhs, lhs_np = _make_const("lhs", lhs_data, DType.R)
    rhs, rhs_np = _make_const("rhs", rhs_data, DType.R)

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
        DType.R,
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
    tensor, tensor_np = _make_const("transpose_batched", data, DType.R)

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
        DType.R,
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
        DType.R,
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
        DType.R,
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
        DType.R,
    )
    expr = _inverse(matrix)
    expected = np.linalg.inv(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == DType.R
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
    matrix, matrix_np = _make_const("inv_batched", data, DType.R)
    expr = _inverse(matrix)
    expected = np.linalg.inv(matrix_np)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == DType.R
# end test_inverse_batched_matrices


def test_trace_matrix_returns_scalar():
    """
    Trace of a single matrix should match NumPy and collapse to scalar.
    """
    matrix, matrix_np = _make_const(
        "T",
        [[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [5.0, 6.0, 0.0]],
        DType.R,
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
    matrices, batch_np = _make_const("batched", batch_data, DType.R)

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
        DType.R,
    )
    rhs, rhs_np = _make_const(
        "rhs_mixed",
        np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4) * 0.5,
        DType.R,
    )
    gather_data = np.broadcast_to(
        np.asarray([[1.0], [-0.5], [2.0], [-1.5]], dtype=np.float64),
        (2, 4, 1),
    )
    gather, gather_np = _make_const("gather_vec", gather_data, DType.R)

    matmul_expr = LA.matmul(lhs, rhs)
    expr = LA.matmul(matmul_expr, gather)
    expected_matmul = np.matmul(lhs_np, rhs_np)
    expected = np.matmul(expected_matmul, gather_np)

    _assert_expr_allclose(expr, expected)
    assert expr.dtype == DType.R
    assert expr.shape.dims == expected.shape
# end test_linear_algebra_batched_mixed_dtype_chain


def test_linear_algebra_with_elementwise_composition():
    """
    Compose linear algebra ops with elementwise transformations.
    """
    mat_a, mat_a_np = _make_const(
        "comp_a",
        [[1.0, 2.0], [3.0, 4.0]],
        DType.R,
    )
    mat_b, mat_b_np = _make_const(
        "comp_b",
        [[0.5, -1.0], [1.5, 2.5]],
        DType.R,
    )
    vec_c, vec_c_np = _make_const("comp_c", [2.0, -1.0], DType.R)

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


def test_norm_constant_expression_order_vector():
    """
    Norm should honor orders expressed purely with constants.
    """
    vector, vector_np = _make_const("norm_const_vec", [3.0, -4.0, 6.0], DType.R)
    order = (
        pm.const("norm_const_a", data=1.0, dtype=DType.R)
        + pm.const("norm_const_b", data=1.0, dtype=DType.R)
    )

    expr = LA.norm(vector, order=order)
    expected = np.linalg.norm(vector_np.astype(np.float32), ord=2.0)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == ()
    assert expr.dtype == DType.R
# end test_norm_constant_expression_order_vector


def test_norm_variable_order_vector():
    """
    Orders supplied as variables should evaluate through the active context.
    """
    vector, vector_np = _make_const(
        "norm_var_vec",
        [1.5, -0.5, 2.5, -3.5],
        DType.R,
    )
    order_var = pm.var("norm_order_var", dtype=DType.R, shape=())
    expr = LA.norm(vector, order=order_var)

    with pm.new_context():
        pm.set_value("norm_order_var", 3.0)
        expected = np.linalg.norm(vector_np, ord=3.0)
        _assert_expr_allclose(expr, expected)
    # end with
    assert expr.shape.dims == ()
    assert expr.dtype == DType.R
# end test_norm_variable_order_vector


def test_norm_mixed_expression_order():
    """
    Mixed constant/variable orders should evaluate correctly for integer inputs.
    """
    vector, vector_np = _make_const(
        "norm_mix_vec",
        [1, -2, 3, -4, 5, -6],
        DType.Z,
    )
    order_var = pm.var("norm_order_mix", dtype=DType.R, shape=())
    order = (
        order_var
        + pm.const("norm_mix_bias", data=1.5, dtype=DType.R)
        - pm.const("norm_mix_shift", data=0.5, dtype=DType.R)
    )
    expr = LA.norm(vector, order=order)

    with pm.new_context():
        pm.set_value("norm_order_mix", 1.0)
        effective_order = 2.0
        expected = np.linalg.norm(vector_np.astype(np.float32), ord=effective_order)
        _assert_expr_allclose(expr, expected)
    # end with
    assert expr.shape.dims == ()
    assert expr.dtype == DType.R
# end test_norm_mixed_expression_order


def test_norm_rejects_non_vector_inputs():
    """
    Passing tensors with rank != 1 should raise a TypeError.
    """
    matrix, _ = _make_const(
        "norm_bad_matrix",
        [[1.0, -2.0], [3.0, 4.0]],
        DType.R,
    )
    with pytest.raises(TypeError):
        LA.norm(matrix)
    # end with
# end test_norm_rejects_non_vector_inputs


@pytest.mark.parametrize(
    "values,dtype,expected_dtype",
    [
        ([1.0, -2.5, 3.0, -4.5], DType.R, DType.R),
        ([1.0, -2.5, 3.0, -4.5], DType.R, DType.R),
        ([1, -5, 2, 7], DType.Z, DType.R),
    ]
)
def test_infty_norm_vector_types(values, dtype, expected_dtype):
    """
    Infinity norm promotes to floating dtypes and collapses to a scalar.
    """
    vector, vector_np = _make_const("inf_vec", values, dtype)
    expr = LA.infty_norm(vector)
    expected = np.max(np.abs(vector_np.astype(to_numpy(expected_dtype))), axis=-1)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == ()
    assert expr.dtype == expected_dtype
# end test_infty_norm_vector_types


def test_infty_norm_batched_vectors():
    """
    Batched vectors should preserve their leading dimensions after the norm.
    """
    values = np.array(
        [
            [1.0, -3.0, 5.0],
            [2.5, -0.5, 0.0],
            [-1.25, 4.75, -2.25],
        ],
        dtype=np.float64
    )
    vector, vector_np = _make_const("inf_batch", values, DType.R)

    expr = LA.infty_norm(vector)
    expected = np.max(np.abs(vector_np), axis=-1)

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == DType.R
# end test_infty_norm_batched_vectors


@pytest.mark.parametrize(
    "values,dtype,expected_dtype",
    [
        ([[1.0, -2.0], [3.0, 4.0]], DType.R, DType.R),
        ([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], DType.R, DType.R),
        ([[2, -1], [0, 5]], DType.Z, DType.R),
    ]
)
def test_frobenius_norm_single_matrix(values, dtype, expected_dtype):
    """
    Frobenius norm collapses 2-D matrices to scalars honoring dtype promotion.
    """
    matrix, matrix_np = _make_const("fro_single", values, dtype)

    expr = LA.frobenius_norm(matrix)
    working = matrix_np.astype(to_numpy(expected_dtype))
    expected = np.sqrt(np.sum(np.square(working), axis=(-2, -1)))

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == ()
    assert expr.dtype == expected_dtype
# end test_frobenius_norm_single_matrix


def test_frobenius_norm_batched_matrices():
    """
    Batched Frobenius norm reduces the last two axes independently.
    """
    data = np.arange(2 * 3 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 3)
    matrix, matrix_np = _make_const("fro_batch", data, DType.R)

    expr = LA.frobenius_norm(matrix)
    expected = np.sqrt(np.sum(np.square(matrix_np), axis=(-2, -1)))

    _assert_expr_allclose(expr, expected)
    assert expr.shape.dims == expected.shape
    assert expr.dtype == DType.R
# end test_frobenius_norm_batched_matrices


def _shape_constants():
    """Create constant symbolic dimensions used by shape examples."""
    n = pm.const("shape_test_N", data=2, dtype=DType.Z)
    m = pm.const("shape_test_M", data=3, dtype=DType.Z)
    return n, m
# end def _shape_constants


def test_shape_example_matmul_dot_outer_trace_transpose():
    """
    Reproduce example-based linear algebra flow with constant shape dimensions.
    """
    n, m = _shape_constants()
    x_mat = pm.var("shape_x_mat", dtype=DType.R, shape=(2, n))
    y_mat = pm.var("shape_y_mat", dtype=DType.R, shape=(n, m))
    x_vec = pm.var("shape_x_vec", dtype=DType.R, shape=(m,))
    y_vec = pm.var("shape_y_vec", dtype=DType.R, shape=(m,))

    matmul_expr = LA.matmul(x_mat, y_mat)
    dot_expr = LA.dot(x_vec, y_vec)
    outer_expr = LA.outer(x_vec, y_vec)
    trace_expr = LA.trace(outer_expr)
    transpose_expr = LA.transpose(outer_expr)

    assert matmul_expr.dtype == DType.R
    assert dot_expr.dtype == DType.R
    assert outer_expr.dtype == DType.R
    assert trace_expr.dtype == DType.R
    assert transpose_expr.dtype == DType.R

    assert matmul_expr.shape.dims[0] == 2
    assert matmul_expr.shape.dims[1] is m
    assert dot_expr.shape.dims == []
    assert outer_expr.shape.dims[0] is m
    assert outer_expr.shape.dims[1] is m
    assert trace_expr.shape.dims == []
    assert transpose_expr.shape.dims[0] is m
    assert transpose_expr.shape.dims[1] is m

    with pm.new_context():
        x_mat_np = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y_mat_np = np.asarray([[5.0, 6.0, 4.0], [7.0, 8.0, 5.0]], dtype=np.float32)
        x_vec_np = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        y_vec_np = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)

        pm.set_value("shape_x_mat", pm.tensor(x_mat_np, dtype=DType.R))
        pm.set_value("shape_y_mat", pm.tensor(y_mat_np, dtype=DType.R))
        pm.set_value("shape_x_vec", pm.tensor(x_vec_np, dtype=DType.R))
        pm.set_value("shape_y_vec", pm.tensor(y_vec_np, dtype=DType.R))

        expected_matmul = np.matmul(x_mat_np, y_mat_np)
        expected_dot = np.dot(x_vec_np, y_vec_np)
        expected_outer = np.outer(x_vec_np, y_vec_np)
        expected_trace = np.trace(expected_outer)
        expected_transpose = np.transpose(expected_outer)

        _assert_expr_allclose(matmul_expr, expected_matmul)
        _assert_expr_allclose(dot_expr, expected_dot)
        _assert_expr_allclose(outer_expr, expected_outer)
        _assert_expr_allclose(trace_expr, expected_trace)
        _assert_expr_allclose(transpose_expr, expected_transpose)

        assert tuple(matmul_expr.eval().shape.dims) == expected_matmul.shape
        assert tuple(dot_expr.eval().shape.dims) == ()
        assert tuple(outer_expr.eval().shape.dims) == expected_outer.shape
        assert tuple(trace_expr.eval().shape.dims) == ()
        assert tuple(transpose_expr.eval().shape.dims) == expected_transpose.shape
    # end with
# end test_shape_example_matmul_dot_outer_trace_transpose


def test_shape_example_det_inverse_norms():
    """
    Reproduce determinant/inverse/norm examples with constant symbolic shapes.
    """
    n, m = _shape_constants()
    x_square = pm.var("shape_x_square", dtype=DType.R, shape=(2, n))
    x_norm = pm.var("shape_x_norm", dtype=DType.R, shape=(m,))
    x_fro = pm.var("shape_x_fro", dtype=DType.R, shape=(n, n))

    det_expr = _det(x_square)
    inverse_expr = _inverse(x_square)
    norm_expr = LA.norm(x_norm)
    norm_inf_expr = LA.infty_norm(x_norm)
    norm_fro_expr = LA.frobenius_norm(x_fro)

    assert det_expr.dtype == DType.R
    assert inverse_expr.dtype == DType.R
    assert norm_expr.dtype == DType.R
    assert norm_inf_expr.dtype == DType.R
    assert norm_fro_expr.dtype == DType.R

    assert det_expr.shape.dims == []
    assert inverse_expr.shape.dims[0] == 2
    assert inverse_expr.shape.dims[1] is n
    assert norm_expr.shape.dims == []
    assert norm_inf_expr.shape.dims == []
    assert norm_fro_expr.shape.dims == []

    with pm.new_context():
        x_square_np = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        x_norm_np = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
        x_fro_np = np.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        pm.set_value("shape_x_square", pm.tensor(x_square_np, dtype=DType.R))
        pm.set_value("shape_x_norm", pm.tensor(x_norm_np, dtype=DType.R))
        pm.set_value("shape_x_fro", pm.tensor(x_fro_np, dtype=DType.R))

        expected_det = np.linalg.det(x_square_np)
        expected_inverse = np.linalg.inv(x_square_np)
        expected_norm = np.linalg.norm(x_norm_np, ord=2)
        expected_norm_inf = np.max(np.abs(x_norm_np), axis=-1)
        expected_norm_fro = np.sqrt(np.sum(np.square(x_fro_np), axis=(-2, -1)))

        _assert_expr_allclose(det_expr, expected_det)
        _assert_expr_allclose(inverse_expr, expected_inverse)
        _assert_expr_allclose(norm_expr, expected_norm)
        _assert_expr_allclose(norm_inf_expr, expected_norm_inf)
        _assert_expr_allclose(norm_fro_expr, expected_norm_fro)

        assert tuple(det_expr.eval().shape.dims) == ()
        assert tuple(inverse_expr.eval().shape.dims) == expected_inverse.shape
        assert tuple(norm_expr.eval().shape.dims) == ()
        assert tuple(norm_inf_expr.eval().shape.dims) == ()
        assert tuple(norm_fro_expr.eval().shape.dims) == ()
    # end with
# end test_shape_example_det_inverse_norms
