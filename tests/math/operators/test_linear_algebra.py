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

from pixelprism.math import utils, DType
from pixelprism.math.functional import linear_algebra as LA


def test_matmul_vector_matrix():
    """
    Ensure vector @ matrix produces the expected 1-D tensor.
    """
    vector = utils.vector(name="v", value=[1.0, 2.0, 3.0], dtype=DType.FLOAT32)
    matrix = utils.matrix(
        name="M",
        value=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=DType.FLOAT32
    )

    expr = LA.matmul(vector, matrix)
    expected = np.matmul(vector.eval(), matrix.eval())

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_vector_matrix


def test_matmul_matrix_vector():
    """
    Ensure matrix @ vector collapses to 1-D tensor.
    """
    matrix = utils.matrix(
        name="M",
        value=[[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]],
        dtype=DType.FLOAT64
    )
    vector = utils.vector(name="v", value=[1.0, 0.0, 1.0], dtype=DType.FLOAT64)

    expr = LA.matmul(matrix, vector)
    expected = np.matmul(matrix.eval(), vector.eval())

    np.testing.assert_allclose(expr.eval(), expected)
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
    a = utils.matrix(name="A", value=left, dtype=DType.FLOAT32)
    b = utils.matrix(name="B", value=right, dtype=DType.FLOAT32)

    expr = LA.matmul(a, b)
    expected = np.matmul(a.eval(), b.eval())

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_matrix_matrix


def test_matmul_batched_matrix_matrix():
    """
    Confirm batched matmul supports leading dimensions.
    """
    lhs_data = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    rhs_data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    lhs = utils.tensor(name="lhs", data=lhs_data.tolist(), dtype=DType.FLOAT32)
    rhs = utils.tensor(name="rhs", data=rhs_data.tolist(), dtype=DType.FLOAT32)

    expr = LA.matmul(lhs, rhs)
    expected = np.matmul(lhs_data, rhs_data)

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == expected.shape
# end test_matmul_batched_matrix_matrix


def test_dot_vector_vector():
    """
    Dot product of two simple vectors should match NumPy's implementation.
    """
    vec_left = utils.vector(name="a", value=[1.0, 2.0, 3.0], dtype=DType.FLOAT32)
    vec_right = utils.vector(name="b", value=[4.0, 5.0, 6.0], dtype=DType.FLOAT32)

    expr = LA.dot(vec_left, vec_right)
    expected = np.dot(vec_left.eval(), vec_right.eval())

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == ()
# end test_dot_vector_vector


def test_dot_batched_vectors():
    """
    Batched vectors should produce a batch of dot products.
    """
    lhs = utils.tensor(
        name="lhs",
        data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=DType.FLOAT32
    )
    rhs = utils.tensor(
        name="rhs",
        data=[[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]],
        dtype=DType.FLOAT32
    )

    expr = LA.dot(lhs, rhs)
    expected = np.sum(lhs.eval() * rhs.eval(), axis=-1)

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == expected.shape
# end test_dot_batched_vectors


def test_outer_vector_vector():
    """
    Standard outer product should expand vectors into a matrix.
    """
    vec_left = utils.vector(name="u", value=[1.0, 2.0], dtype=DType.FLOAT32)
    vec_right = utils.vector(name="v", value=[3.0, 4.0, 5.0], dtype=DType.FLOAT32)

    expr = LA.outer(vec_left, vec_right)
    expected = np.outer(vec_left.eval(), vec_right.eval())

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == expected.shape
# end test_outer_vector_vector


def test_outer_batched_vectors():
    """
    Batched outer products should expand each batch entry independently.
    """
    lhs_data = [[1.0, 2.0], [3.0, 4.0]]
    rhs_data = [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]

    lhs = utils.tensor(name="lhs", data=lhs_data, dtype=DType.FLOAT32)
    rhs = utils.tensor(name="rhs", data=rhs_data, dtype=DType.FLOAT32)

    expr = LA.outer(lhs, rhs)
    lhs_np = np.asarray(lhs_data, dtype=np.float32)
    rhs_np = np.asarray(rhs_data, dtype=np.float32)
    expected = lhs_np[:, :, None] * rhs_np[:, None, :]

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == expected.shape
# end test_outer_batched_vectors
