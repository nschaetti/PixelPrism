#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import unittest
import numpy as np
from pixel_prism.data import (
    Matrix2D, TMatrix2D, add_t, sub_t, mul_t, scalar_mul_t, transpose_t,
    inverse_t, rotate_point_t, determinant_t, trace_t
)
from pixel_prism.data import Point2D, Scalar, TScalar

class TestMatrix2D(unittest.TestCase):

    def test_initialization(self):
        # Test default initialization
        matrix = Matrix2D()
        expected = np.identity(3)
        np.testing.assert_array_equal(matrix.get(), expected)

        # Test initialization with custom matrix
        custom_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        matrix = Matrix2D(custom_matrix)
        np.testing.assert_array_equal(matrix.get(), custom_matrix)
    # end test_initialization

    def test_set_get(self):
        # Test setting and getting the matrix
        matrix = Matrix2D()
        new_matrix = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        matrix.set(new_matrix)
        np.testing.assert_array_equal(matrix.get(), new_matrix)
    # end test_set_get

    def test_copy(self):
        # Test the copy method
        matrix = Matrix2D(np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]))
        matrix_copy = matrix.copy()
        np.testing.assert_array_equal(matrix.get(), matrix_copy.get())
        self.assertIsNot(matrix, matrix_copy)
    # end test_copy

    def test_addition(self):
        # Test addition of two matrices
        matrix1 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        matrix2 = Matrix2D(np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]))
        result = matrix1 + matrix2
        expected = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_addition

    def test_subtraction(self):
        # Test subtraction of two matrices
        matrix1 = Matrix2D(np.array([[10, 9, 8], [7, 6, 5], [4, 3, 2]]))
        matrix2 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = matrix1 - matrix2
        expected = np.array([[9, 7, 5], [3, 1, -1], [-3, -5, -7]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_subtraction

    def test_multiplication(self):
        # Test multiplication of two matrices
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[2, 0], [1, 2]]))
        result = matrix1 * matrix2
        expected = np.array([[4, 4], [10, 8]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_multiplication

    def test_division(self):
        # Test division of a matrix by a scalar
        matrix = Matrix2D(np.array([[2, 4], [6, 8]]))
        result = matrix / 2
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_division

    def test_equality(self):
        # Test equality of two matrices
        matrix1 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        matrix2 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assertTrue(matrix1 == matrix2)
    # end test_equality

    def test_transpose(self):
        # Test the transpose of a matrix
        matrix = Matrix2D(np.array([[1, 2, 3], [4, 5, 6]]))
        result = transpose_t(matrix)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_transpose

    def test_inverse(self):
        # Test the inverse of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = inverse_t(matrix)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result.get(), expected)
    # end test_inverse

    def test_determinant(self):
        # Test the determinant of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = determinant_t(matrix)
        expected = -2.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_determinant

    def test_trace(self):
        # Test the trace of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = trace_t(matrix)
        expected = 5.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_trace

    def test_rotate_point_t(self):
        # Test rotating a point with a matrix
        point = Point2D(1, 0)
        rotation_matrix = Matrix2D(np.array([[0, -1], [1, 0]]))  # 90 degrees rotation matrix
        result = rotate_point_t(rotation_matrix, point)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_rotate_point_t


class TestTMatrix2D(unittest.TestCase):

    def test_add_t(self):
        # Test TMatrix2D addition
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[5, 6], [7, 8]]))
        result = add_t(matrix1, matrix2)
        expected = np.array([[6, 8], [10, 12]])
        np.testing.assert_array_equal(result.get(), expected)

        # Modify one of the source matrices and check the result
        matrix1.matrix = np.array([[2, 2], [2, 2]])
        expected_updated = np.array([[7, 8], [9, 10]])
        np.testing.assert_array_equal(result.get(), expected_updated)

    def test_sub_t(self):
        # Test TMatrix2D subtraction
        matrix1 = Matrix2D(np.array([[5, 6], [7, 8]]))
        matrix2 = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = sub_t(matrix1, matrix2)
        expected = np.array([[4, 4], [4, 4]])
        np.testing.assert_array_equal(result.get(), expected)

        # Modify one of the source matrices and check the result
        matrix1.matrix = np.array([[6, 6], [6, 6]])
        expected_updated = np.array([[5, 4], [3, 2]])
        np.testing.assert_array_equal(result.get(), expected_updated)

    def test_mul_t(self):
        # Test TMatrix2D multiplication
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[2, 0], [1, 2]]))
        result = mul_t(matrix1, matrix2)
        expected = np.array([[4, 4], [10, 8]])
        np.testing.assert_array_equal(result.get(), expected)

    def test_scalar_mul_t(self):
        # Test TMatrix2D scalar multiplication
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        scalar = Scalar(2)
        result = scalar_mul_t(matrix, scalar)
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result.get(), expected)

    def test_transpose_t(self):
        # Test TMatrix2D transpose
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = transpose_t(matrix)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result.get(), expected)

    def test_inverse_t(self):
        # Test TMatrix2D inverse
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = inverse_t(matrix)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result.get(), expected)

    def test_determinant_t(self):
        # Test TMatrix2D determinant
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = determinant_t(matrix)
        expected = -2.0
        self.assertAlmostEqual(result.get(), expected)

    def test_trace_t(self):
        # Test TMatrix2D trace
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = trace_t(matrix)
        expected = 5.0
        self.assertAlmostEqual(result.get(), expected)

    def test_rotate_point_t(self):
        # Test rotating a point with a TMatrix2D
        point = Point2D(1, 0)
        rotation_matrix = Matrix2D(np.array([[0, -1], [1, 0]]))  # 90 degrees rotation matrix
        result = rotate_point_t(rotation_matrix, point)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(result.get(), expected)


if __name__ == '__main__':
    unittest.main()
