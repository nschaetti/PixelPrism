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
    Matrix2D, TMatrix2D, scalar_mul_t, transpose_t,
    inverse_t, determinant_t, trace_t
)
from pixel_prism.data.matrices import (
    add_t, sub_t, mul_t
)
from pixel_prism.data import Point2D, Scalar, TScalar


class TestMatrix2D(unittest.TestCase):
    """
    Test the Matrix2D class
    """

    def test_matrix_initialization(self):
        """
        Test matrix initialization with a 2D array.
        """
        # Test matrix initialization
        matrix = Matrix2D([[1, 2], [3, 4]])
        self.assertTrue(np.array_equal(matrix.get(), np.array([[1, 2], [3, 4]])))
    # end test_matrix_initialization

    def test_matrix_addition(self):
        """
        Test matrix addition with another matrix.
        """
        # Test matrix addition
        matrix1 = Matrix2D([[1, 2], [3, 4]])
        matrix2 = Matrix2D([[5, 6], [7, 8]])
        result = matrix1 + matrix2
        expected = Matrix2D([[6, 8], [10, 12]])
        self.assertTrue(np.array_equal(result.get(), expected.get()))
    # end test_matrix_addition

    def test_set_get(self):
        """
        Test setting and getting the matrix
        """
        # Test setting and getting the matrix
        matrix = Matrix2D()
        new_matrix = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]])
        matrix.set(new_matrix)
        np.testing.assert_array_equal(matrix.get(), new_matrix)
    # end test_set_get

    def test_copy(self):
        """
        Test the copy method
        """
        # Test the copy method
        matrix = Matrix2D(np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]))
        matrix_copy = matrix.copy()
        np.testing.assert_array_equal(matrix.get(), matrix_copy.get())
        self.assertIsNot(matrix, matrix_copy)
    # end test_copy

    def test_matrix_subtraction(self):
        """
        Test matrix subtraction with another matrix.
        """
        # Test matrix subtraction
        matrix1 = Matrix2D([[5, 6], [7, 8]])
        matrix2 = Matrix2D([[1, 2], [3, 4]])
        result = matrix1 - matrix2
        expected = Matrix2D([[4, 4], [4, 4]])
        self.assertTrue(np.array_equal(result.get(), expected.get()))
    # end test_matrix_subtraction

    def test_addition(self):
        """
        Test addition of two matrices
        """
        # Test addition of two matrices
        matrix1 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        matrix2 = Matrix2D(np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]))
        result = matrix1 + matrix2
        expected = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_addition

    def test_subtraction(self):
        """
        Test subtraction of two matrices
        """
        # Test subtraction of two matrices
        matrix1 = Matrix2D(np.array([[10, 9, 8], [7, 6, 5], [4, 3, 2]]))
        matrix2 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        result = matrix1 - matrix2
        expected = np.array([[9, 7, 5], [3, 1, -1], [-3, -5, -7]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_subtraction

    def test_matrix_multiplication(self):
        """
        Test matrix multiplication with another matrix.
        """
        # Test matrix multiplication
        matrix1 = Matrix2D([[1, 2], [3, 4]])
        matrix2 = Matrix2D([[5, 6], [7, 8]])
        result = matrix1 @ matrix2
        expected = Matrix2D([[19, 22], [43, 50]])
        self.assertTrue(np.array_equal(result.get(), expected.get()))
    # end test_matrix_multiplication

    def test_scalar_multiplication(self):
        """
        Test matrix scalar multiplication by a scalar.
        """
        # Test matrix scalar multiplication
        matrix = Matrix2D([[1, 2], [3, 4]])
        result = matrix * 2
        expected = Matrix2D([[2, 4], [6, 8]])
        self.assertTrue(np.array_equal(result.get(), expected.get()))
    # end test_scalar_multiplication

    def test_division(self):
        """
        Test division of a matrix by a scalar
        """
        # Test division of a matrix by a scalar
        matrix = Matrix2D(np.array([[2, 4], [6, 8]]))
        result = matrix / 2
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_division

    def test_transpose(self):
        """
        Test matrix transpose method of a matrix.
        """
        # Test matrix transpose
        matrix = Matrix2D([[1, 2], [3, 4]])
        result = transpose_t(matrix)
        expected = Matrix2D([[1, 3], [2, 4]])
        self.assertTrue(np.array_equal(result.get(), expected.get()))
    # end test_transpose

    def test_equality(self):
        """
        Test equality of two matrices
        """
        # Test equality of two matrices
        matrix1 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        matrix2 = Matrix2D(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assertTrue(matrix1 == matrix2)
    # end test_equality

    def test_inverse(self):
        """
        Test the inverse of a matrix
        """
        # Test the inverse of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = inverse_t(matrix)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result.get(), expected)
    # end test_inverse

    def test_determinant(self):
        """
        Test the determinant of a matrix
        """
        # Test the determinant of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = determinant_t(matrix)
        expected = -2.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_determinant

    def test_trace(self):
        """
        Test the trace of a matrix
        """
        # Test the trace of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = trace_t(matrix)
        expected = 5.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_trace

# end TestMatrix2D


class TestTMatrix2D(unittest.TestCase):

    def test_tmatrix_initialization(self):
        """
        Test TMatrix2D initialization
        """
        # Test TMatrix2D initialization
        matrix1 = Matrix2D([[1, 2], [3, 4]])
        matrix2 = Matrix2D([[5, 6], [7, 8]])
        t_matrix = add_t(matrix1, matrix2)
        expected = Matrix2D([[6, 8], [10, 12]])
        self.assertTrue(np.array_equal(t_matrix.get(), expected.get()))
    # end test_tmatrix_initialization

    def test_tmatrix_dynamic_update(self):
        """
        Test dynamic update in TMatrix
        """
        # Test dynamic update in TMatrix2D
        matrix1 = Matrix2D([[1, 2], [3, 4]])
        matrix2 = Matrix2D([[5, 6], [7, 8]])
        t_matrix = add_t(matrix1, matrix2)

        # Initial check
        expected = Matrix2D([[6, 8], [10, 12]])
        self.assertTrue(np.array_equal(t_matrix.data, expected.data))

        # Update matrix1 and check update in t_matrix
        matrix1.set([[2, 3], [4, 5]])
        expected_after_update = Matrix2D([[7, 9], [11, 13]])
        self.assertTrue(np.array_equal(t_matrix.data, expected_after_update.data))
    # end test_tmatrix_dynamic_update

    def test_add_t(self):
        """
        Test TMatrix2D addition
        """
        # Test TMatrix2D addition
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[5, 6], [7, 8]]))

        result = add_t(matrix1, matrix2)

        expected = np.array([[6, 8], [10, 12]])

        np.testing.assert_array_equal(result.data, expected)

        # Modify one of the source matrices and check the result
        matrix1[0, 0] = 5
        expected_updated = np.array([[10, 8], [10, 12]])
        np.testing.assert_array_equal(result.get(), expected_updated)
    # end test_add_t

    def test_sub_t(self):
        """
        Test TMatrix2D subtraction
        """
        # Test TMatrix2D subtraction
        matrix1 = Matrix2D(np.array([[5, 6], [7, 8]]))
        matrix2 = Matrix2D(np.array([[1, 2], [3, 4]]))

        result = sub_t(matrix1, matrix2)

        expected = np.array([[4, 4], [4, 4]])

        np.testing.assert_array_equal(result.data, expected.data)

        # Modify one of the source matrices and check the result
        matrix1.data = np.array([[6, 6], [6, 6]])
        expected_updated = np.array([[5, 4], [3, 2]])
        np.testing.assert_array_equal(result.data, expected_updated.data)
    # end test_sub_t

    def test_mul_t(self):
        """
        Test TMatrix2D multiplication
        """
        # Test TMatrix2D multiplication
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[2, 0], [1, 2]]))
        result = mul_t(matrix1, matrix2)
        expected = np.array([[2, 0], [3, 8]])
        np.testing.assert_array_equal(result.data, expected.data)
    # end test_mul_t

    def test_scalar_mul_t(self):
        """
        Test TMatrix2D scalar multiplication
        """
        # Test TMatrix2D scalar multiplication
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        scalar = Scalar(2)
        result = scalar_mul_t(matrix, scalar)
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result.data, expected.data)
    # end test_scalar_mul_t

    def test_transpose_t(self):
        """
        Test TMatrix2D transpose
        """
        # Test TMatrix2D transpose
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = transpose_t(matrix)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_transpose_t

    def test_inverse_t(self):
        """
        Test TMatrix2D inverse
        """
        # Test TMatrix2D inverse
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = inverse_t(matrix)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result.get(), expected)
    # end test_inverse_t

    def test_determinant_t(self):
        """
        Test TMatrix2D determinant
        """
        # Test TMatrix2D determinant
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = determinant_t(matrix)
        expected = -2.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_determinant_t

    def test_trace_t(self):
        """
        Test TMatrix2D trace
        """
        # Test TMatrix2D trace
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = trace_t(matrix)
        expected = 5.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_trace_t

# end TestTMatrix2D


if __name__ == '__main__':
    unittest.main()
