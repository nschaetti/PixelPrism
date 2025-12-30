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
# Copyright (C) 2024 Pixel Prism
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
from pixelprism.math_old import (
    Matrix2D, TMatrix2D, EventType
)
from pixelprism.math_old import Scalar


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
        result = TMatrix2D.transpose(matrix)
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
        result = TMatrix2D.inverse(matrix)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result.get(), expected)
    # end test_inverse

    def test_determinant(self):
        """
        Test the determinant of a matrix
        """
        # Test the determinant of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = TMatrix2D.determinant(matrix)
        expected = -2.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_determinant

    def test_trace(self):
        """
        Test the trace of a matrix
        """
        # Test the trace of a matrix
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = TMatrix2D.trace(matrix)
        expected = 5.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_trace

    def test_on_change_set_matrix(self):
        """
        Test that the on_change event is triggered when the matrix is set.
        """
        changes = []

        def on_change(sender, event_type, matrix):
            changes.append((event_type, matrix))
        # end on_change

        # Initialize the Matrix2D with an on_change callback
        matrix = Matrix2D(matrix=np.identity(3), on_change=on_change)

        # Change the matrix
        new_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        matrix.data = new_matrix

        # Check if the on_change event was triggered with the correct matrix
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], EventType.MATRIX_CHANGED)  # Check event type
        np.testing.assert_array_equal(changes[0][1], new_matrix)  # Check matrix math_old
    # end test_on_change_set_matrix

    def test_on_change_modify_matrix_element(self):
        """
        Test that the on_change event is triggered when an element of the matrix is modified.
        """
        changes = []

        def on_change(sender, event_type, matrix):
            changes.append((event_type, matrix))
        # end on_change

        # Initialize the Matrix2D with an on_change callback
        matrix = Matrix2D(matrix=np.identity(3), on_change=on_change)

        # Modify an element of the matrix
        matrix[0, 0] = 5.0

        # Check if the on_change event was triggered with the updated matrix
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], EventType.MATRIX_CHANGED)  # Check event type
        self.assertEqual(matrix[0, 0], 5.0)  # Check that the element was updated
    # end test_on_change_modify_matrix_element

    def test_on_change_addition(self):
        """
        Test that the on_change event is triggered when matrices are added.
        """
        changes = []

        def on_change(sender, event_type, matrix):
            changes.append((event_type, matrix))
        # end on_change

        # Initialize two Matrix2D objects with an on_change callback
        matrix1 = Matrix2D(matrix=np.identity(3), on_change=on_change)
        matrix2 = Matrix2D(matrix=np.ones((3, 3)), on_change=on_change)

        # Add the two matrices
        result_matrix = matrix1 + matrix2

        # Since adding matrices does not modify the original matrices, we expect no on_change event
        self.assertEqual(len(changes), 0)

        # Now update one of the matrices and check for the event
        matrix1.data = matrix1.data + matrix2.data

        # Check if the on_change event was triggered
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], EventType.MATRIX_CHANGED)  # Check event type
        np.testing.assert_array_equal(matrix1.data, np.identity(3) + np.ones((3, 3)))  # Check new matrix math_old
    # end test_on_change_addition

# end TestMatrix2D


class TestTMatrix2D(unittest.TestCase):

    def test_tmatrix_initialization(self):
        """
        Test TMatrix2D initialization
        """
        # Test TMatrix2D initialization
        matrix1 = Matrix2D([[1, 2], [3, 4]])
        matrix2 = Matrix2D([[5, 6], [7, 8]])
        t_matrix = TMatrix2D.add(matrix1, matrix2)
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
        t_matrix = TMatrix2D.add(matrix1, matrix2)

        # Initial check
        expected = Matrix2D([[6, 8], [10, 12]])
        self.assertTrue(np.array_equal(t_matrix.data, expected.data))

        # Update matrix1 and check update in t_matrix
        matrix1.set([[2, 3], [4, 5]])
        expected_after_update = Matrix2D([[7, 9], [11, 13]])
        self.assertTrue(np.array_equal(t_matrix.data, expected_after_update.data))
    # end test_tmatrix_dynamic_update

    def test_add(self):
        """
        Test TMatrix2D addition
        """
        # Test TMatrix2D addition
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[5, 6], [7, 8]]))

        result = TMatrix2D.add(matrix1, matrix2)

        expected = np.array([[6, 8], [10, 12]])

        np.testing.assert_array_equal(result.data, expected)

        # Modify one of the source matrices and check the result
        matrix1[0, 0] = 5
        expected_updated = np.array([[10, 8], [10, 12]])
        np.testing.assert_array_equal(result.get(), expected_updated)
    # end test_add

    def test_sub(self):
        """
        Test TMatrix2D subtraction
        """
        # Test TMatrix2D subtraction
        matrix1 = Matrix2D(np.array([[5, 6], [7, 8]]))
        matrix2 = Matrix2D(np.array([[1, 2], [3, 4]]))

        result = TMatrix2D.sub(matrix1, matrix2)

        expected = np.array([[4, 4], [4, 4]])

        np.testing.assert_array_equal(result.data, expected.data)

        # Modify one of the source matrices and check the result
        matrix1.data = np.array([[6, 6], [6, 6]])
        expected_updated = np.array([[5, 4], [3, 2]])
        np.testing.assert_array_equal(result.data, expected_updated.data)
    # end test_sub

    def test_mul(self):
        """
        Test TMatrix2D multiplication
        """
        # Test TMatrix2D multiplication
        matrix1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        matrix2 = Matrix2D(np.array([[2, 0], [1, 2]]))
        result = TMatrix2D.mul(matrix1, matrix2)
        expected = np.array([[2, 0], [3, 8]])
        np.testing.assert_array_equal(result.data, expected.data)
    # end test_mul

    def test_scalar_mul(self):
        """
        Test TMatrix2D scalar multiplication
        """
        # Test TMatrix2D scalar multiplication
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        scalar = Scalar(2)
        result = TMatrix2D.scalar_mul(matrix, scalar)
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result.data, expected.data)
    # end test_scalar_mul

    def test_transpose(self):
        """
        Test TMatrix2D transpose
        """
        # Test TMatrix2D transpose
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = TMatrix2D.transpose(matrix)
        expected = np.array([[1, 3], [2, 4]])
        np.testing.assert_array_equal(result.get(), expected)
    # end test_transpose

    def test_inverse(self):
        """
        Test TMatrix2D inverse
        """
        # Test TMatrix2D inverse
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = TMatrix2D.inverse(matrix)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(result.get(), expected)
    # end test_inverse

    def test_determinant(self):
        """
        Test TMatrix2D determinant
        """
        # Test TMatrix2D determinant
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = TMatrix2D.determinant(matrix)
        expected = -2.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_determinant

    def test_trace(self):
        """
        Test TMatrix2D trace
        """
        # Test TMatrix2D trace
        matrix = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = TMatrix2D.trace(matrix)
        expected = 5.0
        self.assertAlmostEqual(result.get(), expected)
    # end test_trace

    def test_on_change_sources(self):
        """
        Test that the on_change event is triggered when one of the source matrices is changed.
        """
        changes = []

        def on_change(sender, event_type, matrix):
            changes.append((event_type, matrix))
        # end on_change

        # Create two source matrices
        matrix1 = Matrix2D(matrix=np.identity(3))
        matrix2 = Matrix2D(matrix=np.ones((3, 3)))

        # Create a TMatrix2D that adds the two matrices
        t_matrix = TMatrix2D(lambda m1, m2: m1.data + m2.data, on_change=on_change, m1=matrix1, m2=matrix2)

        # Modify matrix1
        matrix1.data = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])

        # Verify that the on_change event was triggered and matrix was updated
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], EventType.MATRIX_CHANGED)  # Check event type
        np.testing.assert_array_equal(t_matrix.data, matrix1.data + matrix2.data)  # Check computed matrix
    # end test_on_change_sources

    def test_on_change_scalar_in_matrix(self):
        """
        Test that the on_change event is triggered when a scalar in one of the source matrices is changed.
        """
        changes = []

        def on_change(sender, event_type, matrix):
            changes.append((event_type, matrix))
        # end on_change

        # Create a scalar and a matrix
        scalar = Scalar(2)
        matrix1 = Matrix2D(matrix=np.identity(3))

        # Create a TMatrix2D that multiplies the matrix by the scalar
        t_matrix = TMatrix2D(lambda m, s: m.data * s.value, on_change=on_change, m=matrix1, s=scalar)

        # Modify the scalar
        scalar.value = 3

        # Verify that the on_change event was triggered and matrix was updated
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0][0], EventType.MATRIX_CHANGED)  # Check event type
        np.testing.assert_array_equal(t_matrix.data, matrix1.data * scalar.value)  # Check computed matrix
    # end test_on_change_scalar_in_matrix

    def test_on_change_chained_matrices(self):
        """
        Test that the on_change event is triggered when one of the source matrices in a chain of TMatrix2D objects is changed.
        """
        changes1 = []
        changes2 = []

        def on_change_tmatrix1(sender, event_type, matrix):
            changes1.append((event_type, matrix))
        # end on_change

        def on_change_tmatrix2(sender, event_type, matrix):
            changes2.append((event_type, matrix))
        # end on_change

        # Create three source matrices
        # [[1, 0], [0, 1]]
        # [[1, 1], [1, 1]]
        matrix1 = Matrix2D(matrix=np.identity(2))
        matrix2 = Matrix2D(matrix=np.ones((2, 2)))

        # Create a first TMatrix2D that adds the two matrices
        # [[2, 1], [1, 2]]
        t_matrix1 = TMatrix2D(lambda m1, m2: m1.data + m2.data, on_change=on_change_tmatrix1, m1=matrix1, m2=matrix2)

        # Create a second TMatrix2D that multiplies the result by a scalar
        # 2 * [[2, 1], [1, 2]] = [[4, 2], [2, 4]]
        scalar = Scalar(2)
        t_matrix2 = TMatrix2D(lambda m, s: m.data * s.value, on_change=on_change_tmatrix2, m=t_matrix1, s=scalar)

        self.assertEqual(len(changes1), 0)
        self.assertEqual(len(changes2), 0)
        np.testing.assert_array_equal(t_matrix2.data, np.array([[4, 2], [2, 4]]))  # Check computed matrix

        # Modify matrix1
        matrix1.data = np.array([[2, 2], [2, 2]])

        # Verify that the on_change event was triggered and matrix was updated in the second TMatrix2D
        self.assertEqual(len(changes1), 1)
        self.assertEqual(len(changes2), 1)
        self.assertEqual(changes1[0][0], EventType.MATRIX_CHANGED)  # Check event type
        self.assertEqual(changes2[0][0], EventType.MATRIX_CHANGED)  # Check event type
        np.testing.assert_array_equal(t_matrix2.data, (matrix1.data + matrix2.data) * scalar.value)  # Check computed matrix
    # end test_on_change_chained_matrices

# end TestTMatrix2D


if __name__ == '__main__':
    unittest.main()
# end if
