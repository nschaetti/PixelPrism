"""
Test the functional operations in pixel_prism.data.functional.
"""

import unittest
import numpy as np

from pixel_prism.data.scalar import Scalar
from pixel_prism.data.points import Point2D
from pixel_prism.data.matrices import Matrix2D
import pixel_prism.data.functional as F


class TestFunctional(unittest.TestCase):
    """
    Test the functional operations in pixel_prism.data.functional.
    """

    def test_scalar_add(self):
        """
        Test adding two scalars.
        """
        a = Scalar(1)
        b = Scalar(2)
        c = F.add(a, b)
        self.assertEqual(c.value, 3)

    def test_scalar_sub(self):
        """
        Test subtracting two scalars.
        """
        a = Scalar(3)
        b = Scalar(1)
        c = F.sub(a, b)
        self.assertEqual(c.value, 2)

    def test_scalar_mul(self):
        """
        Test multiplying two scalars.
        """
        a = Scalar(2)
        b = Scalar(3)
        c = F.mul(a, b)
        self.assertEqual(c.value, 6)

    def test_scalar_div(self):
        """
        Test dividing two scalars.
        """
        a = Scalar(6)
        b = Scalar(2)
        c = F.div(a, b)
        self.assertEqual(c.value, 3)

    def test_scalar_floor(self):
        """
        Test applying the floor function to a scalar.
        """
        a = Scalar(3.7)
        b = F.floor(a)
        self.assertEqual(b.value, 3)

    def test_scalar_ceil(self):
        """
        Test applying the ceiling function to a scalar.
        """
        a = Scalar(3.2)
        b = F.ceil(a)
        self.assertEqual(b.value, 4)

    def test_scalar_abs(self):
        """
        Test applying the absolute value function to a scalar.
        """
        a = Scalar(-3)
        b = F.abs(a)
        self.assertEqual(b.value, 3)

    def test_scalar_neg(self):
        """
        Test applying the negation function to a scalar.
        """
        a = Scalar(3)
        b = F.neg(a)
        self.assertEqual(b.value, -3)

    def test_point2d_add(self):
        """
        Test adding two points.
        """
        a = Point2D(1, 2)
        b = Point2D(3, 4)
        c = F.add(a, b)
        self.assertEqual(c.x, 4)
        self.assertEqual(c.y, 6)

    def test_point2d_sub(self):
        """
        Test subtracting two points.
        """
        a = Point2D(3, 4)
        b = Point2D(1, 2)
        c = F.sub(a, b)
        self.assertEqual(c.x, 2)
        self.assertEqual(c.y, 2)

    def test_point2d_mul(self):
        """
        Test multiplying two points.
        """
        a = Point2D(2, 3)
        b = Point2D(3, 4)
        c = F.mul(a, b)
        self.assertEqual(c.x, 6)
        self.assertEqual(c.y, 12)

    def test_point2d_scalar_mul(self):
        """
        Test multiplying a point by a scalar.
        """
        a = Point2D(2, 3)
        b = Scalar(2)
        c = F.mul(a, b)
        self.assertEqual(c.x, 4)
        self.assertEqual(c.y, 6)

    def test_point2d_div(self):
        """
        Test dividing a point by a scalar.
        """
        a = Point2D(4, 6)
        b = Scalar(2)
        c = F.div(a, b)
        self.assertEqual(c.x, 2)
        self.assertEqual(c.y, 3)

    def test_matrix2d_add(self):
        """
        Test adding two matrices.
        """
        a = Matrix2D(np.array([[1, 2], [3, 4]]))
        b = Matrix2D(np.array([[5, 6], [7, 8]]))
        c = F.add(a, b)
        np.testing.assert_array_equal(c.data, np.array([[6, 8], [10, 12]]))

    def test_matrix2d_sub(self):
        """
        Test subtracting two matrices.
        """
        a = Matrix2D(np.array([[5, 6], [7, 8]]))
        b = Matrix2D(np.array([[1, 2], [3, 4]]))
        c = F.sub(a, b)
        np.testing.assert_array_equal(c.data, np.array([[4, 4], [4, 4]]))

    def test_matrix2d_mul(self):
        """
        Test multiplying two matrices element-wise.
        """
        a = Matrix2D(np.array([[1, 2], [3, 4]]))
        b = Matrix2D(np.array([[5, 6], [7, 8]]))
        c = F.mul(a, b)
        np.testing.assert_array_equal(c.data, np.array([[5, 12], [21, 32]]))

    def test_matrix2d_scalar_mul(self):
        """
        Test multiplying a matrix by a scalar.
        """
        a = Matrix2D(np.array([[1, 2], [3, 4]]))
        b = Scalar(2)
        c = F.mul(a, b)
        np.testing.assert_array_equal(c.data, np.array([[2, 4], [6, 8]]))

    def test_matrix2d_mm(self):
        """
        Test matrix-matrix multiplication.
        """
        a = Matrix2D(np.array([[1, 2], [3, 4]]))
        b = Matrix2D(np.array([[5, 6], [7, 8]]))
        c = F.mm(a, b)
        np.testing.assert_array_equal(c.data, np.array([[19, 22], [43, 50]]))

    def test_matrix2d_transpose(self):
        """
        Test transposing a matrix.
        """
        a = Matrix2D(np.array([[1, 2], [3, 4]]))
        b = F.transpose(a)
        np.testing.assert_array_equal(b.data, np.array([[1, 3], [2, 4]]))

    def test_matrix2d_inverse(self):
        """
        Test inverting a matrix.
        """
        a = Matrix2D(np.array([[1, 2], [3, 4]]))
        b = F.inverse(a)
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(b.data, expected)


if __name__ == "__main__":
    unittest.main()