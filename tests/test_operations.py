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


from pixel_prism.data import Scalar, TScalar, Point2D, TPoint2D, Matrix2D, TMatrix2D


class TestOperations(unittest.TestCase):

    def test_scalar_op_scalar(self):
        """
        Test operations between two scalar values (addition, subtraction, multiplication, division)
        """
        s1 = Scalar(5)
        s2 = Scalar(3)

        # Test addition
        result = s1 + s2
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.get(), 8)

        # Test subtraction
        result = s1 - s2
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.get(), 2)

        # Test multiplication
        result = s1 * s2
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.get(), 15)

        # Test division
        result = s1 / s2
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.get(), 5 / 3)
    # end test_scalar_op_scalar

    def test_scalar_op_tscalar(self):
        """
        Test operations between a scalar value and a TScalar (addition, subtraction, multiplication, division)
        """
        s1 = Scalar(5)
        t_s2 = TScalar(lambda: 3)

        # Test addition
        result = s1 + t_s2
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8)

        # Test subtraction
        result = s1 - t_s2
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 2)

        # Test multiplication
        result = s1 * t_s2
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 15)

        # Test division
        result = s1 / t_s2
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 5 / 3)
    # end test_scalar_op_tscalar

    def test_scalar_op_point(self):
        """
        Test operations between a scalar value and a Point2D (addition, subtraction, multiplication, division)
        """
        s = Scalar(2)
        p = Point2D(3, 4)

        # Test addition
        result = s + p
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 6)

        # Test subtraction
        result = s - p
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, -1)
        self.assertEqual(result.y, -2)

        # Test multiplication
        result = s * p
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 8)

        # Test division
        result = s / p
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 2 / 3)
        self.assertEqual(result.y, 2 / 4)

    # end test_scalar_op_point

    def test_scalar_op_tpoint(self):
        s = Scalar(2)
        tp = TPoint2D(lambda: (3, 4))

        # Test addition
        result = s + tp
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 6)

        # Test subtraction
        result = s - tp
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, -1)
        self.assertEqual(result.y, -2)

        # Test multiplication
        result = s * tp
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 8)

        # Test division
        result = s / tp
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 2 / 3)
        self.assertEqual(result.y, 2 / 4)

    # end test_scalar_op_tpoint

    def test_scalar_op_matrix2d(self):
        s = Scalar(2)
        m = Matrix2D([[1, 2], [3, 4]])

        # Test addition
        result = s + m
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[3, 4], [5, 6]])))

        # Test subtraction
        result = s - m
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[1, 0], [-1, -2]])))

        # Test multiplication
        result = s * m
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[2, 4], [6, 8]])))

        # Test division
        result = s / m
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.allclose(result.get(), np.array([[2 / 1, 2 / 2], [2 / 3, 2 / 4]])))

    # end test_scalar_op_matrix2d

    def test_scalar_op_tmatrix2d(self):
        s = Scalar(2)
        tm = TMatrix2D(lambda: np.array([[1, 2], [3, 4]]))

        # Test addition
        result = s + tm
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[3, 4], [5, 6]])))

        # Test subtraction
        result = s - tm
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[1, 0], [-1, -2]])))

        # Test multiplication
        result = s * tm
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[2, 4], [6, 8]])))

        # Test division
        result = s / tm
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.allclose(result.get(), np.array([[2 / 1, 2 / 2], [2 / 3, 2 / 4]])))
    # end test_scalar_op_tmatrix2d


# Run the tests
if __name__ == '__main__':
    unittest.main()
