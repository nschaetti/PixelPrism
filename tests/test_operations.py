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
        Scalar x Scalar
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
        Scalar x TScalar
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

    def test_scalar_op_tscalar_reverse(self):
        """
        Test operations between a scalar value and a TScalar (addition, subtraction, multiplication, division)
        Scalar x TScalar
        """
        # s1 = 5
        # t_s2 = 3
        s1 = Scalar(5)
        t_s2 = TScalar(lambda: 3)

        # Test addition
        result = t_s2 + s1
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8)

        # Test subtraction
        # 3 - 5 = -2
        result = t_s2 - s1
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), -2)

        # Test multiplication
        # 3 * 5 = 15
        result = t_s2 * s1
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 15)

        # Test division
        # 3 / 5
        result = t_s2 / s1
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 3 / 5)
    # end test_scalar_op_tscalar_reverse

    def test_scalar_op_point(self):
        """
        Test operations between a scalar value and a Point2D (addition, subtraction, multiplication, division)
        Scalar x Point2D
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

    def test_scalar_op_point_reverse(self):
        """
        Test operations between a scalar value and a Point2D (addition, subtraction, multiplication, division)
        Scalar x Point2D
        """
        # s = 2
        # p = (3, 4)
        s = Scalar(2)
        p = Point2D(3, 4)

        # Test addition
        # (3, 4) + 2 = (5, 6)
        result = p + s
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 6)

        # Test subtraction
        # (3, 4) - 2 = (1, 2)
        result = p - s
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)

        # Test multiplication
        # (3, 4) * 2 = (6, 8)
        result = p * s
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 8)

        # Test division
        # (3, 4) / 2 = (3/2, 4/2)
        result = p / s
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 1.5)
        self.assertEqual(result.y, 2.0)
    # end test_scalar_op_point_reverse

    def test_scalar_op_tpoint(self):
        """
        Test operations between a scalar value and a TPoint2D (addition, subtraction, multiplication, division)
        Scalar x TPoint2D
        """
        # 2 and (3, 4)
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

    def test_scalar_op_tpoint_reverse(self):
        """
        Test operations between a scalar value and a TPoint2D (addition, subtraction, multiplication, division)
        Scalar x TPoint2D
        """
        # s = 2
        # tp = (3, 4)
        s = Scalar(2)
        tp = TPoint2D(lambda: (3, 4))

        # Test addition
        # (3, 4) + 2 = (5, 6)
        result = tp + s
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 6)

        # Test subtraction
        # (3, 4) - 2 = (1, 2)
        result = tp - s
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)

        # Test multiplication
        # (3, 4) * 2 = (6, 8)
        result = tp * s
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 8)

        # Test division
        # (3, 4) / 2 = (3/2, 4/2)
        result = tp / s
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 3 / 2)
        self.assertEqual(result.y, 4 / 2)
    # end test_scalar_op_tpoint_reverse

    def test_scalar_op_matrix2d(self):
        """
        Test operations between a scalar value and a Matrix2D (addition, subtraction, multiplication, division)
        Scalar x Matrix2D
        """
        # 2, [[1, 2], [3, 4]]
        s = Scalar(2)
        m = Matrix2D([[1, 2], [3, 4]])

        # Test addition
        result = s + m
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[3, 4], [5, 6]])))

        # Test subtraction
        # 2 - [[1, 2], [3, 4]] = [[1, 0], [-1, -2]]
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

    def test_scalar_op_matrix2d_reverse(self):
        """
        Test operations between a scalar value and a Matrix2D (addition, subtraction, multiplication, division)
        Scalar x Matrix2D
        """
        # s = 2
        # m = [[1, 2], [3, 4]]
        s = Scalar(2)
        m = Matrix2D([[1, 2], [3, 4]])

        # Test addition
        # [[1, 2], [3, 4]] + 2 = [[3, 4], [5, 6]]
        result = m + s
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[3, 4], [5, 6]])))

        # Test subtraction
        # [[1, 2], [3, 4]] - 2 = [[-1, 0], [1, 2]]
        result = m - s
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[-1, 0], [1, 2]])))

        # Test multiplication
        # [[1, 2], [3, 4]] * 2 = [[2, 4], [6, 8]]
        result = m * s
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[2, 4], [6, 8]])))

        # Test division
        # [[1, 2], [3, 4]] / 2 = [[2/1, 2/2], [2/3, 2/4]]
        result = m / s
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.allclose(result.get(), np.array([[1 / 2, 2 / 2], [3 / 2, 4 / 2]])))
    # end test_scalar_op_matrix2d_reverse

    def test_scalar_op_tmatrix2d(self):
        """
        Test operations between a scalar value and a TMatrix2D (addition, subtraction, multiplication, division)
        Scalar x TMatrix2D
        """
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

    def test_scalar_op_tmatrix2d_reverse(self):
        """
        Test operations between a scalar value and a TMatrix2D (addition, subtraction, multiplication, division)
        TMatrix2D x Scalar
        """
        s = Scalar(2)
        tm = TMatrix2D(lambda: np.array([[1, 2], [3, 4]]))

        # Test addition
        result = tm + s
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.data, np.array([[3, 4], [5, 6]])))

        # Test subtraction
        # [[1, 2], [3, 4]] - 2 = [[-1, 0], [1, 2]]
        result = tm - s
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.data, np.array([[-1, 0], [1, 2]])))

        # Test multiplication
        result = tm * s
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.data, np.array([[2, 4], [6, 8]])))

        # Test division
        # [[1, 2], [3, 4]] / 2 = [[2/1, 2/2], [2/3, 2/4]]
        result = tm / s
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.allclose(result.data, np.array([[1 / 2, 2 / 2], [3 / 2, 4 / 2]])))
    # end test_scalar_op_tmatrix2d_reverse

    def test_scalar_plus_float(self):
        s = Scalar(5)
        result = s + 3.0
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.get(), 8.0)
    # end test_scalar_plus_float

    def test_scalar_plus_tscalar(self):
        s = Scalar(5)
        ts = TScalar(lambda: 3)
        result = s + ts
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8)
    # end test_scalar_plus_tscalar

    def test_scalar_plus_scalar(self):
        s1 = Scalar(5)
        s2 = Scalar(3)
        result = s1 + s2
        self.assertIsInstance(result, Scalar)
        self.assertEqual(result.get(), 8)
    # end test_scalar_plus_scalar

    def test_scalar_plus_tpoint2d(self):
        s = Scalar(5)
        tp = TPoint2D(lambda: (1, 2))
        result = s + tp
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 7)
    # end test_scalar_plus_tpoint2d

    def test_scalar_plus_point2d(self):
        s = Scalar(5)
        p = Point2D(1, 2)
        result = s + p
        self.assertIsInstance(result, Point2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 7)
    # end test_scalar_plus_point2d

    def test_scalar_plus_tmatrix2d(self):
        s = Scalar(5)
        tm = TMatrix2D(lambda: np.array([[1, 2], [3, 4]]))
        result = s + tm
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[6, 7], [8, 9]])))
    # end test_scalar_plus_tmatrix2d

    def test_scalar_plus_matrix2d(self):
        s = Scalar(5)
        m = Matrix2D(np.array([[1, 2], [3, 4]]))
        result = s + m
        self.assertIsInstance(result, Matrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[6, 7], [8, 9]])))
    # end test_scalar_plus_matrix2d

    def test_tscalar_plus_scalar(self):
        ts = TScalar(lambda: 5)
        s = Scalar(3)
        result = ts + s
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8)
    # end test_tscalar_plus_scalar

    def test_tscalar_plus_tscalar(self):
        ts1 = TScalar(lambda: 5)
        ts2 = TScalar(lambda: 3)
        result = ts1 + ts2
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8)
    # end test_tscalar_plus_tscalar

    def test_tscalar_plus_float(self):
        ts = TScalar(lambda: 5)
        result = ts + 3.0
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8.0)
    # end test_tscalar_plus_float

    def test_tscalar_plus_int(self):
        ts = TScalar(lambda: 5)
        result = ts + 3
        self.assertIsInstance(result, TScalar)
        self.assertEqual(result.get(), 8)
    # end test_tscalar_plus_int

    def test_tscalar_plus_tpoint2d(self):
        ts = TScalar(lambda: 5)
        tp = TPoint2D(lambda: (1, 2))
        result = ts + tp
        self.assertIsInstance(result, TPoint2D)
        self.assertEqual(result.x, 6)
        self.assertEqual(result.y, 7)
    # end test_tscalar_plus_tpoint2d

    def test_tscalar_plus_tmatrix2d(self):
        ts = TScalar(lambda: 5)
        tm = TMatrix2D(lambda: np.array([[1, 2], [3, 4]]))
        result = ts + tm
        self.assertIsInstance(result, TMatrix2D)
        self.assertTrue(np.array_equal(result.get(), np.array([[6, 7], [8, 9]])))
    # end test_tscalar_plus_tmatrix2d

    def test_scalar_plus_string(self):
        s = Scalar(5)
        with self.assertRaises(TypeError):
            result = s + "string"
        # end with
    # end test_scalar_plus_string

    def test_tscalar_plus_string(self):
        ts = TScalar(lambda: 5)
        with self.assertRaises(TypeError):
            result = ts + "string"
        # end with
    # end test_tscalar_plus_string

    def test_scalar_plus_string_reverse(self):
        s = Scalar(5)
        with self.assertRaises(TypeError):
            result = "string" + s
        # end with
    # end test_scalar_plus_string_reverse

    def test_tscalar_plus_string_reverse(self):
        ts = TScalar(lambda: 5)
        with self.assertRaises(TypeError):
            result = "string" + ts
        # end with
    # end test_tscalar_plus_string_reverse

# end TestOperations


# Run the tests
if __name__ == '__main__':
    unittest.main()
