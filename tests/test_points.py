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
    Point2D, TPoint2D, Scalar, add_t, sub_t, mul_t, div_t, rotate_t, scale_t, dot_t,
    cross_t, norm_t, normalize_t, angle_t, distance_t, distance_squared_t,
    distance_manhattan_t, distance_chebyshev_t, distance_canberra_t, distance_minkowski_t,
    distance_hamming_t, distance_jaccard_t, distance_braycurtis_t,
    distance_cosine_t, distance_correlation_t,
    distance_euclidean_t, distance_mahalanobis_t, distance_seuclidean_t,
    distance_sqeuclidean_t
)


class TestPoint2D(unittest.TestCase):

    def test_set_and_get(self):
        """
        Test that the point can be set and get correctly.
        """
        # Test 1
        point = Point2D(1, 2)
        self.assertEqual(point.get(), (1, 2))

        # Test 2
        point.set(3, 4)
        self.assertEqual(point.get(), (3, 4))
    # end test_set_and_get

    def test_on_change(self):
        """
        Test that the on_change event is triggered when the point is changed.
        """
        changes = []

        def on_change(event):
            changes.append((event.x, event.y))
        # end on_change

        # Test set
        point = Point2D(1, 2, on_change=on_change)
        point.set(3, 4)
        self.assertEqual(changes, [(3, 4)])
    # end test_on_change

    def test_addition(self):
        """
        Test that two points can be added together.
        """
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)
        result = p1 + p2
        self.assertEqual(result, Point2D(4, 6))
    # end test_addition

    def test_addition_scalar(self):
        """
        Test that a scalar can be added to a point.
        """
        p1 = Point2D(1, 2)
        result = p1 + 5
        self.assertEqual(result, Point2D(6, 7))
    # end test_addition_scalar

    def test_subtraction(self):
        """
        Test that two points can be subtracted.
        """
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)
        result = p1 - p2
        self.assertEqual(result, Point2D(-2, -2))
    # end test_subtraction

    def test_subtraction_scalar(self):
        """
        Test that a scalar can be subtracted from a point.
        """
        p1 = Point2D(1, 2)
        result = p1 - 1
        self.assertEqual(result, Point2D(0, 1))
    # end test_subtraction_scalar

    def test_multiplication(self):
        """
        Test that a point can be multiplied by a scalar.
        """
        p1 = Point2D(1, 2)
        result = p1 * 2
        self.assertEqual(result, Point2D(2, 4))
    # end test_multiplication

    def test_division(self):
        """
        Test that a point can be divided by a scalar.
        """
        p1 = Point2D(4, 6)
        result = p1 / 2
        self.assertEqual(result, Point2D(2, 3))
    # end test_division

    def test_equality(self):
        """
        Test that two points can be compared for equality.
        """
        p1 = Point2D(1, 2)
        p2 = Point2D(1, 2)
        self.assertTrue(p1 == p2)
    # end test_equality

    def test_initialization(self):
        point = Point2D(3, 4)
        self.assertEqual(point.x, 3)
        self.assertEqual(point.y, 4)
    # end test_initialization

    def test_set_and_get2(self):
        point = Point2D()
        point.set(5, 6)
        self.assertEqual(point.x, 5)
        self.assertEqual(point.y, 6)
    # end test_set_and_get2

    def test_copy(self):
        point = Point2D(7, 8)
        point_copy = point.copy()
        self.assertEqual(point_copy.x, point.x)
        self.assertEqual(point_copy.y, point.y)
    # end test_copy

    def test_addition2(self):
        point1 = Point2D(3, 4)
        point2 = Point2D(1, 2)
        result = point1 + point2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 6)

        result = point1 + 2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 6)

        result = point1 + (1, 1)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 5)
    # end test_addition2

    def test_subtraction2(self):
        point1 = Point2D(5, 6)
        point2 = Point2D(1, 2)
        result = point1 - point2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 4)

        result = point1 - 2
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 4)

        result = point1 - (1, 1)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 5)
    # end test_subtraction2

    def test_multiplication2(self):
        point = Point2D(2, 3)
        result = point * 2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 6)

        result = point * Point2D(2, 3)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 9)
    # end test_multiplication2

    def test_division2(self):
        point = Point2D(4, 6)
        result = point / 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 3)

        result = point / Point2D(2, 3)
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 2)
    # end test_division2

    def test_equality2(self):
        point1 = Point2D(3, 4)
        point2 = Point2D(3, 4)
        point3 = Point2D(4, 3)
        self.assertTrue(point1 == point2)
        self.assertFalse(point1 == point3)
    # end test_equality2

    def test_norm2(self):
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)
        self.assertEqual(point1.norm2(point2), 5)
    # end test_norm2

    def test_translate(self):
        point = Point2D(1, 1)
        point.translate_(Point2D(3, 4))
        self.assertEqual(point.x, 4)
        self.assertEqual(point.y, 5)
    # end test_translate

    def test_rotate(self):
        point = Point2D(1, 0)
        point.rotate_(np.pi / 2)
        self.assertAlmostEqual(point.x, 0, places=5)
        self.assertAlmostEqual(point.y, 1, places=5)
    # end test_rotate

    def test_scale(self):
        point = Point2D(2, 3)
        point.scale_(2)
        self.assertEqual(point.x, 4)
        self.assertEqual(point.y, 6)
    # end test_scale

    def test_event_trigger(self):
        changes = []

        def on_change(event):
            changes.append((event.params['x'], event.params['y']))
        # end on_change

        point = Point2D(1, 2, on_change=on_change)
        point.x = 3
        point.y = 4

        self.assertEqual(changes, [(3, 2), (3, 4)])
    # end test_event_trigger

    def test_event_trigger_tpoint2d(self):
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        tpoint = add_t(point1, point2)

        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        point1.x = 5

        self.assertEqual(tpoint.x, 8)
        self.assertEqual(tpoint.y, 6)
    # end test_event_trigger_tpoint2d

    def test_tpoint2d_restriction(self):
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        tpoint = add_t(point1, point2)

        with self.assertRaises(AttributeError):
            tpoint.x = 10
        # end with
    # end test_tpoint2d_restriction

    def test_add_t(self):
        """
        Test that two points can be added together.
        """
        # 1, 2
        # 3, 4
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        tpoint = add_t(point1, point2)

        # 4 = 1 + 3
        # 6 = 2 + 4
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        # Modify one of the sources and check the updated result
        # 2, 2
        # 3, 4
        point1.x = 2

        # 5 = 2 + 3
        # 6 = 2 + 4
        self.assertEqual(tpoint.x, 5)
        self.assertEqual(tpoint.y, 6)

        point2.y = 5
        # 2, 2
        # 3, 5
        # 5 = 2 + 3
        # 7 = 2 + 5
        self.assertEqual(tpoint.x, 5)
        self.assertEqual(tpoint.y, 7)
    # end test_add_t

    def test_sub_t(self):
        """
        Test that two points can be subtracted.
        """
        # 5, 7
        # 2, 3
        point1 = Point2D(5, 7)
        point2 = Point2D(2, 3)
        tpoint = sub_t(point1, point2)

        # 3 = 5 - 2
        # 4 = 7 - 3
        self.assertEqual(tpoint.x, 3)
        self.assertEqual(tpoint.y, 4)

        # Modify one of the sources and check the updated result
        point1.y = 8
        # 5, 8
        # 2, 3
        # 3 = 5 - 2
        # 5 = 8 - 3
        self.assertEqual(tpoint.x, 3)
        self.assertEqual(tpoint.y, 5)

        point2.x = 1
        # 5, 8
        # 1, 3
        # 4 = 5 - 1
        # 5 = 8 - 3
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 5)
    # end test_sub_t

    def test_mul_t(self):
        """
        Test that a point can be multiplied by a scalar.
        """
        # 2, 3
        point1 = Point2D(2, 3)
        scalar = Scalar(2)
        tpoint = mul_t(point1, scalar)

        # 4 = 2 * 2
        # 6 = 3 * 2
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        # Modify the scalar and check the updated result
        scalar.value = 3
        # 6 = 2 * 3
        # 9 = 3 * 3
        self.assertEqual(tpoint.x, 6)
        self.assertEqual(tpoint.y, 9)

        # Modify the point and check the updated result
        # 4, 3
        point1.x = 4

        # 12 = 4 * 3
        # 9 = 3 * 3
        self.assertEqual(tpoint.x, 12)
        self.assertEqual(tpoint.y, 9)
    # end test_mul_t

    def test_mul_t2(self):
        point1 = Point2D(2, 3)
        scalar = 2.0
        tpoint = mul_t(point1, scalar)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        # Modify the point and check the updated result
        point1.y = 4
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 8)
    # end test_mul_t2

    def test_div_t(self):
        point1 = Point2D(8, 6)
        scalar = Scalar(2)
        tpoint = div_t(point1, scalar)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 3)

        # Modify the scalar and check the updated result
        scalar.value = 4
        self.assertEqual(tpoint.x, 2)
        self.assertEqual(tpoint.y, 1.5)

        # Modify the point and check the updated result
        point1.x = 16
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 1.5)
    # end test_div_t

    def test_div_t2(self):
        point1 = Point2D(8, 6)
        scalar = 2.0
        tpoint = div_t(point1, scalar)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 3)

        # Modify the point and check the updated result
        point1.y = 12
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)
    # end test_div_t2

    def test_rotate_t(self):
        point = Point2D(1, 0)
        center = Point2D(0, 0)
        angle = Scalar(np.pi / 2)
        tpoint = rotate_t(point, angle, center)

        self.assertAlmostEqual(tpoint.x, 0, places=5)
        self.assertAlmostEqual(tpoint.y, 1, places=5)

        # Modify the angle and check the updated result
        angle.value = np.pi
        self.assertAlmostEqual(tpoint.x, -1, places=5)
        self.assertAlmostEqual(tpoint.y, 0, places=5)

        # Modify the point and check the updated result
        point.x = 0
        self.assertAlmostEqual(tpoint.x, 0, places=5)
        self.assertAlmostEqual(tpoint.y, 0, places=5)
    # end test_rotate_t

    def test_rotate_t2(self):
        point = Point2D(1, 0)
        center = Point2D(0, 0)
        angle = np.pi / 2.0
        tpoint = rotate_t(point, angle, center)

        self.assertAlmostEqual(tpoint.x, 0, places=5)
        self.assertAlmostEqual(tpoint.y, 1, places=5)

        # Modify the point and check the updated result
        point.y = 1
        self.assertAlmostEqual(tpoint.x, -1, places=5)
        self.assertAlmostEqual(tpoint.y, 1, places=5)
    # end test_rotate_t2

    def test_scale_t(self):
        point = Point2D(2, 3)
        center = Point2D(0, 0)
        scale = Scalar(2)
        tpoint = scale_t(point, scale, center)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        # Modify the scale and check the updated result
        scale.value = 3
        self.assertEqual(tpoint.x, 6)
        self.assertEqual(tpoint.y, 9)

        # Modify the point and check the updated result
        point.x = 4
        self.assertEqual(tpoint.x, 12)
        self.assertEqual(tpoint.y, 9)
    # end test_scale_t

    def test_dot_t(self):
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        result = dot_t(point1, point2)
        self.assertEqual(result.value, 11)

        # Modify one of the points and check the updated result
        point1.y = 3
        self.assertEqual(result.value, 15)
    # end test_dot_t

    def test_cross_t(self):
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        result = cross_t(point1, point2)
        self.assertEqual(result.value, -2)

        # Modify one of the points and check the updated result
        point2.y = 3
        self.assertEqual(result.value, -3)
    # end test_cross_t

    def test_norm_t(self):
        point = Point2D(3, 4)
        result = norm_t(point)
        self.assertEqual(result.value, 5)

        # Modify the point and check the updated result
        point.x = 6
        self.assertAlmostEqual(result.value, np.sqrt(36 + 16), places=5)
    # end test_norm_t

    def test_normalize_t(self):
        point = Point2D(3, 4)
        result = normalize_t(point)
        self.assertAlmostEqual(result.x, 0.6, places=5)
        self.assertAlmostEqual(result.y, 0.8, places=5)

        # Modify the point and check the updated result
        point.x = 0
        self.assertAlmostEqual(result.x, 0.0, places=5)
        self.assertAlmostEqual(result.y, 1.0, places=5)
    # end test_normalize_t

    def test_angle_t(self):
        """
        Test the angle between two points.
        """
        # 1, 0
        # 0, 1
        point1 = Point2D(1, 0)
        point2 = Point2D(0, 1)
        result = angle_t(point1, point2)

        # The angle between the two points is pi/2
        self.assertAlmostEqual(result.value, np.pi / 2, places=5)

        # Modify one of the points and check the updated result
        # 1, 0
        # -1, 0
        point2.x = -1
        point2.y = 0
        self.assertAlmostEqual(result.value, np.pi, places=5)
    # end test_angle_t

    def test_distance_t(self):
        """
        Test the distance between two points.
        """
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)
        result = distance_t(point1, point2)
        self.assertEqual(result.value, 5)

        # Modify one of the points and check the updated result
        point2.x = 6
        self.assertEqual(result.value, 7.211102550927978)
    # end test_distance_t

    def test_distance_squared_t(self):
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)
        result = distance_squared_t(point1, point2)
        self.assertEqual(result.value, 25)

        # Modify one of the points and check the updated result
        point1.x = 1
        self.assertEqual(result.value, 20)
    # end test_distance_squared_t

    # TODO: check that the value of the output
    def test_distance_manhattan_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_manhattan_t(point1, point2)
        self.assertEqual(result.value, 5)
    # end test_distance_manhattan_t

    # TODO: check that the value of the output
    def test_distance_chebyshev_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_chebyshev_t(point1, point2)
        self.assertEqual(result.value, 3)
    # end test_distance_chebyshev_t

    # TODO: check that the value of the output
    def test_distance_canberra_t(self):
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        result = distance_canberra_t(point1, point2)
        self.assertAlmostEqual(result.value, 0.8333, places=4)
    # end test_distance_canberra_t

    # TODO: check that the value of the output
    def test_distance_minkowski_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_minkowski_t(point1, point2, p=3)
        self.assertAlmostEqual(result.value, 3.2710664, places=4)
    # end test_distance_minkowski_t

    # TODO: check that the value of the output
    def test_distance_minkowski_t2(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_minkowski_t(point1, point2, p=Scalar(3))
        self.assertAlmostEqual(result.value, 3.2710664, places=4)
    # end test_distance_minkowski_t

    # TODO: check that the value of the output
    def test_distance_hamming_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(1, 4)
        result = distance_hamming_t(point1, point2)
        self.assertEqual(result.value, 1.0)
    # end test_distance_hamming_t

    # TODO: check that the value of the output
    def test_distance_jaccard_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(1, 4)
        result = distance_jaccard_t(point1, point2)
        self.assertEqual(result.value, 0.6)
    # end test_distance_jaccard_t

    # TODO: check that the value of the output
    def test_distance_braycurtis_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_braycurtis_t(point1, point2)
        self.assertEqual(result.value, 0.5555556)
    # end test_distance_braycurtis_t

    def test_distance_cosine_t(self):
        point1 = Point2D(1, 0)
        point2 = Point2D(0, 1)
        result = distance_cosine_t(point1, point2)
        self.assertEqual(result.value, 1)
    # end test_distance_cosine_t

    def test_distance_correlation_t(self):
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        result = distance_correlation_t(point1, point2)
        self.assertAlmostEqual(result.value, 0)
    # end test_distance_correlation_t

    # TODO: Does not work
    # def test_distance_haversine_t(self):
    #     point1 = Point2D(0, 0)
    #     point2 = Point2D(0, 90)
    #     result = distance_haversine_t(point1, point2)
    #     self.assertAlmostEqual(result.value, np.pi / 2)
    # end test_distance_haversine_t

    def test_angle_euclidean_t(self):
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)
        result = distance_euclidean_t(point1, point2)
        self.assertEqual(result.value, 5)
    # end test_angle_euclidean_t

    def test_distance_mahalanobis_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)

        # Example of a 2x2 covariance matrix
        cov_matrix = np.array([[1, 0], [0, 1]])

        # Calculate the Mahalanobis distance
        result = distance_mahalanobis_t(point1, point2, cov_matrix)

        # The expected result is the Euclidean distance
        expected_result = np.sqrt((2 ** 2) + (3 ** 2))  # sqrt(4 + 9) = sqrt(13)

        self.assertAlmostEqual(result.value, expected_result, places=4)
    # end test_distance_mahalanobis_t

    def test_distance_seuclidean_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        std_devs = np.array([1, 1])  # Assuming standard deviations of 1 for both dimensions
        result = distance_seuclidean_t(point1, point2, std_devs)
        self.assertAlmostEqual(result.value, 3.6056, places=4)
    # end test_distance_seuclidean_t

    def test_distance_sqeuclidean_t(self):
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_sqeuclidean_t(point1, point2)
        self.assertEqual(result.value, 13)
    # end test_distance_sqeuclidean_t

# end TestPoint2D


if __name__ == '__main__':
    unittest.main()
# end if
