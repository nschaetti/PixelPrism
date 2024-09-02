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
from pixel_prism.data import Point2D, TPoint2D, Scalar, TScalar
from pixel_prism.data.points import (
    add_t, sub_t, mul_t, div_t, rotate_t, scale_t, dot_t,
    cross_t, norm_t, normalize_t, angle_t, distance_t, distance_squared_t,
    distance_manhattan_t, distance_chebyshev_t, distance_canberra_t, distance_minkowski_t,
    distance_hamming_t, distance_jaccard_t, distance_braycurtis_t,
    distance_cosine_t, distance_correlation_t,
    distance_euclidean_t, distance_mahalanobis_t, distance_seuclidean_t,
    distance_sqeuclidean_t, tpoint2d,
    point_range, linspace, logspace, uniform, logspace, poisson, randint, shuffle, point_arange,
    normal, choice, point_arange, meshgrid, scalar_mul_t
)


class TestPoint2D(unittest.TestCase):

    # region POINT2D

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
        """
        Test that the point can be copied.
        """
        point = Point2D(7, 8)
        point_copy = point.copy()
        self.assertEqual(point_copy.x, point.x)
        self.assertEqual(point_copy.y, point.y)
    # end test_copy

    def test_addition2(self):
        """
        Test that two points can be added together.
        """
        # 3, 4
        # 1, 2
        point1 = Point2D(3, 4)
        point2 = Point2D(1, 2)

        # (3, 4) + (1, 2) = (4, 6)
        result = point1 + point2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 6)

        # (3, 4) + 2 = (5, 6)
        result = point1 + 2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 6)

        # (3, 4) + (1, 1) = (4, 5)
        result = point1 + (1, 1)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 5)
    # end test_addition2

    def test_subtraction2(self):
        """
        Test that two points can be subtracted.
        """
        # (5, 6), (1, 2)
        point1 = Point2D(5, 6)
        point2 = Point2D(1, 2)

        # (5, 6) - (1, 2) = (4, 4)
        result = point1 - point2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 4)

        # (5, 6) - 2 = (3, 4)
        result = point1 - 2
        self.assertEqual(result.x, 3)
        self.assertEqual(result.y, 4)

        # (5, 6) - (1, 1) = (4, 5)
        result = point1 - (1, 1)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 5)
    # end test_subtraction2

    def test_multiplication2(self):
        """
        Test that two points can be subtracted.
        """
        # (2, 3)
        point = Point2D(2, 3)

        # (2, 3) * 2 = (4, 6)
        result = point * 2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 6)

        # (2, 3) * (2, 3) = (4, 9)
        result = point * Point2D(2, 3)
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 9)
    # end test_multiplication2

    def test_division2(self):
        """
        Test that two points can be subtracted.
        """
        # (4, 6)
        point = Point2D(4, 6)

        # (4, 6) / 2 = (2, 3)
        result = point / 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 3)

        # (4, 6) / (2, 3) = (2, 2)
        result = point / Point2D(2, 3)
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 2)
    # end test_division2

    def test_equality2(self):
        """
        Test that two points can be compared for equality.
        """
        # (3, 4), (3, 4), (4, 3)
        point1 = Point2D(3, 4)
        point2 = Point2D(3, 4)
        point3 = Point2D(4, 3)

        # (3, 4) == (3, 4)
        self.assertTrue(point1 == point2)
        self.assertFalse(point1 == point3)
    # end test_equality2

    def test_norm2(self):
        """
        Test that the norm of a point can be computed.
        """
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)
        self.assertEqual(point1.norm2(point2), 5)
    # end test_norm2

    def test_translate(self):
        """
        Test that a point can be translated.
        """
        point = Point2D(1, 1)
        point.translate_(Point2D(3, 4))
        self.assertEqual(point.x, 4)
        self.assertEqual(point.y, 5)
    # end test_translate

    def test_rotate(self):
        """
        Test that a point can be rotated.
        """
        point = Point2D(1, 0)
        point.rotate_(np.pi / 2)
        self.assertAlmostEqual(point.x, 0, places=5)
        self.assertAlmostEqual(point.y, 1, places=5)
    # end test_rotate

    def test_scale(self):
        """
        Test that a point can be scaled.
        """
        point = Point2D(2, 3)
        point.scale_(2)
        self.assertEqual(point.x, 4)
        self.assertEqual(point.y, 6)
    # end test_scale

    def test_event_trigger(self):
        """
        Test that the on_change event is triggered when the point is changed.
        """
        changes = []

        def on_change(event):
            changes.append((event.params['x'], event.params['y']))
        # end on_change

        # (1, 2)
        point = Point2D(1, 2, on_change=on_change)
        point.x = 3
        point.y = 4

        self.assertEqual(changes, [(3, 2), (3, 4)])
    # end test_event_trigger

    # endregion POINT2D

    # region TPOINT2D

    def test_event_trigger_tpoint2d(self):
        """
        Test that the on_change event is triggered when the point is changed.
        """
        # (1, 2), (3, 4)
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        tpoint = add_t(point1, point2)

        # (1, 2) + (3, 4) = (4, 6)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        # Modify one of the sources and check the updated result
        point1.x = 5
        self.assertEqual(tpoint.x, 8)
        self.assertEqual(tpoint.y, 6)
    # end test_event_trigger_tpoint2d

    def test_tpoint2d_restriction(self):
        """
        Test that the x and y attributes of a TPoint2D object cannot be modified.
        """
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
        tpoint = scalar_mul_t(point1, scalar)

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
        """
        Test that a point can be multiplied by a scalar.
        """
        # 2, 3
        point1 = Point2D(2, 3)
        scalar = 2.0
        tpoint = scalar_mul_t(point1, scalar)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)

        # Modify the point and check the updated result
        point1.y = 4
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 8)
    # end test_mul_t2

    def test_div_t(self):
        """
        Test that a point can be divided by a scalar.
        """
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
        """
        Test that a point can be divided by a scalar.
        """
        # (8, 6)
        point1 = Point2D(8, 6)

        # (8, 6) / 2 = (4, 3)
        scalar = 2.0
        tpoint = div_t(point1, scalar)
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 3)

        # Modify the point and check the updated result
        # (8, 12) / 2 = (4, 6)
        point1.y = 12
        self.assertEqual(tpoint.x, 4)
        self.assertEqual(tpoint.y, 6)
    # end test_div_t2

    def test_rotate_t(self):
        """
        Test that a point can be rotated.
        """
        # (1, 0), (0, 0), pi/2
        point = Point2D(1, 0)
        center = Point2D(0, 0)
        angle = Scalar(np.pi / 2)

        # Rotate the point 90 degrees around the origin
        tpoint = rotate_t(point, angle, center)
        self.assertAlmostEqual(tpoint.x, 0, places=5)
        self.assertAlmostEqual(tpoint.y, 1, places=5)

        # Modify the angle and check the updated result
        # Rotate the point 180 degrees around the origin
        angle.value = np.pi
        self.assertAlmostEqual(tpoint.x, -1, places=5)
        self.assertAlmostEqual(tpoint.y, 0, places=5)

        # Modify the point and check the updated result
        # (0, 0) -> (0, 0)
        point.x = 0
        self.assertAlmostEqual(tpoint.x, 0, places=5)
        self.assertAlmostEqual(tpoint.y, 0, places=5)
    # end test_rotate_t

    def test_rotate_t2(self):
        """
        Test that a point can be rotated.
        """
        # (1, 0), (0, 0), pi/2
        point = Point2D(1, 0)
        center = Point2D(0, 0)
        angle = np.pi / 2.0

        # Rotate the point 90 degrees around the origin
        tpoint = rotate_t(point, angle, center)
        self.assertAlmostEqual(tpoint.x, 0, places=5)
        self.assertAlmostEqual(tpoint.y, 1, places=5)

        # Modify the point and check the updated result
        # (1, 1) -> (-1, 1)
        point.y = 1
        self.assertAlmostEqual(tpoint.x, -1, places=5)
        self.assertAlmostEqual(tpoint.y, 1, places=5)
    # end test_rotate_t2

    def test_scale_t(self):
        """
        Test that a point can be scaled.
        """
        #
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
        """
        Test the dot product of two points.
        """
        # (1, 2), (3, 4)
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)

        # (1, 2) . (3, 4) = 1 * 3 + 2 * 4 = 11
        result = dot_t(point1, point2)
        self.assertEqual(result.value, 11)

        # Modify one of the points and check the updated result
        # (1, 3) . (3, 4) = 1 * 3 + 3 * 4 = 15
        point1.y = 3
        self.assertEqual(result.value, 15)
    # end test_dot_t

    def test_cross_t(self):
        """
        Test the cross product of two points.
        """
        # (1, 2), (3, 4)
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)

        # (1, 2) x (3, 4) = 1 * 4 - 2 * 3 = -2
        result = cross_t(point1, point2)
        self.assertEqual(result.value, -2)

        # Modify one of the points and check the updated result
        # (1, 2) x (3, 3) = 1 * 3 - 3 * 2 = -3
        point2.y = 3
        self.assertEqual(result.value, -3)
    # end test_cross_t

    def test_norm_t(self):
        """
        Test the norm of a point.
        """
        # (3, 4)
        point = Point2D(3, 4)

        # ||(3, 4)|| = sqrt(3^2 + 4^2) = 5
        result = norm_t(point)
        self.assertEqual(result.value, 5)

        # Modify the point and check the updated result
        # ||(6, 4)|| = sqrt(6^2 + 4^2) = 7.211102550927978
        point.x = 6
        self.assertAlmostEqual(result.value, np.sqrt(36 + 16), places=5)
    # end test_norm_t

    def test_normalize_t(self):
        """
        Test the normalization of a point.
        """
        # (3, 4)
        point = Point2D(3, 4)

        # (3, 4) / 5 = (0.6, 0.8)
        result = normalize_t(point)
        self.assertAlmostEqual(result.x, 0.6, places=5)
        self.assertAlmostEqual(result.y, 0.8, places=5)

        # Modify the point and check the updated result
        # (0, 4) / 4 = (0, 1)
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
        """
        Test the squared distance between two points.
        """
        # (0, 0), (3, 4)
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)

        # ||(0, 0) - (3, 4)||^2 = 3^2 + 4^2 = 25
        result = distance_squared_t(point1, point2)
        self.assertEqual(result.value, 25)

        # Modify one of the points and check the updated result
        # ||(1, 0) - (3, 4)||^2 = (-2)^2 + (-4)^2 = 4 + 16 = 20
        point1.x = 1
        self.assertEqual(result.value, 20)
    # end test_distance_squared_t

    # TODO: check that the value of the output
    def test_distance_manhattan_t(self):
        """
        Test the Manhattan distance between two points.
        """
        # (1, 1), (3, 4)
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)

        # |1 - 3| + |1 - 4| = 2 + 3 = 5
        result = distance_manhattan_t(point1, point2)
        self.assertEqual(result.value, 5)
    # end test_distance_manhattan_t

    # TODO: check that the value of the output
    def test_distance_chebyshev_t(self):
        """
        Test the Chebyshev distance between two points.
        """
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_chebyshev_t(point1, point2)
        self.assertEqual(result.value, 3)
    # end test_distance_chebyshev_t

    # TODO: check that the value of the output
    def test_distance_canberra_t(self):
        """
        Test the Canberra distance between two points.
        """
        point1 = Point2D(1, 2)
        point2 = Point2D(3, 4)
        result = distance_canberra_t(point1, point2)
        self.assertAlmostEqual(result.value, 0.8333, places=4)
    # end test_distance_canberra_t

    # TODO: check that the value of the output
    def test_distance_minkowski_t(self):
        """
        Test the Minkowski distance between two points.
        """
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_minkowski_t(point1, point2, p=3)
        self.assertAlmostEqual(result.value, 3.2710664, places=4)
    # end test_distance_minkowski_t

    # TODO: check that the value of the output
    def test_distance_minkowski_t2(self):
        """
        Test the Minkowski distance between two points.
        """
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_minkowski_t(point1, point2, p=Scalar(3))
        self.assertAlmostEqual(result.value, 3.2710664, places=4)
    # end test_distance_minkowski_t

    # TODO: check that the value of the output
    def test_distance_hamming_t(self):
        """
        Test the Hamming distance between two points.
        """
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
        """
        Test the Bray-Curtis distance between two points.
        """
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_braycurtis_t(point1, point2)
        self.assertEqual(result.value, 0.5555556)
    # end test_distance_braycurtis_t

    def test_distance_cosine_t(self):
        """
        Test the cosine distance between two points.
        """
        point1 = Point2D(1, 0)
        point2 = Point2D(0, 1)
        result = distance_cosine_t(point1, point2)
        self.assertEqual(result.value, 1)
    # end test_distance_cosine_t

    def test_distance_correlation_t(self):
        """
        Test the correlation distance between two points.
        """
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
        """
        Test the Euclidean distance between two points.
        """
        point1 = Point2D(0, 0)
        point2 = Point2D(3, 4)
        result = distance_euclidean_t(point1, point2)
        self.assertEqual(result.value, 5)
    # end test_angle_euclidean_t

    def test_distance_mahalanobis_t(self):
        """
        Test the Mahalanobis distance between two points.
        """
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
        """
        Test the standardized Euclidean distance between two points.
        """
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        std_devs = np.array([1, 1])  # Assuming standard deviations of 1 for both dimensions
        result = distance_seuclidean_t(point1, point2, std_devs)
        self.assertAlmostEqual(result.value, 3.6056, places=4)
    # end test_distance_seuclidean_t

    def test_distance_sqeuclidean_t(self):
        """
        Test the squared Euclidean distance between two points.
        """
        point1 = Point2D(1, 1)
        point2 = Point2D(3, 4)
        result = distance_sqeuclidean_t(point1, point2)
        self.assertEqual(result.value, 13)
    # end test_distance_sqeuclidean_t

    # endregion TPOINT2D

    # region OPERATORS

    def test_add_t2(self):
        """
        Test that two points can be added together.
        """
        # p1 = (1, 2)
        # p2 = (3, 4)
        p1 = Point2D(1, 2)
        p2 = Point2D(3, 4)

        # (1, 2) + (3, 4) = (4, 6)
        tp = add_t(p1, p2)

        # Check addition is correct
        self.assertEqual(tp.x, 4)
        self.assertEqual(tp.y, 6)

        # Modify one of the sources and check the updated result
        # (2, 3) + (3, 4) = (5, 7)
        p1.x = 2
        p1.y = 3
        self.assertEqual(tp.x, 5)
        self.assertEqual(tp.y, 7)

        # Check another modification
        # (2, 3) + (1, 1) = (3, 4)
        p2.x = 1
        p2.y = 1
        self.assertEqual(tp.x, 3)
        self.assertEqual(tp.y, 4)
    # end test_add_t

    def test_sub_t2(self):
        """
        Test that two points can be subtracted.
        """
        # p1 = (5, 7), p2 = (2, 3)
        p1 = Point2D(5, 7)
        p2 = Point2D(2, 3)

        # tpoint
        tp1 = tpoint2d(p1)
        tp2 = tpoint2d(p2)

        # (5, 7) - (2, 3) = (3, 4)
        tp = tp1 - tp2
        self.assertEqual(tp.x, 3)
        self.assertEqual(tp.y, 4)

        # Modify one of the sources and check the updated result
        p1.x = 8
        p1.y = 9
        self.assertEqual(tp.x, 6)
        self.assertEqual(tp.y, 6)

        p2.x = 1
        p2.y = 1
        self.assertEqual(tp.x, 7)
        self.assertEqual(tp.y, 8)
    # end test_sub_t

    def test_mul_t3(self):
        """
        Test that a point can be multiplied by a scalar.
        """
        # p1 = (2, 3)
        p1 = Point2D(2, 3)

        # tpoint
        tp1 = tpoint2d(p1)

        # Scalar multiplication
        scalar = 2.0
        tp = tp1 * scalar

        self.assertEqual(tp.x, 4)
        self.assertEqual(tp.y, 6)

        # Modify the source and check the updated result
        p1.x = 3
        p1.y = 4
        self.assertEqual(tp.x, 6)
        self.assertEqual(tp.y, 8)
    # end test_mul_t

    def test_div_t3(self):
        """
        Test that a point can be divided by a scalar.
        """
        p1 = Point2D(8, 6)

        # tpoint
        tp1 = tpoint2d(p1)

        # Scalar division
        scalar = 2.0
        tp = tp1 / scalar

        self.assertEqual(tp.x, 4)
        self.assertEqual(tp.y, 3)

        # Modify the source and check the updated result
        p1.x = 10
        p1.y = 8
        self.assertEqual(tp.x, 5)
        self.assertEqual(tp.y, 4)

    # end test_div_t

    def test_nested_tpoint2d_operations(self):
        """
        Test that nested operations can be performed on TPoint2D objects.
        """
        # (2, 3), (4, 5), (1, 1)
        p1 = Point2D(2, 3)
        p2 = Point2D(4, 5)
        p3 = Point2D(1, 1)

        # TPoint2D
        tp1 = tpoint2d(p1)
        tp2 = tpoint2d(p2)
        tp3 = tpoint2d(p3)

        # (2, 3) + (4, 5) = (6, 8)
        tp4 = tp1 + tp2
        tp5 = tp4 * tp3

        self.assertEqual(tp5.x, 6)  # (2+4)*1 = 6
        self.assertEqual(tp5.y, 8)  # (3+5)*1 = 8

        # Modify one of the sources and check the updated result
        p1.x = 3
        p1.y = 4
        self.assertEqual(tp5.x, 7)  # (3+4)*1 = 7
        self.assertEqual(tp5.y, 9)  # (4+5)*1 = 9

        p2.x = 2
        p2.y = 2
        self.assertEqual(tp5.x, 5)  # (3+2)*1 = 5
        self.assertEqual(tp5.y, 6)  # (4+2)*1 = 6
    # end test_nested_tpoint2d_operations

    def test_multiple_nested_tpoint2d_operations(self):
        """
        Test that multiple nested operations can be performed on TPoint2D objects.
        """
        # (2, 3), (4, 5), (1, 1)
        p1 = Point2D(2, 3)
        p2 = Point2D(4, 5)
        p3 = Point2D(1, 1)

        # TPoint2D
        sp1 = tpoint2d(p1)
        sp2 = tpoint2d(p2)
        sp3 = tpoint2d(p3)

        # Operations
        tp1 = sp1 + sp2 # (2, 3) + (4, 5) = (6, 8)
        tp2 = tp1 * sp3 # (6, 8) * (1, 1) = (6, 8)
        tp3 = tp2 / 2.0 # (6, 8) / 2 = (3, 4)
        tp4 = tp3 - sp1 # (3, 4) - (2, 3) = (1, 1)

        # Test
        self.assertEqual(tp4.x, 1.0)
        self.assertEqual(tp4.y, 1.0)

        # Modify one of the sources and check the updated result
        # tp1  = (3, 4) + (4, 5) = (7, 9)
        # tp2  = (7, 9) * (1, 1) = (7, 9)
        # tp3  = (7, 9) / 2 = (3.5, 4.5)
        # tp4  = (3.5, 4.5) - (3, 4) = (0.5, 0.5)
        p1.x = 3
        p1.y = 4
        self.assertEqual(tp4.x, 0.5)
        self.assertEqual(tp4.y, 0.5)

        # Modify one of the sources and check the updated result
        # tp1  = (3, 4) + (2, 2) = (5, 6)
        # tp2  = (5, 6) * (1, 1) = (5, 6)
        # tp3  = (5, 6) / 2 = (2.5, 3.0)
        # tp4  = (2.5, 3.0) - (3, 4) = (-0.5, -1)
        p2.x = 2
        p2.y = 2
        self.assertAlmostEqual(tp4.x, -0.5)
        self.assertAlmostEqual(tp4.y, -1)
    # end test_multiple_nested_tpoint2d_operations

    def test_mixed_tpoint2d_and_scalar_operations(self):
        """
        Test that mixed operations can be performed on TPoint
        """
        # (2, 3)
        p1 = Point2D(2, 3)

        # TPoint2D
        sp1 = tpoint2d(p1)

        # Scalar multiplication
        scalar = 2.0
        tp = sp1 * scalar

        # (2, 3) * 2 = (4, 6)
        self.assertEqual(tp.x, 4)
        self.assertEqual(tp.y, 6)

        # Modify the source and check the updated result
        # (3, 4) * 2 = (6, 8)
        p1.x = 3
        p1.y = 4
        self.assertEqual(tp.x, 6)
        self.assertEqual(tp.y, 8)

        # Test reverse operation
        # (3, 4) * 2 = (6, 8)
        tp2 = scalar * sp1
        self.assertEqual(tp2.x, 6)
        self.assertEqual(tp2.y, 8)
    # end test_mixed_tpoint2d_and_scalar_operations

    def test_mixed_tpoint2d_and_tscalar_operations(self):
        """
        Test that mixed operations can be performed on TPoint
        """
        # (3, 4)
        p1 = Point2D(3, 4)

        # TPoint2D
        tscalar = TScalar(lambda: 2)

        # Scalar multiplication
        tp = p1 * tscalar

        self.assertEqual(tp.x, 6)
        self.assertEqual(tp.y, 8)

        # Modify the TScalar and ensure the result updates
        tscalar._func = lambda: 3
        self.assertEqual(tp.x, 9)
        self.assertEqual(tp.y, 12)
    # end test_mixed_tpoint2d_and_tscalar_operations

    def test_mixed_tpoint2d_and_tscalar_addition(self):
        """
        Test that mixed operations can be performed on TPoint
        """
        p1 = Point2D(3, 4)
        tscalar = TScalar(lambda: 2)

        # Addition
        tp = p1 + tscalar

        self.assertEqual(tp.x, 5)
        self.assertEqual(tp.y, 6)

        # Modify the TScalar and ensure the result updates
        tscalar._func = lambda: 3
        self.assertEqual(tp.x, 6)
        self.assertEqual(tp.y, 7)

        # Reverse addition
        tp_reverse = tscalar + p1
        self.assertEqual(tp_reverse.x, 6)
        self.assertEqual(tp_reverse.y, 7)
    # end test_mixed_tpoint2d_and_tscalar_addition

    def test_mixed_tpoint2d_and_tscalar_subtraction(self):
        """
        Test that mixed operations can be performed on TPoint
        """
        p1 = Point2D(5, 7)
        tscalar = TScalar(lambda: 2)

        # Subtraction
        tp = p1 - tscalar

        self.assertEqual(tp.x, 3)
        self.assertEqual(tp.y, 5)

        # Modify the TScalar and ensure the result updates
        tscalar._func = lambda: 3
        self.assertEqual(tp.x, 2)
        self.assertEqual(tp.y, 4)

        # Reverse subtraction
        tp_reverse = tscalar - p1
        self.assertEqual(tp_reverse.x, -2)
        self.assertEqual(tp_reverse.y, -4)
    # end test_mixed_tpoint2d_and_tscalar_subtraction

    def test_mixed_tpoint2d_and_tscalar_multiplication(self):
        """
        Test that mixed operations can be performed on TPoint
        Point2D x TScalar -> TPoint2D
        """
        p1 = Point2D(3, 4)
        tscalar = TScalar(lambda: 2)

        # Multiplication
        # (3, 4) * 2 = (6, 8)
        tp = p1 * tscalar
        self.assertEqual(tp.x, 6)
        self.assertEqual(tp.y, 8)

        # Modify the TScalar and ensure the result updates
        # (3, 4) * 3 = (9, 12)
        tscalar._func = lambda: 3
        self.assertEqual(tp.x, 9)
        self.assertEqual(tp.y, 12)

        # Reverse multiplication
        # 3 * (3, 4) = (9, 12)
        tp_reverse = tscalar * p1
        self.assertEqual(tp_reverse.x, 9)
        self.assertEqual(tp_reverse.y, 12)
    # end test_mixed_tpoint2d_and_tscalar_multiplication

    def test_mixed_tpoint2d_and_tscalar_division(self):
        """
        Test that mixed operations can be performed on TPoint
        """
        p1 = Point2D(6, 8)
        tscalar = TScalar(lambda: 2)

        # Division
        tp = p1 / tscalar
        self.assertEqual(tp.x, 3)
        self.assertEqual(tp.y, 4)

        # Modify the TScalar and ensure the result updates
        tscalar._func = lambda: 4
        self.assertEqual(tp.x, 1.5)
        self.assertEqual(tp.y, 2)

        # Reverse division
        tp_reverse = tscalar / p1
        self.assertAlmostEqual(tp_reverse.x, 2/3, places=5)
        self.assertAlmostEqual(tp_reverse.y, 0.5, places=5)
    # end test_mixed_tpoint2d_and_tscalar_division

    # endregion OPERATORS

    # region GENERATION

    def test_point_range_point2d(self):
        """
        Test the generation of points using a range.
        """
        points = point_range(Point2D(0, 0), Point2D(5, 10), Point2D(1, 2), return_tpoint=False)
        expected_points = [
            Point2D(0, 0),
            Point2D(1, 2),
            Point2D(2, 4),
            Point2D(3, 6),
            Point2D(4, 8)
        ]
        self.assertEqual(points, expected_points)
    # end test_point_range_point2d

    def test_point_range_tpoint2d(self):
        """
        Test the generation of points using a range.
        """
        tpoints = point_range(Point2D(0, 0), Point2D(5, 10), Point2D(1, 2), return_tpoint=True)
        expected_points = [
            Point2D(0, 0),
            Point2D(1, 2),
            Point2D(2, 4),
            Point2D(3, 6),
            Point2D(4, 8)
        ]
        for tp, expected in zip(tpoints, expected_points):
            self.assertEqual(tp.x, expected.x)
            self.assertEqual(tp.y, expected.y)
        # end for

    # end test_point_range_tpoint2d

    def test_linspace_point2d(self):
        """
        Test the generation of points using linspace.
        """
        points = linspace(Point2D(0, 0), Point2D(4, 8), num=5, return_tpoint=False)
        expected_points = [
            Point2D(0, 0),
            Point2D(1, 2),
            Point2D(2, 4),
            Point2D(3, 6),
            Point2D(4, 8)
        ]
        self.assertEqual(points, expected_points)
    # end test_linspace_point2d

    def test_linspace_tpoint2d(self):
        """
        Test the generation of points using linspace.
        """
        tpoints = linspace(Point2D(0, 0), Point2D(4, 8), num=5, return_tpoint=True)
        expected_points = [
            Point2D(0, 0),
            Point2D(1, 2),
            Point2D(2, 4),
            Point2D(3, 6),
            Point2D(4, 8)
        ]
        for tp, expected in zip(tpoints, expected_points):
            self.assertEqual(tp.x, expected.x)
            self.assertEqual(tp.y, expected.y)
        # end for
    # end test_linspace_tpoint2d

    def test_logspace_point2d(self):
        """
        Test the generation of points using logspace.
        """
        points = logspace(Point2D(1, 1), Point2D(100, 1000), num=5, return_tpoint=False)
        expected_points = [
            Point2D(1.0, 1.0),
            Point2D(3.1622776601683795, 5.623414039611816),
            Point2D(10.0, 31.622785568237305),
            Point2D(31.622776601683793, 177.82803344726562),
            Point2D(100.00005340576172, 1000.0005493164062)
        ]
        for p, expected in zip(points, expected_points):
            self.assertAlmostEqual(p.x, expected.x, places=4)
            self.assertAlmostEqual(p.y, expected.y, places=4)
        # end for
    # end test_logspace_point2d

    def test_logspace_tpoint2d(self):
        """
        Test the generation of points using logspace.
        """
        tpoints = logspace(Point2D(1, 1), Point2D(100, 1000), num=5, return_tpoint=True)
        expected_points = [
            Point2D(1.0, 1.0),
            Point2D(3.1622776601683795, 5.623414039611816),
            Point2D(10.0, 31.622785568237305),
            Point2D(31.622776601683793, 177.82803344726562),
            Point2D(100.00005, 1000.0005493164062)
        ]
        for tp, expected in zip(tpoints, expected_points):
            self.assertAlmostEqual(tp.x, expected.x, places=4)
            self.assertAlmostEqual(tp.y, expected.y, places=4)
        # end for
    # end test_logspace_tpoint2d

    def test_point_uniform(self):
        """
        Test the generation of points using a uniform distribution.
        """
        low = (0, 0)
        high = (10, 10)
        points = uniform(low, high, size=5)
        self.assertEqual(len(points), 5)
        self.assertTrue(all(0 <= p.x <= 10 for p in points))
    # end test_point_uniform

    def test_uniform_different_ranges(self):
        """
        Test the generation of points using a uniform distribution with different ranges.
        """
        points = uniform(low=(0, 10), high=(5, 15), size=5, return_tpoint=True)
        self.assertEqual(len(points), 5)
        for point in points:
            self.assertTrue(isinstance(point, TPoint2D))
            self.assertTrue(0 <= point.x <= 5)
            self.assertTrue(10 <= point.y <= 15)
        # end for
    # end test_uniform_different_ranges

    def test_uniform_point2d(self):
        """
        Test the generation of points using a uniform distribution.
        """
        points = uniform(low=(0, 10), high=(5, 15), size=5, return_tpoint=False)
        self.assertEqual(len(points), 5)
        for point in points:
            self.assertTrue(isinstance(point, Point2D))
            self.assertTrue(0 <= point.x <= 5)
            self.assertTrue(10 <= point.y <= 15)
        # end for
    # end test_uniform_point2d

    def test_uniform_tpoint2d(self):
        """
        Test the generation of points using a uniform distribution.
        """
        tpoints = uniform(low=(0, 10), high=(5, 15), size=5, return_tpoint=True)
        self.assertEqual(len(tpoints), 5)
        for tpoint in tpoints:
            self.assertTrue(isinstance(tpoint, TPoint2D))
            self.assertTrue(0 <= tpoint.x <= 5)
            self.assertTrue(10 <= tpoint.y <= 15)
        # end for
    # end test_uniform_tpoint2d

    def test_normal_point2d(self):
        """
        Test the generation of points using a normal distribution.
        """
        points = normal(Point2D(0, 0), Point2D(1, 1), size=5, return_tpoint=False)
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertIsInstance(p, Point2D)
        # end for
    # end test_normal_point2d

    def test_normal_tpoint2d(self):
        """
        Test the generation of points using a normal distribution.
        """
        tpoints = normal(Point2D(0, 0), Point2D(1, 1), size=5, return_tpoint=True)
        self.assertEqual(len(tpoints), 5)
        for tp in tpoints:
            self.assertIsInstance(tp, TPoint2D)
        # end for
    # end test_normal_tpoint2d

    def test_poisson_point2d(self):
        """
        Test the generation of points using a Poisson distribution.
        """
        points = poisson(Point2D(3, 5), size=5, return_tpoint=False)
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertIsInstance(p, Point2D)
        # end for
    # end test_poisson_point2d

    def test_poisson_tpoint2d(self):
        """
        Test the generation of points using a Poisson distribution.
        """
        tpoints = poisson(Point2D(3, 5), size=5, return_tpoint=True)
        self.assertEqual(len(tpoints), 5)
        for tp in tpoints:
            self.assertIsInstance(tp, TPoint2D)
        # end for
    # end test_poisson_tpoint2d

    def test_randint_point2d(self):
        """
        Test the generation of points using randint.
        """
        points = randint(Point2D(0, 0), Point2D(10, 10), size=5, return_tpoint=False)
        self.assertEqual(len(points), 5)
        for p in points:
            self.assertIsInstance(p, Point2D)
            self.assertGreaterEqual(p.x, 0)
            self.assertLess(p.x, 10)
            self.assertGreaterEqual(p.y, 0)
            self.assertLess(p.y, 10)
        # end for
    # end test_randint_point2d

    def test_randint_tpoint2d(self):
        """
        Test the generation of points using randint.
        """
        tpoints = randint(Point2D(0, 0), Point2D(10, 10), size=5, return_tpoint=True)
        self.assertEqual(len(tpoints), 5)
        for tp in tpoints:
            self.assertIsInstance(tp, TPoint2D)
            self.assertGreaterEqual(tp.x, 0)
            self.assertLess(tp.x, 10)
            self.assertGreaterEqual(tp.y, 0)
            self.assertLess(tp.y, 10)
        # end for
    # end test_randint_tpoint2d

    def test_choice_point2d(self):
        """
        Test the choice of points.
        """
        points = [Point2D(1, 1), Point2D(2, 2), Point2D(3, 3)]
        chosen_points = choice(points, size=2, replace=False, return_tpoint=False)
        self.assertEqual(len(chosen_points), 2)
        for p in chosen_points:
            self.assertIn(p, points)
            self.assertIsInstance(p, Point2D)
        # end for
    # end test_choice_point2d

    def test_choice_tpoint2d(self):
        """
        Test the choice of points.
        """
        points = [Point2D(1, 1), Point2D(2, 2), Point2D(3, 3)]
        tpoints = choice(points, size=2, replace=False, return_tpoint=True)
        self.assertEqual(len(tpoints), 2)
        for tp in tpoints:
            self.assertIsInstance(tp, TPoint2D)
            self.assertIn((tp.x, tp.y), [(1, 1), (2, 2), (3, 3)])
        # end for
    # end test_choice_tpoint2d

    def test_shuffle_point2d(self):
        """
        Test the shuffling of points.
        """
        points = [Point2D(1, 1), Point2D(2, 2), Point2D(3, 3)]
        shuffled_points = shuffle(points, return_tpoint=False)
        self.assertEqual(len(shuffled_points), 3)
        self.assertCountEqual(shuffled_points,
                              points)  # Vérifie que les éléments sont les mêmes, indépendamment de l'ordre
        for p in shuffled_points:
            self.assertIsInstance(p, Point2D)
        # end for
    # end test_shuffle_point2d

    def test_shuffle_tpoint2d(self):
        """
        Test the shuffling of TPoints.
        """
        points = [Point2D(1, 1), Point2D(2, 2), Point2D(3, 3)]
        tpoints = shuffle(points, return_tpoint=True)
        self.assertEqual(len(tpoints), 3)
        self.assertCountEqual([(tp.x, tp.y) for tp in tpoints],
                              [(1, 1), (2, 2), (3, 3)])  # Vérifie que les valeurs sont correctes
        for tp in tpoints:
            self.assertIsInstance(tp, TPoint2D)
        # end for
    # end test_shuffle_tpoint2d

    def test_point_arange(self):
        """
        Test the generation of points using arange.
        """
        start = Point2D(0, 0)
        stop = Point2D(5, 5)
        step = Point2D(1, 1)
        points = point_arange(start, stop, step, return_tpoint=False)

        self.assertEqual(len(points), 5)
        self.assertEqual(points[0].x, 0)
        self.assertEqual(points[0].y, 0)
        self.assertEqual(points[-1].x, 4)
        self.assertEqual(points[-1].y, 4)
        for p in points:
            self.assertIsInstance(p, Point2D)
        # end for
    # end test_point_arange

    def test_point_arange_tpoint(self):
        """
        Test the generation of points using arange.
        """
        start = Point2D(0, 0)
        stop = Point2D(5, 5)
        step = Point2D(1, 1)
        tpoints = point_arange(start, stop, step, return_tpoint=True)

        self.assertEqual(len(tpoints), 5)
        self.assertEqual(tpoints[0].x, 0)
        self.assertEqual(tpoints[0].y, 0)
        self.assertEqual(tpoints[-1].x, 4)
        self.assertEqual(tpoints[-1].y, 4)
        for tp in tpoints:
            self.assertIsInstance(tp, TPoint2D)
        # end for
    # end test_point_arange_tpoint

    def test_meshgrid_point2d(self):
        """
        Test meshgrid with Point2D objects.
        """
        x_values = np.linspace(0, 2, 3)
        y_values = np.linspace(0, 2, 3)
        grid = meshgrid(x_values, y_values, return_tpoint=False)

        self.assertEqual(len(grid), 3)  # 3 rows
        self.assertEqual(len(grid[0]), 3)  # 3 columns

        # Verify individual points
        self.assertEqual(grid[0][0].x, 0)
        self.assertEqual(grid[0][0].y, 0)

        self.assertEqual(grid[2][2].x, 2)
        self.assertEqual(grid[2][2].y, 2)

        # Ensure all objects are Point2D instances
        for row in grid:
            for point in row:
                self.assertIsInstance(point, Point2D)
        # end for
    # end test_meshgrid_point2d

    def test_meshgrid_tpoint2d(self):
        """
        Test meshgrid with TPoint2D objects.
        """
        x_values = np.linspace(0, 2, 3)
        y_values = np.linspace(0, 2, 3)
        grid = meshgrid(x_values, y_values, return_tpoint=True)

        self.assertEqual(len(grid), 3)  # 3 rows
        self.assertEqual(len(grid[0]), 3)  # 3 columns

        # Verify individual points
        self.assertEqual(grid[0][0].x, 0)
        self.assertEqual(grid[0][0].y, 0)

        self.assertEqual(grid[2][2].x, 2)
        self.assertEqual(grid[2][2].y, 2)

        # Ensure all objects are TPoint2D instances
        for row in grid:
            for tpoint in row:
                self.assertIsInstance(tpoint, TPoint2D)
            # end for
        # end for
    # end test_meshgrid_tpoint2d

    # endregion GENERATION

# end TestPoint2D


if __name__ == '__main__':
    unittest.main()
# end if
