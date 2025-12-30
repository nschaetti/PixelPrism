import unittest
import numpy as np
from pixel_prism.data import Point3D, TPoint3D, Scalar, TScalar


class TestPoint3D(unittest.TestCase):
    """
    Test cases for Point3D class.
    """

    def test_set_and_get(self):
        """
        Test setting and getting coordinates.
        """
        p = Point3D(1, 2, 3)
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 2)
        self.assertEqual(p.z, 3)
        
        p.x = 4
        p.y = 5
        p.z = 6
        self.assertEqual(p.x, 4)
        self.assertEqual(p.y, 5)
        self.assertEqual(p.z, 6)

    def test_addition(self):
        """
        Test addition of two points.
        """
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        p3 = p1 + p2
        self.assertEqual(p3.x, 5)
        self.assertEqual(p3.y, 7)
        self.assertEqual(p3.z, 9)

    def test_addition_scalar(self):
        """
        Test addition of a point and a scalar.
        """
        p1 = Point3D(1, 2, 3)
        p2 = p1 + 2
        self.assertEqual(p2.x, 3)
        self.assertEqual(p2.y, 4)
        self.assertEqual(p2.z, 5)

    def test_addition_tuple(self):
        """
        Test addition of a point and a tuple.
        """
        p1 = Point3D(1, 2, 3)
        p2 = p1 + (4, 5, 6)
        self.assertEqual(p2.x, 5)
        self.assertEqual(p2.y, 7)
        self.assertEqual(p2.z, 9)
        
        # Test with invalid tuple length
        with self.assertRaises(ValueError):
            p1 + (4, 5)

    def test_subtraction(self):
        """
        Test subtraction of two points.
        """
        p1 = Point3D(4, 5, 6)
        p2 = Point3D(1, 2, 3)
        p3 = p1 - p2
        self.assertEqual(p3.x, 3)
        self.assertEqual(p3.y, 3)
        self.assertEqual(p3.z, 3)

    def test_subtraction_scalar(self):
        """
        Test subtraction of a point and a scalar.
        """
        p1 = Point3D(4, 5, 6)
        p2 = p1 - 2
        self.assertEqual(p2.x, 2)
        self.assertEqual(p2.y, 3)
        self.assertEqual(p2.z, 4)

    def test_subtraction_tuple(self):
        """
        Test subtraction of a point and a tuple.
        """
        p1 = Point3D(4, 5, 6)
        p2 = p1 - (1, 2, 3)
        self.assertEqual(p2.x, 3)
        self.assertEqual(p2.y, 3)
        self.assertEqual(p2.z, 3)
        
        # Test with invalid tuple length
        with self.assertRaises(ValueError):
            p1 - (1, 2)

    def test_multiplication(self):
        """
        Test multiplication of a point and a scalar.
        """
        p1 = Point3D(1, 2, 3)
        p2 = p1 * 2
        self.assertEqual(p2.x, 2)
        self.assertEqual(p2.y, 4)
        self.assertEqual(p2.z, 6)

    def test_multiplication_point(self):
        """
        Test multiplication of two points.
        """
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        p3 = p1 * p2
        self.assertEqual(p3.x, 4)
        self.assertEqual(p3.y, 10)
        self.assertEqual(p3.z, 18)

    def test_multiplication_tuple(self):
        """
        Test multiplication of a point and a tuple.
        """
        p1 = Point3D(1, 2, 3)
        p2 = p1 * (4, 5, 6)
        self.assertEqual(p2.x, 4)
        self.assertEqual(p2.y, 10)
        self.assertEqual(p2.z, 18)
        
        # Test with invalid tuple length
        with self.assertRaises(ValueError):
            p1 * (4, 5)

    def test_division(self):
        """
        Test division of a point by a scalar.
        """
        p1 = Point3D(2, 4, 6)
        p2 = p1 / 2
        self.assertEqual(p2.x, 1)
        self.assertEqual(p2.y, 2)
        self.assertEqual(p2.z, 3)

    def test_division_point(self):
        """
        Test division of two points.
        """
        p1 = Point3D(4, 6, 8)
        p2 = Point3D(2, 3, 4)
        p3 = p1 / p2
        self.assertEqual(p3.x, 2)
        self.assertEqual(p3.y, 2)
        self.assertEqual(p3.z, 2)

    def test_division_tuple(self):
        """
        Test division of a point and a tuple.
        """
        p1 = Point3D(4, 6, 8)
        p2 = p1 / (2, 3, 4)
        self.assertEqual(p2.x, 2)
        self.assertEqual(p2.y, 2)
        self.assertEqual(p2.z, 2)
        
        # Test with invalid tuple length
        with self.assertRaises(ValueError):
            p1 / (2, 3)

    def test_equality(self):
        """
        Test equality of two points.
        """
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(1, 2, 3)
        p3 = Point3D(4, 5, 6)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertEqual(p1, (1, 2, 3))
        self.assertNotEqual(p1, (4, 5, 6))
        
        # Test with invalid tuple length
        self.assertNotEqual(p1, (1, 2))

    def test_abs(self):
        """
        Test absolute value of a point.
        """
        p1 = Point3D(-1, -2, -3)
        p2 = abs(p1)
        self.assertEqual(p2.x, 1)
        self.assertEqual(p2.y, 2)
        self.assertEqual(p2.z, 3)


class TestTPoint3D(unittest.TestCase):
    """
    Test cases for TPoint3D class.
    """

    def test_tpoint3d_creation(self):
        """
        Test creation of TPoint3D from Point3D.
        """
        p1 = Point3D(1, 2, 3)
        tp1 = TPoint3D.tpoint3d(p1)
        self.assertEqual(tp1.x, 1)
        self.assertEqual(tp1.y, 2)
        self.assertEqual(tp1.z, 3)

    def test_tpoint3d_addition(self):
        """
        Test addition of TPoint3D and Point3D.
        """
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(4, 5, 6)
        tp1 = TPoint3D.tpoint3d(p1)
        tp2 = tp1 + p2
        self.assertEqual(tp2.x, 5)
        self.assertEqual(tp2.y, 7)
        self.assertEqual(tp2.z, 9)

    def test_tpoint3d_subtraction(self):
        """
        Test subtraction of TPoint3D and Point3D.
        """
        p1 = Point3D(4, 5, 6)
        p2 = Point3D(1, 2, 3)
        tp1 = TPoint3D.tpoint3d(p1)
        tp2 = tp1 - p2
        self.assertEqual(tp2.x, 3)
        self.assertEqual(tp2.y, 3)
        self.assertEqual(tp2.z, 3)

    def test_tpoint3d_multiplication(self):
        """
        Test multiplication of TPoint3D and scalar.
        """
        p1 = Point3D(1, 2, 3)
        tp1 = TPoint3D.tpoint3d(p1)
        tp2 = tp1 * 2
        self.assertEqual(tp2.x, 2)
        self.assertEqual(tp2.y, 4)
        self.assertEqual(tp2.z, 6)

    def test_tpoint3d_division(self):
        """
        Test division of TPoint3D by scalar.
        """
        p1 = Point3D(2, 4, 6)
        tp1 = TPoint3D.tpoint3d(p1)
        tp2 = tp1 / 2
        self.assertEqual(tp2.x, 1)
        self.assertEqual(tp2.y, 2)
        self.assertEqual(tp2.z, 3)

    def test_tpoint3d_equality(self):
        """
        Test equality of TPoint3D objects.
        """
        p1 = Point3D(1, 2, 3)
        p2 = Point3D(1, 2, 3)
        tp1 = TPoint3D.tpoint3d(p1)
        tp2 = TPoint3D.tpoint3d(p2)
        self.assertEqual(tp1, tp2)
        self.assertEqual(tp1, p1)
        self.assertEqual(tp1, (1, 2, 3))

    def test_tpoint3d_on_change(self):
        """
        Test on_change event for TPoint3D.
        """
        p1 = Point3D(1, 2, 3)
        tp1 = TPoint3D.tpoint3d(p1)
        
        # Create a counter to track changes
        change_count = [0]
        
        def on_change(sender, event_type, **kwargs):
            change_count[0] += 1
        
        tp1.on_change.subscribe(on_change)
        
        # Modify the source point
        p1.x = 4
        self.assertEqual(tp1.x, 4)
        self.assertEqual(change_count[0], 1)
        
        p1.y = 5
        self.assertEqual(tp1.y, 5)
        self.assertEqual(change_count[0], 2)
        
        p1.z = 6
        self.assertEqual(tp1.z, 6)
        self.assertEqual(change_count[0], 3)


if __name__ == "__main__":
    unittest.main()