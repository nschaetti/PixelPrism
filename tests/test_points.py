
# Imports
import unittest
from pixel_prism.data import Point2D


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

        def on_change(x, y):
            changes.append((x, y))
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

# end TestPoint2D


if __name__ == '__main__':
    unittest.main()
# end if
