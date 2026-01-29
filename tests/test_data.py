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

#
# Unit tests for the Scalar class
#

# Imports
import unittest
from pixelprism.data.scalar import Scalar


class TestScalar(unittest.TestCase):

    def test_initial_value(self):
        """
        Test that the initial value is set correctly.
        """
        scalar = Scalar(10)
        self.assertEqual(scalar.value, 10)
    # end test_initial_value

    def test_value_change(self):
        """
        Test that the value of the scalar can be changed.
        """
        changes = []

        def on_change(value):
            changes.append(value)
        # end on_change

        # Create a scalar with an initial value of 10
        scalar = Scalar(10, on_change=on_change)
        scalar.value = 20

        # Check that the value has been updated
        self.assertEqual(scalar.value, 20)
        self.assertEqual(changes, [20])
    # end test_value_change

    def test_set_method_triggers_event(self):
        """
        Test that the set method triggers the on_change event.
        """
        changes = []

        def on_change(value):
            changes.append(value)
        # end on_change

        # Create a scalar with an initial value of 10
        scalar = Scalar(10, on_change=on_change)
        scalar.set(30)

        # Check that the value has been updated
        self.assertEqual(scalar.value, 30)
        self.assertEqual(changes, [30])
    # end test_set_method_triggers_event

    def test_copy_method(self):
        """
        Test that the copy method returns a copy of the scalar.
        """
        scalar = Scalar(10)
        scalar_copy = scalar.copy()

        # Check that the copy has the same value
        self.assertEqual(scalar.value, scalar_copy.value)

        # Change the value of the copy
        scalar_copy.value = 20
        self.assertNotEqual(scalar.value, scalar_copy.value)
    # end test_copy_method

    def test_addition(self):
        """
        Test that the addition operator works as expected.
        """
        a = Scalar(10)
        b = Scalar(5)
        self.assertEqual((a + b).value, 15)
        self.assertEqual((a + 5).value, 15)
        self.assertEqual((5 + a).value, 15)
    # end test_addition

    def test_subtraction(self):
        """
        Test that the subtraction operator works as expected.
        """
        a = Scalar(10)
        b = Scalar(5)
        self.assertEqual((a - b).value, 5)
        self.assertEqual((a - 5).value, 5)
        self.assertEqual((15 - a).value, 5)
    # end test_subtraction

    def test_multiplication(self):
        """
        Test that the multiplication operator works as expected.
        """
        a = Scalar(10)
        b = Scalar(5)
        self.assertEqual((a * b).value, 50)
        self.assertEqual((a * 5).value, 50)
        self.assertEqual((5 * a).value, 50)
    # end test_multiplication

    def test_division(self):
        """
        Test that the division operator works as expected.
        """
        a = Scalar(10)
        b = Scalar(5)
        self.assertEqual((a / b).value, 2)
        self.assertEqual((a / 2).value, 5)
        self.assertEqual((20 / a).value, 2)
    # end test_division

    def test_equality(self):
        """
        Test that the equality operator works as expected.
        """
        a = Scalar(10)
        b = Scalar(10)
        c = Scalar(5)
        self.assertTrue(a == b)
        self.assertFalse(a == c)
        self.assertTrue(a == 10)
        self.assertFalse(a == 5)
    # end test_equality

    def test_inequality(self):
        """
        Test that the inequality operator works as expected.
        """
        a = Scalar(10)
        b = Scalar(10)
        c = Scalar(5)
        self.assertFalse(a != b)
        self.assertTrue(a != c)
        self.assertFalse(a != 10)
        self.assertTrue(a != 5)
    # end test_inequality

    def test_addition_creates_new_object(self):
        """
        Test that the addition operator creates a new object.
        """
        a = Scalar(10)
        b = Scalar(5)
        c = a + b
        self.assertNotEqual(id(a), id(c))
        self.assertNotEqual(id(b), id(c))
        self.assertEqual(c.value, 15)
    # end test_addition_creates_new_object

# end TestScalar


if __name__ == '__main__':
    unittest.main()
# end if
