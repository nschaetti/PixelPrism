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
from pixelprism.math_old import Color


class TestColor(unittest.TestCase):
    """
    Test the Color class.
    """

    # Tests for the __eq__ and __ne__ methods
    def test_equality(self):
        """
        Test the equality and inequality operators.
        """
        color1 = Color(100, 150, 200, 1.0)
        color2 = Color(100, 150, 200, 1.0)
        color3 = Color(50, 50, 50, 0.5)
        self.assertTrue(color1 == color2)
        self.assertFalse(color1 == color3)
        self.assertTrue(color1 != color3)
        self.assertFalse(color1 != color2)
    # end test_equality

    # Other operator tests
    def test_addition(self):
        """
        Test the addition operator.
        """
        color1 = Color(100, 150, 200, 1.0)
        color2 = Color(50, 50, 50, 0.5)
        result = color1 + color2
        self.assertTrue(np.array_equal(result.value, [150, 200, 250, 1.5]))
    # end test_addition

    def test_subtraction(self):
        """
        Test the subtraction operator.
        """
        color1 = Color(100, 150, 200, 1.0)
        color2 = Color(50, 50, 50, 0.5)
        result = color1 - color2
        self.assertTrue(np.array_equal(result.value, [50, 100, 150, 0.5]))
    # end test_subtraction

    def test_multiplication(self):
        """
        Test the multiplication operator.
        """
        color1 = Color(100, 150, 200, 1.0)
        color2 = Color(2, 2, 2, 2)
        result = color1 * color2
        self.assertTrue(np.array_equal(result.value, [200, 300, 400, 2.0]))
    # end test_multiplication

    def test_division(self):
        """
        Test the division operator.
        """
        color1 = Color(100, 150, 200, 1.0)
        color2 = Color(2, 3, 4, 2)
        result = color1 / color2
        self.assertTrue(np.array_equal(result.value, [50, 50, 50, 0.5]))
    # end test_division

    def test_in_place_addition(self):
        """
        Test the in-place addition operator.
        """
        color = Color(100, 150, 200, 1.0)
        color += Color(50, 50, 50, 0.5)
        self.assertTrue(np.array_equal(color.value, [150, 200, 250, 1.5]))
    # end test_in_place_addition

    def test_in_place_subtraction(self):
        """
        Test the in-place subtraction operator.
        """
        color = Color(100, 150, 200, 1.0)
        color -= Color(50, 50, 50, 0.5)
        self.assertTrue(np.array_equal(color.value, [50, 100, 150, 0.5]))
    # end test_in_place_subtraction

    def test_in_place_multiplication(self):
        """
        Test the in-place multiplication operator.
        """
        color = Color(100, 150, 200, 1.0)
        color *= Color(2, 2, 2, 2)
        self.assertTrue(np.array_equal(color.value, [200, 300, 400, 2.0]))
    # end test_in_place_multiplication

    def test_in_place_division(self):
        """
        Test the in-place division operator.
        """
        color = Color(100, 150, 200, 1.0)
        color /= Color(2, 3, 4, 2)
        self.assertTrue(np.array_equal(color.value, [50, 50, 50, 0.5]))
    # end test_in_place_division

    def test_on_change_event_triggered(self):
        """
        Test that the on_change event is triggered when the color is modified.
        """
        def on_change_listener(value):
            nonlocal changed
            changed = True
        # end on_change_listeners

        # Test set
        color = Color(100, 150, 200, 1.0)
        changed = False
        color.add_event_listener("on_change", on_change_listener)

        # Test set
        color += Color(10, 10, 10, 0.1)
        self.assertTrue(changed)
    # end test_on_change_event_triggered

# end TestColor
