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
from pixelprism.math_old import ArrayData


class TestArrayData(unittest.TestCase):

    def test_initialization(self):
        """
        Test that the array math_old can be initialized correctly.
        """
        array_data = ArrayData([1, 2, 3])
        self.assertTrue(np.array_equal(array_data.data, np.array([1, 2, 3])))
    # end test_initialization

    def test_set_data(self):
        """
        Test that the array math_old can be set correctly.
        """
        array_data = ArrayData([1, 2, 3])
        array_data.data = [4, 5, 6]
        self.assertTrue(np.array_equal(array_data.data, np.array([4, 5, 6])))
    # end test_set_data

    def test_set_item(self):
        """
        Test that an item in the array math_old can be set correctly
        """
        array_data = ArrayData([1, 2, 3])
        array_data[0] = 10
        self.assertEqual(array_data[0], 10)
    # end test_set_item

    def test_on_change_callback(self):
        """
        Test that the on_change event is triggered when the array math_old is changed.
        """
        changes = []

        def on_change_callback(data):
            changes.append(data)
        # end on_change_callback

        # Modified the test to use np.array_equal instead of comparing the list directly
        array_data = ArrayData([1, 2, 3], on_change=on_change_callback)
        array_data[0] = 10
        array_data.data = [4, 5, 6]

        # Test that the on_change event is triggered twice
        self.assertEqual(len(changes), 2)
        self.assertTrue(np.array_equal(changes[0], np.array([10, 2, 3])))
        self.assertTrue(np.array_equal(changes[1], np.array([4, 5, 6])))
    # end test_on_change_callback

    def test_addition(self):
        """
        Test that two array math_old can be added together.
        """
        array_data1 = ArrayData([1, 2, 3])
        array_data2 = ArrayData([4, 5, 6])
        result = array_data1 + array_data2
        self.assertTrue(np.array_equal(result.data, np.array([5, 7, 9])))
    # end test_addition

    def test_subtraction(self):
        """
        Test that two array math_old can be subtracted.
        """
        array_data1 = ArrayData([4, 5, 6])
        array_data2 = ArrayData([1, 2, 3])
        result = array_data1 - array_data2
        self.assertTrue(np.array_equal(result.data, np.array([3, 3, 3])))
    # end test_subtraction

    def test_multiplication(self):
        """
        Test that an array math_old can be multiplied by a scalar.
        """
        array_data1 = ArrayData([1, 2, 3])
        result = array_data1 * 2
        self.assertTrue(np.array_equal(result.data, np.array([2, 4, 6])))
    # end test_multiplication

    def test_division(self):
        """
        Test that an array math_old can be divided by a scalar.
        """
        array_data1 = ArrayData([2, 4, 6])
        result = array_data1 / 2
        self.assertTrue(np.array_equal(result.data, np.array([1, 2, 3])))
    # end test_division

    def test_equality(self):
        """
        Test that two array math_old objects can be compared for equality.
        """
        array_data1 = ArrayData([1, 2, 3])
        array_data2 = ArrayData([1, 2, 3])
        self.assertTrue(array_data1 == array_data2)
    # end test_equality

    def test_inequality(self):
        """
        Test that two array math_old objects can be compared for inequality.
        """
        array_data1 = ArrayData([1, 2, 3])
        array_data2 = ArrayData([4, 5, 6])
        self.assertTrue(array_data1 != array_data2)
    # end test_inequality

    def test_copy(self):
        """
        Test that an array math_old object can be copied.
        """
        array_data1 = ArrayData([1, 2, 3])
        array_data_copy = array_data1.copy()
        self.assertTrue(np.array_equal(array_data1.data, array_data_copy.data))
        self.assertIsNot(array_data1.data, array_data_copy.data)
    # end test_copy

# end TestArrayData
