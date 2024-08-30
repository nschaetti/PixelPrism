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
    Scalar, TScalar,
    floor_t, ceil_t, trunc_t, frac_t,
    sqrt_t, exp_t, expm1_t, log_t, log1p_t,
    log2_t, log10_t, sin_t, cos_t, tan_t, asin_t,
    acos_t, atan_t, atan2_t, sinh_t, cosh_t, tanh_t,
    asinh_t, acosh_t, atanh_t, degrees_t,
    add_t, sub_t, mul_t, div_t
)


class TestScalar(unittest.TestCase):
    """
    Test cases for the Scalar class.
    """

    def test_initialization(self):
        """
        Test the initialization of a Scalar object.
        """
        scalar = Scalar(5)
        self.assertEqual(scalar.value, 5)
    # end test_initialization

    def test_set_and_get(self):
        """
        Test the set and get methods of a Scalar objects
        """
        scalar = Scalar()
        scalar.value = 10
        self.assertEqual(scalar.value, 10)
    # end test_set_and_get

    def test_copy(self):
        """
        Test the copy method of a Scalar object.
        """
        scalar = Scalar(10)
        scalar_copy = scalar.copy()
        self.assertEqual(scalar_copy.value, scalar.value)
    # end test_copy

    def test_addition(self):
        """
        Test the addition operation of a Scalar object.
        """
        scalar1 = Scalar(5)
        scalar2 = Scalar(3)
        result = scalar1 + scalar2
        self.assertEqual(result.value, 8)

        result = scalar1 + 2
        self.assertEqual(result.value, 7)
    # end test_addition

    def test_subtraction(self):
        """
        Test the subtraction operation of a Scalar object.
        """
        scalar1 = Scalar(5)
        scalar2 = Scalar(3)
        result = scalar1 - scalar2
        self.assertEqual(result.value, 2)

        result = scalar1 - 2
        self.assertEqual(result.value, 3)
    # end test_subtraction

    def test_multiplication(self):
        """
        Test the multiplication operation of a Scalar object.
        """
        scalar1 = Scalar(5)
        scalar2 = Scalar(3)
        result = scalar1 * scalar2
        self.assertEqual(result.value, 15)

        result = scalar1 * 2
        self.assertEqual(result.value, 10)
    # end test_multiplication

    def test_division(self):
        """
        Test the division operation of a Scalar object.
        """
        scalar1 = Scalar(6)
        scalar2 = Scalar(3)
        result = scalar1 / scalar2
        self.assertEqual(result.value, 2)

        result = scalar1 / 2
        self.assertEqual(result.value, 3)
    # end test_division

    def test_comparison(self):
        """
        Test the comparison operations of a Scalar object.
        """
        scalar1 = Scalar(5)
        scalar2 = Scalar(3)
        self.assertTrue(scalar1 > scalar2)
        self.assertTrue(scalar1 >= scalar2)
        self.assertFalse(scalar1 < scalar2)
        self.assertFalse(scalar1 <= scalar2)
        self.assertTrue(scalar1 != scalar2)
        self.assertFalse(scalar1 == scalar2)
    # end test_comparison

    def test_event_trigger(self):
        def on_change(value):
            nonlocal changed_value
            changed_value = value
        # end on_change

        changed_value = None
        scalar = Scalar(5, on_change=on_change)
        scalar.value = 10
        self.assertEqual(changed_value, 10)
    # end test_event_trigger

    def test_event_trigger_tscalar(self):
        scalar1 = Scalar(3)
        scalar2 = Scalar(4)
        tscalar = TScalar(lambda s1, s2: s1.value + s2.value, s1=scalar1, s2=scalar2)
        self.assertEqual(tscalar.value, 7)
        scalar1.value = 5
        self.assertEqual(tscalar.value, 9)
    # ned test_event_trigger_tscalar

    def test_tscalar_restriction(self):
        scalar1 = Scalar(3)
        scalar2 = Scalar(4)
        tscalar = TScalar(lambda s1, s2: s1.value + s2.value, s1=scalar1, s2=scalar2)

        with self.assertRaises(AttributeError):
            tscalar.value = 10
        # end with
    # end test_tscalar_restriction

    def test_floor_t(self):
        scalar = Scalar(3.7)
        tscalar = floor_t(scalar)
        self.assertEqual(tscalar.value, 3)
    # end test_floor_t

    def test_ceil_t(self):
        scalar = Scalar(3.1)
        tscalar = ceil_t(scalar)
        self.assertEqual(tscalar.value, 4)
    # end test_ceil_t

    def test_trunc_t(self):
        scalar = Scalar(3.7)
        tscalar = trunc_t(scalar)
        self.assertEqual(tscalar.value, 3)
    # end test_trunc_t

    def test_frac_t(self):
        scalar = Scalar(3.7)
        tscalar = frac_t(scalar)
        self.assertAlmostEqual(tscalar.value, 0.7)
    # end test_frac_t

    def test_sqrt_t(self):
        scalar = Scalar(9)
        tscalar = sqrt_t(scalar)
        self.assertEqual(tscalar.value, 3)
    # end test_sqrt_t

    def test_exp_t(self):
        scalar = Scalar(1)
        tscalar = exp_t(scalar)
        self.assertAlmostEqual(tscalar.value, np.exp(1))
    # end test_exp_t

    def test_expm1_t(self):
        scalar = Scalar(1)
        tscalar = expm1_t(scalar)
        self.assertAlmostEqual(tscalar.value, np.expm1(1))
    # end test_expm1_t

    def test_log_t(self):
        scalar = Scalar(1)
        tscalar = log_t(scalar)
        self.assertAlmostEqual(tscalar.value, np.log(1))
    # end test_log_t

    def test_log1p_t(self):
        scalar = Scalar(1)
        tscalar = log1p_t(scalar)
        self.assertAlmostEqual(tscalar.value, np.log1p(1))
    # end test_log1p_t

    def test_log2_t(self):
        scalar = Scalar(2)
        tscalar = log2_t(scalar)
        self.assertAlmostEqual(tscalar.value, np.log2(2))
    # end test_log2_t

    def test_log10_t(self):
        scalar = Scalar(10)
        tscalar = log10_t(scalar)
        self.assertAlmostEqual(tscalar.value, np.log10(10))
    # end test_log10_t

    def test_trigonometric_functions(self):
        scalar = Scalar(np.pi / 2)
        self.assertAlmostEqual(sin_t(scalar).value, 1)
        self.assertAlmostEqual(cos_t(scalar).value, 0)
        self.assertAlmostEqual(tan_t(scalar).value, np.tan(np.pi / 2))

        self.assertAlmostEqual(asin_t(Scalar(1)).value, np.pi / 2)
        self.assertAlmostEqual(acos_t(Scalar(0)).value, np.pi / 2)
        self.assertAlmostEqual(atan_t(Scalar(1)).value, np.pi / 4)
    # end test_trigonometric_functions

    def test_atan2_t(self):
        y = Scalar(1)
        x = Scalar(1)
        self.assertAlmostEqual(atan2_t(y, x).value, np.pi / 4)
    # end test_atan2_t

    def test_hyperbolic_functions(self):
        scalar = Scalar(1)
        self.assertAlmostEqual(sinh_t(scalar).value, np.sinh(1))
        self.assertAlmostEqual(cosh_t(scalar).value, np.cosh(1))
        self.assertAlmostEqual(tanh_t(scalar).value, np.tanh(1))
    # end test_hyperbolic_functions

    def test_inverse_hyperbolic_functions(self):
        scalar = Scalar(1)
        self.assertAlmostEqual(asinh_t(scalar).value, np.arcsinh(1))
        self.assertAlmostEqual(acosh_t(Scalar(2)).value, np.arccosh(2))
        self.assertAlmostEqual(atanh_t(Scalar(0.5)).value, np.arctanh(0.5))
    # end test_inverse_hyperbolic_functions

    def test_degrees_t(self):
        scalar = Scalar(np.pi)
        tscalar = degrees_t(scalar)
        self.assertAlmostEqual(tscalar.value, 180)
    # end test_degrees_t

    def test_event_dispatch_on_operations(self):
        changes = []

        def on_change(value):
            changes.append(value)
        # end on_change

        scalar1 = Scalar(10, on_change=on_change)
        scalar1.value = 20
        scalar2 = Scalar(5)
        _ = scalar1 + scalar2
        _ = scalar1 - scalar2
        _ = scalar1 * scalar2
        _ = scalar1 / scalar2

        self.assertEqual(len(changes), 1)  # Only one change should have triggered
    # end test_event_dispatch_on_operations

    def test_tscalar_with_chained_operations(self):
        scalar1 = Scalar(3)
        scalar2 = Scalar(4)
        tscalar = sin_t(scalar1 + scalar2)
        self.assertAlmostEqual(tscalar.value, np.sin(7))
    # end test_tscalar_with_chained_operations

    def test_invalid_operation(self):
        scalar = Scalar(5)
        with self.assertRaises(TypeError):
            _ = scalar + "invalid"
        # end with
    # end test_invalid_operation

    def test_initialization_and_computation(self):
        # Test that TScalar initializes correctly and computes the value
        scalar1 = Scalar(2)
        scalar2 = Scalar(3)
        tscalar = TScalar(lambda x, y: x + y, x=scalar1, y=scalar2)
        self.assertEqual(tscalar.value, 5)
    # end test_initialization_and_computation

    def test_on_change_trigger(self):
        # Test that the on_change callback is triggered when a scalar changes
        callback_triggered = False

        def on_change_callback(value):
            nonlocal callback_triggered
            callback_triggered = True

        scalar1 = Scalar(2)
        scalar2 = Scalar(3)
        tscalar = TScalar(lambda x, y: x + y, on_change=on_change_callback, x=scalar1, y=scalar2)

        # Change one of the scalars
        scalar1.value = 4

        # Check if the callback was triggered
        self.assertTrue(callback_triggered)
    # end test_on_change_trigger

    def test_on_change_trigger_with_multiple_scalars(self):
        # Test that the on_change callback is triggered when any scalar changes
        callback_triggered = False

        def on_change_callback(value):
            nonlocal callback_triggered
            callback_triggered = True
        # end on_change_callback

        scalar1 = Scalar(2)
        scalar2 = Scalar(3)
        scalar3 = Scalar(4)
        tscalar = TScalar(lambda x, y, z: x + y + z, on_change=on_change_callback, x=scalar1, y=scalar2, z=scalar3)

        # Change one of the scalars
        scalar2.value = 5

        # Check if the callback was triggered
        self.assertTrue(callback_triggered)
    # end test_on_change_trigger_with_multiple_scalars

    def test_tscalar_updates_value_on_scalar_change(self):
        # Test that TScalar updates its value when a dependent scalar changes
        scalar1 = Scalar(2)
        scalar2 = Scalar(3)
        tscalar = TScalar(lambda x, y: x + y, x=scalar1, y=scalar2)

        # Change one of the scalars
        scalar1.value = 5

        # Check if the TScalar value updated correctly
        self.assertEqual(tscalar.value, 8)
    # end test_tscalar_updates_value_on_scalar_change

    def test_on_change_not_triggered_without_change(self):
        # Test that the on_change callback is not triggered if the scalar doesn't change
        callback_triggered = False

        def on_change_callback(value):
            nonlocal callback_triggered
            callback_triggered = True
        # end on_change_callback

        scalar1 = Scalar(2)
        scalar2 = Scalar(3)
        tscalar = TScalar(lambda x, y: x + y, on_change=on_change_callback, x=scalar1, y=scalar2)

        # Check that the callback was not triggered
        self.assertFalse(callback_triggered)
    # end test_on_change_not_triggered_without_change

    def test_multiple_callbacks(self):
        # Test that multiple callbacks can be attached and triggered
        callback1_triggered = False
        callback2_triggered = False

        def callback1(value):
            nonlocal callback1_triggered
            callback1_triggered = True
        # end callback1

        def callback2(value):
            nonlocal callback2_triggered
            callback2_triggered = True
        # end callback2

        scalar1 = Scalar(2)
        scalar2 = Scalar(3)
        tscalar = TScalar(lambda x, y: x + y, on_change=callback1, x=scalar1, y=scalar2)
        scalar1.add_event_listener("on_change", callback2)

        # Change one of the scalars
        scalar1.value = 5

        # Check that both callbacks were triggered
        self.assertTrue(callback1_triggered)
        self.assertTrue(callback2_triggered)
    # end test_multiple_callbacks

    # region OPERATORS

    def test_add_t(self):
        """
        Test the addition of two TScalar objects.
        """
        # s1, s2
        s1 = Scalar(2)
        s2 = Scalar(3)

        # Addition
        # Assert the result (2 + 3 = 5)
        s3 = add_t(s1, s2)
        self.assertEqual(s3.value, 5)

        # Modify one of the sources and check the updated result
        # Assert the updated result (4 + 3 = 7)
        s1.value = 4
        self.assertEqual(s3.value, 7)

        # Update again
        # Assert the updated result (4 + 1 = 5)
        s2.value = 1
        self.assertEqual(s3.value, 5)
    # end test_add_t

    def test_sub_t(self):
        """
        Test the subtraction of two TScalar objects.
        """
        # s1, s2
        s1 = Scalar(7)
        s2 = Scalar(4)

        # Subtraction
        s3 = sub_t(s1, s2)
        self.assertEqual(s3.value, 3)

        # Modify one of the sources and check the updated result
        s1.value = 10
        self.assertEqual(s3.value, 6)

        # Check 2
        s2.value = 2
        self.assertEqual(s3.value, 8)
    # end test_sub_t

    def test_mul_t(self):
        """
        Test the multiplication of two TScalar objects.
        """
        # s1, s2
        s1 = Scalar(2)
        s2 = Scalar(3)

        # Multiplication
        s3 = mul_t(s1, s2)
        self.assertEqual(s3.value, 6)

        # Modify one of the sources and check the updated result
        s1.value = 4
        self.assertEqual(s3.value, 12)

        # Check value update 2
        s2.value = 5
        self.assertEqual(s3.value, 20)
    # end test_mul_t

    def test_mul_t2(self):
        """
        Test the multiplication of two TScalar objects.
        """
        # s1, s2
        s1 = Scalar(2)
        s2 = 3.0

        # Multiplication
        s3 = mul_t(s1, s2)
        self.assertEqual(s3.value, 6)

        # Modify one of the sources and check the updated result
        s1.value = 4
        self.assertEqual(s3.value, 12)

        # Check value update 2
        # Changing s2 desn't affect the result
        s2 = 5
        self.assertEqual(s3.value, 12)
    # end test_mul_t2

    def test_div_t(self):
        """
        Test the division of two TScalar objects.
        """
        # s1, s2
        s1 = Scalar(8)
        s2 = Scalar(4)

        # Division
        s3 = div_t(s1, s2)
        self.assertEqual(s3.value, 2)

        # Modify one of the sources and check the updated result
        s1.value = 16
        self.assertEqual(s3.value, 4)

        # Second update
        s2.value = 2
        self.assertEqual(s3.value, 8)
    # end test_div_t

    def test_div_t2(self):
        """
        Test the division of two TScalar objects.
        """
        # s1, s2
        s1 = Scalar(8)
        s2 = 4.0

        # Division
        s3 = div_t(s1, s2)
        self.assertEqual(s3.value, 2)

        # Modify one of the sources and check the updated result
        s1.value = 16
        self.assertEqual(s3.value, 4)

        # Second update
        # Changing s2 desn't affect the result
        s2 = 2
        self.assertEqual(s3.value, 4)
    # end test_div_t2

    def test_nested_tscalar_operations(self):
        """
        Test nested TScalar operations.
        """
        # s1, s2
        s1 = Scalar(2)
        s2 = Scalar(3)

        # s3 = s1 + s2
        s3 = add_t(s1, s2)

        # s4 = 5
        s4 = Scalar(5)

        # We test operator *
        s5 = s3 * s4

        # Assert the result
        # (2 + 3) * 5 = 25
        self.assertEqual(s5.value, 25)

        # Modify one of the sources and check the updated result
        # (4 + 3) * 5 = 35
        s1.value = 4
        self.assertEqual(s5.value, 35)

        # (4 + 1) * 5 = 25
        s2.value = 1
        self.assertEqual(s5.value, 25)

        # (4 + 1) * 10 = 50
        s4.value = 10
        self.assertEqual(s5.value, 50)
    # end test_nested_tscalar_operations

    def test_multiple_nested_tscalar_operations(self):
        """
        Test multiple nested TScalar operations.
        """
        # s1, s2, s3
        s1 = Scalar(2)
        s2 = Scalar(3)
        s3 = Scalar(5)

        # 5 = 2 + 3
        s4 = add_t(s1, s2)

        # (2 + 3) * 5 = 25
        s5 = s4 * s3

        # 25 - 2 = 23
        s6 = s5 - s1

        # 23 / 3 = 7.6667
        s7 = s6 / s2

        # Assert the result of the final operation
        # ((2 + 3) * 5 - 2) / 3 = 7.6667
        self.assertAlmostEqual(s7.value, 7.6667, places=4)

        # Modify the sources and check the updated result
        # ((4 + 3) * 5 - 4) / 3 = 7.6667
        s1.value = 4
        self.assertAlmostEqual(s7.value, 10.3333, places=4)  # ((4 + 3) * 5 - 4) / 3 = 11.3333

        # Update s2 and check
        # ((4 + 2) * 5 - 4) / 2 = 13
        s2.value = 2
        self.assertAlmostEqual(s7.value, 13.0, places=4)  # ((4 + 2) * 5 - 4) / 2 = 17.0

        # Update s3 and check
        # ((4 + 2) * 10 - 4) / 2 = 28
        s3.value = 10
        self.assertAlmostEqual(s7.value, 28.0, places=4)  # ((4 + 2) * 10 - 4) / 2 = 29.0
    # end test_multiple_nested_tscalar_operations

    def test_mixed_tscalar_and_scalar_operations(self):
        """
        Test mixed operations between TScalar and Scalar objects.
        """
        # s1 = 3
        s1 = Scalar(2)
        s2 = 3.0

        # Test addition, 5 = 2 + 3
        s3 = add_t(s1, s2)
        self.assertEqual(s3.value, 5)

        # Modify the Scalar and check the updated result
        # 7 = 4 + 3
        s1.value = 4
        self.assertEqual(s3.value, 7)

        # Test the reverse operation
        # 3 + 4 = 7
        s4 = s2 + s1
        self.assertEqual(s4.value, 7)

        # Test subtraction with mixed types
        # 1 = 4 - 3
        s5 = s1 - s2
        self.assertEqual(s5.value, 1)

        # Test reverse subtraction
        # -1 = 3 - 4
        s6 = s2 - s1
        self.assertEqual(s6.value, -1)
    # end test_mixed_tscalar_and_scalar_operations

    def test_mixed_tscalar_float_operations(self):
        """
        Test mixed operations between TScalar and float objects.
        """
        # s1 = 5
        s0 = Scalar(2)
        s1 = TScalar(lambda: 5)

        # Test subtraction, 3 = 5 - 2
        s2 = sub_t(s1, s0)
        self.assertEqual(s2.value, 3)

        # s3 = (5 - 2) + 10 = 13
        s3 = s2 + 10
        self.assertEqual(s3.value, 13)

        # Modify the TScalar lambda function and ensure the result updates
        # s3 = (5 - 7) + 10 = 15
        s0.value = 7
        self.assertEqual(s3.value, 8)
    # end test_mixed_tscalar_float_operations

    # endregion OPERATORS

# end TestScalar
