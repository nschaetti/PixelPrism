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
# Copyright (C) 2026 Pixel Prism
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

"""Tests for easing functions in ``pixelprism.anim.easing``."""

# Imports
import unittest

from pixelprism.anim.easing import (
    EASINGS,
    ease_in_bounce,
    ease_in_cubic,
    ease_in_elastic,
    ease_in_expo,
    ease_in_quad,
    ease_in_sine,
    ease_out_bounce,
    ease_out_cubic,
    ease_out_elastic,
    ease_out_expo,
    ease_out_quad,
    ease_out_sine,
    get_easing,
    interpolate_value,
    linear,
    list_easings,
    step_end,
    step_start,
    steps,
)


class TestEasing(unittest.TestCase):
    """Unit tests for easing primitives and helpers."""

    def test_endpoints_common_easings(self):
        """All tested easings must map 0->0 and 1->1."""
        functions = [
            linear,
            ease_in_quad,
            ease_out_quad,
            ease_in_cubic,
            ease_out_cubic,
            ease_in_sine,
            ease_out_sine,
            ease_in_expo,
            ease_out_expo,
            ease_in_bounce,
            ease_out_bounce,
            ease_in_elastic,
            ease_out_elastic,
        ]

        for fn in functions:
            self.assertAlmostEqual(fn(0.0), 0.0, places=7)
            self.assertAlmostEqual(fn(1.0), 1.0, places=7)
        # end for
    # end def test_endpoints_common_easings

    def test_linear_midpoint(self):
        """Linear easing should preserve midpoint progression."""
        self.assertAlmostEqual(linear(0.5), 0.5, places=7)
    # end def test_linear_midpoint

    def test_interpolate_value_uses_easing(self):
        """Interpolate helper must apply easing before lerp."""
        def lerp_float(start: float, end: float, t: float) -> float:
            return start + (end - start) * t
        # end def lerp_float

        linear_value = interpolate_value(0.0, 100.0, 0.25, lerp_float, linear)
        quad_value = interpolate_value(0.0, 100.0, 0.25, lerp_float, ease_in_quad)

        self.assertAlmostEqual(linear_value, 25.0, places=7)
        self.assertAlmostEqual(quad_value, 6.25, places=7)
    # end def test_interpolate_value_uses_easing

    def test_steps(self):
        """Step easing should produce discrete quantized values."""
        end_fn = steps(4, mode="end")
        start_fn = steps(4, mode="start")

        self.assertAlmostEqual(end_fn(0.24), 0.0, places=7)
        self.assertAlmostEqual(end_fn(0.26), 0.25, places=7)

        self.assertAlmostEqual(start_fn(0.01), 0.25, places=7)
        self.assertAlmostEqual(start_fn(1.0), 1.0, places=7)
    # end def test_steps

    def test_step_start_end(self):
        """Built-in step start/end should match expected jump behavior."""
        self.assertEqual(step_start(0.0), 0.0)
        self.assertEqual(step_start(0.1), 1.0)
        self.assertEqual(step_end(0.9), 0.0)
        self.assertEqual(step_end(1.0), 1.0)
    # end def test_step_start_end

    def test_registry(self):
        """Registry accessors should expose all named easings."""
        self.assertIn("linear", EASINGS)
        self.assertIs(get_easing("linear"), linear)
        self.assertIn("ease_in_out_cubic", list_easings())

        with self.assertRaises(KeyError):
            get_easing("unknown_curve")
        # end with
    # end def test_registry

    def test_steps_validation(self):
        """Invalid step configurations should raise errors."""
        with self.assertRaises(ValueError):
            steps(0)
        # end with
        with self.assertRaises(ValueError):
            steps(4, mode="bad")
        # end with
    # end def test_steps_validation

# end class TestEasing


if __name__ == "__main__":
    unittest.main()
# end if
