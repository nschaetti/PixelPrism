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
from pixel_prism.data import Scalar, Point2D, Matrix2D, tscalar, tpoint2d, tmatrix2d, mv_t


class TestComplexEquationIntegration(unittest.TestCase):

    def setUp(self):
        """
        Set up the test case.
        """
        # Scalars
        self.s1 = Scalar(2)
        self.s2 = Scalar(3)
        self.s3 = Scalar(4)

        # Points
        self.p1 = Point2D(1, 2)
        self.p2 = Point2D(3, 4)

        # Matrices
        self.m1 = Matrix2D(np.array([[1, 2], [3, 4]]))
        self.m2 = Matrix2D(np.array([[2, 0], [1, 2]]))

        # TScalars
        self.ts1 = tscalar(self.s1)
        self.ts2 = tscalar(self.s2)
        self.ts3 = tscalar(self.s3)

        # TPoints
        self.tp1 = tpoint2d(self.p1)
        self.tp2 = tpoint2d(self.p2)

        # TMatrices
        self.tm1 = tmatrix2d(self.m1)
        self.tm2 = tmatrix2d(self.m2)

        # Hook counters
        self.hook_counts = {
            'ts1': 0,
            'ts2': 0,
            'ts3': 0,
            'tp1': 0,
            'tp2': 0,
            'tm1': 0,
            'tm2': 0,
            'matrix_product': 0,
            'mv_result': 0,
            'eq_part11': 0,
            'eq_part12': 0,
            'eq_part1': 0,
            'equation': 0
        }

        # Register hooks
        self.ts1.add_event_listener("on_change", self.make_hook('ts1'))
        self.ts2.add_event_listener("on_change", self.make_hook('ts2'))
        self.ts3.add_event_listener("on_change", self.make_hook('ts3'))
        self.tp1.add_event_listener("on_change", self.make_hook('tp1'))
        self.tp2.add_event_listener("on_change", self.make_hook('tp2'))
        self.tm1.add_event_listener("on_change", self.make_hook('tm1'))
        self.tm2.add_event_listener("on_change", self.make_hook('tm2'))
    # end setUp

    def make_hook(self, key):
        """
        Factory method to create hooks for a specific key.
        """
        def hook(event):
            self.hook_counts[key] += 1
        return hook
    # end make_hook

    def test_complex_equation(self):
        """
        Test a complex equation involving TScalar, TPoint2D, and TMatrix2D.
        """
        # Create a complex equation involving TScalar, TPoint2D, and TMatrix2D
        # Equation: (ts1 + ts2) * (tp1 - tp2) + mv_t(tm1 @ tm2, tp1) - ts3
        matrix_product = self.tm1 @ self.tm2
        matrix_product.add_event_listener("on_change", self.make_hook('matrix_product'))

        mv_result = mv_t(matrix_product, self.tp1)
        mv_result.add_event_listener("on_change", self.make_hook('mv_result'))

        eq_part11 = self.ts1 + self.ts2
        eq_part11.add_event_listener("on_change", self.make_hook('eq_part11'))

        eq_part12 = self.tp1 - self.tp2
        eq_part12.add_event_listener("on_change", self.make_hook('eq_part12'))

        eq_part1 = eq_part11 * eq_part12
        eq_part1.add_event_listener("on_change", self.make_hook('eq_part1'))

        equation = eq_part1 + mv_result - self.ts3
        equation.add_event_listener("on_change", self.make_hook('equation'))

        # (2 + 3) * ([1, 2] - [3, 4]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 4
        # 5 * ([-2, -2]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 4
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [1, 2] = [12, 26]
        # [-10, -10] + [12, 26] - 4 = [2, 16] - 4 = [-2, 12]
        # Initial checks
        self.assertAlmostEqual(equation.x, -2)  # Expected x value
        self.assertAlmostEqual(equation.y, 12)  # Expected y value

        # (5 + 3) * ([1, 2] - [3, 4]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 4
        # 8 * ([-2, -2]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 4
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [1, 2] = [12, 26]
        # [-16, -16] + [12, 26] - 4 = [-4, 10] - 4 = [-8, 6]
        # Modify the Scalars and check results
        self.s1.value = 5
        # ts1 is modified (1)
        # eq_part11 is modified (1)
        # eq_part1 is modified (1)
        # equation is modified (1)
        self.assertAlmostEqual(equation.x, -8)  # Expected x value after modification
        self.assertAlmostEqual(equation.y, 6)  # Expected y value after modification
        self.assertEqual(self.hook_counts['ts1'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['eq_part11'], 1)  # eq_part11 should update twice
        self.assertEqual(self.hook_counts['eq_part1'], 1)  # eq_part1 should update twice
        self.assertEqual(self.hook_counts['equation'], 1)  # Final equation should update seven times

        # (5 + 1) * ([1, 2] - [3, 4]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 4
        # 6 * ([-2, -2]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 4
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [1, 2] = [12, 26]
        # [-12, -12] + [12, 26] - 4 = [0, 14] - 4 = [-4, 10]
        self.s2.value = 1
        # ts2 is modified
        # eq_part11 is modified
        # eq_part1 is modified
        # equation is modified
        self.assertAlmostEqual(equation.x, -4)  # Expected x value after modification
        self.assertAlmostEqual(equation.y, 10)  # Expected y value after modification
        self.assertEqual(self.hook_counts['ts2'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['eq_part11'], 2)  # eq_part11 should update twice
        self.assertEqual(self.hook_counts['eq_part1'], 2)  # eq_part1 should update twice
        self.assertEqual(self.hook_counts['equation'], 2)  # Final equation should update seven times

        # (5 + 1) * ([1, 2] - [3, 4]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 0
        # 6 * ([-2, -2]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [1, 2] - 0
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [1, 2] = [12, 26]
        # [-12, -12] + [12, 26] - 0 = [0, 14] - 0 = [0, 14]
        self.s3.value = 0
        # ts3 is modified
        # equation is modified
        self.assertAlmostEqual(equation.x, 0)  # Expected x value after modification
        self.assertAlmostEqual(equation.y, 14) # Expected y value after modification
        self.assertEqual(self.hook_counts['ts3'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['equation'], 3)  # Final equation should update seven times

        # Modify the Points and check results
        # (5 + 1) * ([4, 2] - [3, 4]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # 6 * ([1, -2]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [4, 2] = [24, 56]
        # [6, -12] + [24, 56] - 0 = [30, 44] - 0 = [30, 44]
        self.p1.x = 4
        # tp1 is modified
        # mv_result is modified
        # equation is modified
        self.assertAlmostEqual(equation.x, 30)  # Expected x value after modification
        self.assertEqual(self.hook_counts['tp1'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['mv_result'], 1)
        self.assertEqual(self.hook_counts['equation'], 4)  # Final equation should update seven times

        # (5 + 1) * ([4, 2] - [3, 1]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # 6 * ([1, 1]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [4, 2] = [24, 56]
        # [6, 6] + [24, 56] - 0 = [30, 62] - 0 = [30, 62]
        self.p2.y = 1
        # tp2 is modified
        # eq_part12 is modified
        # eq_part1 is modified
        # equation is modified
        self.assertAlmostEqual(equation.y, 62)  # Expected y value after modification
        self.assertEqual(self.hook_counts['tp2'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['eq_part12'], 1)
        self.assertEqual(self.hook_counts['eq_part1'], 3)
        self.assertEqual(self.hook_counts['equation'], 4)  # Final equation should update seven times

        # Modify the Matrices and check results
        # (5 + 1) * ([4, 2] - [3, 1]) + ([[2, 1], [1, 3]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # 6 * ([1, 1]) + ([[2, 1], [1, 3]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # ([[2, 1], [1, 3]] @ [[2, 0], [1, 2]]) = [[5, 2], [5, 6]]
        # [[5, 2], [5, 6]] x [4, 2] = [24, 36]
        # [6, 6] + [24, 36] - 0 = [30, 42] - 0 = [30, 42]
        self.m1.data = np.array([[2, 1], [1, 3]])
        # tm1 is modified
        # matrix_product is modified
        # mv_result is modified
        # equation is modified
        self.assertAlmostEqual(equation.x, 30)  # Expected x value after modification
        self.assertEqual(self.hook_counts['tm1'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['matrix_product'], 1)
        self.assertEqual(self.hook_counts['mv_result'], 2)
        self.assertEqual(self.hook_counts['equation'], 5)  # Final equation should update seven times

        # (5 + 1) * ([4, 2] - [3, 1]) + ([[2, 1], [1, 3]] @ [[1, 0], [0, 1]]) x [4, 2] - 0
        # 6 * ([1, 1]) + ([[2, 1], [1, 3]] @ [[1, 0], [0, 1]]) x [4, 2] - 0
        # ([[2, 1], [1, 3]] @ [[1, 0], [0, 1]]) = [[2, 1], [1, 3]]
        # [[2, 1], [1, 3]] x [4, 2] = [10, 10]
        # [6, 6] + [10, 10] - 0 = [16, 16] - 0 = [16, 16]
        self.m2.data = np.array([[1, 0], [0, 1]])
        # tm2 is modified
        # matrix_product is modified
        # mv_result is modified
        # equation is modified
        self.assertAlmostEqual(equation.y, 16)  # Expected y value after modification
        self.assertEqual(self.hook_counts['ts1'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['ts2'], 1)  # s2 changed once
        self.assertEqual(self.hook_counts['ts3'], 1)  # ts3 changed once
        self.assertEqual(self.hook_counts['tp1'], 1)  # p1 changed once
        self.assertEqual(self.hook_counts['tm1'], 1)  # m1 changed once
        self.assertEqual(self.hook_counts['tm2'], 1)  # m3 changed once
        self.assertEqual(self.hook_counts['matrix_product'], 2)  # Matrix product should update twice
        self.assertEqual(self.hook_counts['mv_result'], 3)  # mv_result should update three times
        self.assertEqual(self.hook_counts['eq_part11'], 2)  # eq_part11 should update twice
        self.assertEqual(self.hook_counts['eq_part12'], 1)  # eq_part12 should update twice
        self.assertEqual(self.hook_counts['eq_part1'], 3)  # eq_part1 should update twice
        self.assertEqual(self.hook_counts['equation'], 6)  # Final equation should update seven times
    # end test_complex_equation

# end class TestComplexEquationIntegration


if __name__ == '__main__':
    unittest.main()
# end if