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
from pixel_prism.data import Scalar, Point2D, Matrix2D, TScalar, TMatrix2D, TPoint2D


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
        self.ts1 = TScalar.tscalar(self.s1)
        self.ts2 = TScalar.tscalar(self.s2)
        self.ts3 = TScalar.tscalar(self.s3)

        # TPoints
        self.tp1 = TPoint2D.tpoint2d(self.p1)
        self.tp2 = TPoint2D.tpoint2d(self.p2)

        # TMatrices
        self.tm1 = TMatrix2D.tmatrix2d(self.m1)
        self.tm2 = TMatrix2D.tmatrix2d(self.m2)

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
        self.ts1.on_change.subscribe(self.make_hook('ts1'))
        self.ts2.on_change.subscribe(self.make_hook('ts2'))
        self.ts3.on_change.subscribe(self.make_hook('ts3'))
        self.tp1.on_change.subscribe(self.make_hook('tp1'))
        self.tp2.on_change.subscribe(self.make_hook('tp2'))
        self.tm1.on_change.subscribe(self.make_hook('tm1'))
        self.tm2.on_change.subscribe(self.make_hook('tm2'))
    # end setUp

    def make_hook(self, key):
        """
        Factory method to create hooks for a specific key.
        """
        def hook(sender, event_type, **kwargs):
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
        matrix_product.on_change.subscribe(self.make_hook('matrix_product'))

        mv_result = TMatrix2D.mv(matrix_product, self.tp1)
        mv_result.on_change.subscribe(self.make_hook('mv_result'))

        eq_part11 = self.ts1 + self.ts2
        eq_part11.on_change.subscribe(self.make_hook('eq_part11'))
        print("111")
        eq_part12 = self.tp1 - self.tp2
        eq_part12.on_change.subscribe(self.make_hook('eq_part12'))
        print("222")
        eq_part1 = eq_part11 * eq_part12
        eq_part1.on_change.subscribe(self.make_hook('eq_part1'))

        equation = eq_part1 + mv_result - self.ts3
        equation.on_change.subscribe(self.make_hook('equation'))

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
        # ts2 is modified (1)
        # eq_part11 is modified (2)
        # eq_part1 is modified  (2)
        # equation is modified (2)
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
        # ts3 is modified (1)
        # equation is modified (3)
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
        # Equation: (ts1 + ts2) * (tp1 - tp2) + mv_t(tm1 @ tm2, tp1) - ts3
        self.p1.x = 4
        # tp1 is modified (1)
        # eq_part12 is modified (1)
        # eq_part1 is modified (3)
        # mv_result is modified (1)
        # equation is modified two times (5)
        self.assertAlmostEqual(equation.x, 30)  # Expected x value after modification
        self.assertEqual(self.hook_counts['tp1'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['eq_part12'], 1)
        self.assertEqual(self.hook_counts['eq_part1'], 3)
        self.assertEqual(self.hook_counts['mv_result'], 1)
        self.assertEqual(self.hook_counts['equation'], 5)  # Final equation should update four times

        # (5 + 1) * ([4, 2] - [3, 1]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # 6 * ([1, 1]) + ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # ([[1, 2], [3, 4]] @ [[2, 0], [1, 2]]) = [[4, 4], [10, 8]]
        # [[4, 4], [10, 8]] x [4, 2] = [24, 56]
        # [6, 6] + [24, 56] - 0 = [30, 62] - 0 = [30, 62]
        self.p2.y = 1
        # tp2 is modified (1)
        # eq_part12 is modified (2)
        # eq_part1 is modified (4)
        # equation is modified (6)
        self.assertAlmostEqual(equation.y, 62)  # Expected y value after modification
        self.assertEqual(self.hook_counts['tp2'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['eq_part12'], 2)
        self.assertEqual(self.hook_counts['eq_part1'], 4)
        self.assertEqual(self.hook_counts['equation'], 6)

        # Modify the Matrices and check results
        # (5 + 1) * ([4, 2] - [3, 1]) + ([[2, 1], [1, 3]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # 6 * ([1, 1]) + ([[2, 1], [1, 3]] @ [[2, 0], [1, 2]]) x [4, 2] - 0
        # ([[2, 1], [1, 3]] @ [[2, 0], [1, 2]]) = [[5, 2], [5, 6]]
        # [[5, 2], [5, 6]] x [4, 2] = [24, 36]
        # [6, 6] + [24, 36] - 0 = [30, 42] - 0 = [30, 42]
        self.m1.data = np.array([[2, 1], [1, 3]])
        # tm1 is modified (1)
        # matrix_product is modified (1)
        # mv_result is modified (2)
        # equation is modified (7)
        self.assertAlmostEqual(equation.x, 30)  # Expected x value after modification
        self.assertEqual(self.hook_counts['tm1'], 1)  # s1 changed once
        self.assertEqual(self.hook_counts['matrix_product'], 1)
        self.assertEqual(self.hook_counts['mv_result'], 2)
        self.assertEqual(self.hook_counts['equation'], 7)  # Final equation should update seven times

        # (5 + 1) * ([4, 2] - [3, 1]) + ([[2, 1], [1, 3]] @ [[1, 0], [0, 1]]) x [4, 2] - 0
        # 6 * ([1, 1]) + ([[2, 1], [1, 3]] @ [[1, 0], [0, 1]]) x [4, 2] - 0
        # ([[2, 1], [1, 3]] @ [[1, 0], [0, 1]]) = [[2, 1], [1, 3]]
        # [[2, 1], [1, 3]] x [4, 2] = [10, 10]
        # [6, 6] + [10, 10] - 0 = [16, 16] - 0 = [16, 16]
        self.m2.data = np.array([[1, 0], [0, 1]])
        # tm2 is modified (1)
        # matrix_product is modified (2)
        # mv_result is modified (3)
        # equation is modified (8)
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
        self.assertEqual(self.hook_counts['eq_part12'], 2)  # eq_part12 should update twice
        self.assertEqual(self.hook_counts['eq_part1'], 4)  # eq_part1 should update twice
        self.assertEqual(self.hook_counts['equation'], 8)  # Final equation should update seven times
    # end test_complex_equation

# end class TestComplexEquationIntegration


class TestAdditionalComplexEquations(unittest.TestCase):

    def setUp(self):
        """
        Set up the test case.
        """
        # Scalars
        self.s1 = Scalar(1)
        self.s2 = Scalar(5)
        self.s3 = Scalar(10)

        # TScalars
        self.ts1 = TScalar.tscalar(self.s1)
        self.ts2 = TScalar.tscalar(self.s2)
        self.ts3 = TScalar.tscalar(self.s3)

        # Points
        self.p1 = Point2D(2, 3)
        self.p2 = Point2D(4, 6)

        # Matrices
        self.m1 = Matrix2D(np.array([[1, 0], [0, 1]]))
        self.m2 = Matrix2D(np.array([[2, 1], [0, 3]]))

        # TPoints
        self.tp1 = TPoint2D.tpoint2d(self.p1)
        self.tp2 = TPoint2D.tpoint2d(self.p2)

        # TMatrices
        self.tm1 = TMatrix2D.tmatrix2d(self.m1)
        self.tm2 = TMatrix2D.tmatrix2d(self.m2)

        # Hook counters for debugging
        self.hook_counts = {
            'ts1': 0,
            'ts2': 0,
            'ts3': 0,
            'tp1': 0,
            'tp2': 0,
            'tm1': 0,
            'tm2': 0,
            'complex_expr': 0
        }

        # Register hooks
        self.ts1.on_change.subscribe(self.make_hook('ts1'))
        self.ts2.on_change.subscribe(self.make_hook('ts2'))
        self.ts3.on_change.subscribe(self.make_hook('ts3'))
        self.tp1.on_change.subscribe(self.make_hook('tp1'))
        self.tp2.on_change.subscribe(self.make_hook('tp2'))
        self.tm1.on_change.subscribe(self.make_hook('tm1'))
        self.tm2.on_change.subscribe(self.make_hook('tm2'))
    # end setUp

    def make_hook(self, key):
        """
        Factory method to create hooks for a specific key.
        """
        def hook(sender, event_type, **kwargs):
            self.hook_counts[key] += 1
        return hook
    # end make_hook

    def test_tscalar_operations(self):
        """
        Test operations involving TScalars.
        """
        # Create an equation using TScalar
        # (1 + 5) / 10
        # Equation: (ts1 + ts2) / ts3
        complex_expr = (self.ts1 + self.ts2) / self.ts3
        complex_expr.on_change.subscribe(self.make_hook('complex_expr'))

        # Initial checks
        # (1 + 5) / 10 = 6 / 10 = 0.6
        self.assertAlmostEqual(complex_expr.value, 0.6)

        # Change scalar s1 to 4
        self.s1.value = 4
        # (4 + 5) / 10 = 9 / 10 = 0.9
        self.assertAlmostEqual(complex_expr.value, 0.9)
        self.assertEqual(self.hook_counts['ts1'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 1)

        # Change scalar s3 to 3
        self.s3.value = 3
        # (4 + 5) / 3 = 9 / 3 = 3
        self.assertAlmostEqual(complex_expr.value, 3)
        self.assertEqual(self.hook_counts['ts3'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 2)
    # end test_tscalar_operations

    def test_point_matrix_operations(self):
        """
        Test operations involving TPoints and TMatrices.
        """
        # Create an equation involving a point and matrix multiplication
        # Equation: tm1 @ tp1 + tp2
        matrix_vector_result = TMatrix2D.mv(self.tm1, self.tp1) + self.tp2
        matrix_vector_result.on_change.subscribe(self.make_hook('complex_expr'))

        # Initial checks
        # [[1, 0], [0, 1]] @ [2, 3] = [2, 3], [2, 3] + [4, 6] = [6, 9]
        self.assertAlmostEqual(matrix_vector_result.x, 6)
        self.assertAlmostEqual(matrix_vector_result.y, 9)

        # Change p1.x to 5
        self.p1.x = 5
        # [[1, 0], [0, 1]] @ [5, 3] = [5, 3], [5, 3] + [4, 6] = [9, 9]
        self.assertAlmostEqual(matrix_vector_result.x, 9)
        self.assertAlmostEqual(matrix_vector_result.y, 9)
        self.assertEqual(self.hook_counts['tp1'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 1)

        # Modify matrix tm1
        self.m1.data = np.array([[2, 0], [1, 1]])
        # [[2, 0], [1, 1]] @ [5, 3] = [10, 8], [10, 8] + [4, 6] = [14, 14]
        self.assertAlmostEqual(matrix_vector_result.x, 14)
        self.assertAlmostEqual(matrix_vector_result.y, 14)
        self.assertEqual(self.hook_counts['tm1'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 2)
    # end test_point_matrix_operations

    def test_mixed_equation(self):
        """
        Test a complex equation involving TScalars, TPoints, and TMatrices.
        """
        # Equation: (ts1 * ts2) + (tm1 @ tp1) - tp2
        complex_expr = (self.ts1 * self.ts2) + (TMatrix2D.mv(self.tm1, self.tp1)) - self.tp2
        complex_expr.on_change.subscribe(self.make_hook('complex_expr'))

        # Initial checks
        # (1 * 5) + ([[1, 0], [0, 1]] @ [2, 3]) - [4, 6] = 5 + [2, 3] - [4, 6] = [3, 2]
        self.assertAlmostEqual(complex_expr.x, 3)
        self.assertAlmostEqual(complex_expr.y, 2)

        # Change scalar s2 to 6
        self.s2.value = 6
        # (1 * 6) + ([[1, 0], [0, 1]] @ [2, 3]) - [4, 6] = 6 + [2, 3] - [4, 6] = [4, 3]
        self.assertAlmostEqual(complex_expr.x, 4)
        self.assertAlmostEqual(complex_expr.y, 3)
        self.assertEqual(self.hook_counts['ts2'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 1)

        # Change point tp1.x to 5
        self.p1.x = 5
        # (1 * 6) + ([[1, 0], [0, 1]] @ [5, 3]) - [4, 6] = 6 + [5, 3] - [4, 6] = [7, 3]
        self.assertAlmostEqual(complex_expr.x, 7)
        self.assertAlmostEqual(complex_expr.y, 3)
        self.assertEqual(self.hook_counts['tp1'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 2)

        # Modify the matrix tm1
        self.m1.data = np.array([[3, 0], [0, 2]])
        # (1 * 6) + ([[3, 0], [0, 2]] @ [5, 3]) - [4, 6] = 6 + [15, 6] - [4, 6] = [17, 6]
        self.assertAlmostEqual(complex_expr.x, 17)
        self.assertAlmostEqual(complex_expr.y, 6)
        self.assertEqual(self.hook_counts['tm1'], 1)
        self.assertEqual(self.hook_counts['complex_expr'], 3)
    # end test_mixed_equation

# end class TestAdditionalComplexEquations


if __name__ == '__main__':
    unittest.main()
# end if