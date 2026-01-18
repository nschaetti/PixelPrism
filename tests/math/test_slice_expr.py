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
# Copyright (C) 2025 Pixel Prism
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

import pytest

import pixelprism.math as pm
from pixelprism.math.math_expr import SliceExpr, MathExprValidationError


def test_slice_expr_from_ints():
    expr = SliceExpr.create(start=1, stop=5, step=2)

    assert expr.start_value == 1
    assert expr.stop_value == 5
    assert expr.step_value == 2
    assert expr.start.dtype == pm.DType.INT64
    assert isinstance(expr.as_slice, slice)
    assert expr.as_slice == slice(1, 5, 2)
# end test_slice_expr_from_ints


def test_slice_expr_from_constants():
    start = pm.const("slice_start", data=2, dtype=pm.DType.INT32)
    stop = pm.const("slice_stop", data=10, dtype=pm.DType.INT32)

    expr = SliceExpr.create(start=start, stop=stop)

    assert expr.start is start
    assert expr.stop is stop
    assert expr.step is None
    assert expr.as_slice == slice(2, 10, None)
# end test_slice_expr_from_constants


def test_slice_expr_rejects_non_integer_constant():
    bad_const = pm.const("slice_bad", data=3.14, dtype=pm.DType.FLOAT32)

    with pytest.raises(MathExprValidationError):
        SliceExpr.create(start=bad_const)
# end test_slice_expr_rejects_non_integer_constant
