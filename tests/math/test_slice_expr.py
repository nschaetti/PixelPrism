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

# Imports
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


def test_slice_expr_accepts_constant_expression():
    start = pm.const("slice_start_expr", data=1, dtype=pm.DType.INT64)
    offset = pm.const("slice_offset_expr", data=2, dtype=pm.DType.INT64)
    stop_base = pm.const("slice_stop_expr", data=4, dtype=pm.DType.INT64)
    expr = SliceExpr.create(start=start + offset, stop=offset + stop_base)

    assert expr.start_value == 3
    assert expr.stop_value == 6
    assert isinstance(expr.start, pm.MathNode)
# end test_slice_expr_accepts_constant_expression


def test_slice_expr_rejects_non_constant_expression():
    var = pm.var("slice_var", dtype=pm.DType.INT32, shape=())
    const = pm.const("slice_const", data=5, dtype=pm.DType.INT32)

    with pytest.raises(MathExprValidationError):
        SliceExpr.create(start=var + const)
# end test_slice_expr_rejects_non_constant_expression
