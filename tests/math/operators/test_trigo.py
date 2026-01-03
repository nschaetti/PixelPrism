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
import numpy as np
import pytest

from pixelprism.math.functional.trigo import (
    asin as asin_op,
    asinh as asinh_op,
    acos as acos_op,
    acosh as acosh_op,
    atan as atan_op,
    atan2 as atan2_op,
    atanh as atanh_op,
    cos as cos_op,
    cosh as cosh_op,
    cot as cot_op,
    csc as csc_op,
    sec as sec_op,
    sin as sin_op,
    sinh as sinh_op,
    tan as tan_op,
    tanh as tanh_op,
)
from ._helpers import (
    SCALAR_KINDS,
    TENSOR_SCALAR_KINDS,
    UNARY_SCALAR_KINDS,
    as_expected_array as _as_expected_array,
    make_tensor_array as _make_tensor_array,
    operand_factory as _operand_factory,
)

TRIG_UNARY_CASES = (
    ("sin", sin_op, np.sin, 0.5, [[0.1, 0.2], [0.3, 0.4]]),
    ("cos", cos_op, np.cos, 0.5, [[0.1, 0.2], [0.3, 0.4]]),
    ("tan", tan_op, np.tan, 0.2, [[0.05, 0.1], [0.2, 0.3]]),
    ("asin", asin_op, np.arcsin, 0.5, [[0.1, -0.2], [0.3, -0.4]]),
    ("acos", acos_op, np.arccos, 0.5, [[0.1, -0.2], [0.3, -0.4]]),
    ("atan", atan_op, np.arctan, 0.5, [[0.1, 0.2], [0.3, 0.4]]),
    ("sec", sec_op, lambda x: 1.0 / np.cos(x), 1.0, [[0.1, 0.2], [0.3, 0.4]]),
    ("csc", csc_op, lambda x: 1.0 / np.sin(x), 1.0, [[0.2, 0.3], [0.4, 0.5]]),
    ("cot", cot_op, lambda x: 1.0 / np.tan(x), 1.0, [[0.2, 0.3], [0.4, 0.5]]),
    ("sinh", sinh_op, np.sinh, 0.5, [[0.1, 0.2], [0.3, 0.4]]),
    ("cosh", cosh_op, np.cosh, 0.5, [[0.1, 0.2], [0.3, 0.4]]),
    ("tanh", tanh_op, np.tanh, 0.5, [[0.1, 0.2], [0.3, 0.4]]),
    ("asinh", asinh_op, np.arcsinh, 0.5, [[0.1, -0.2], [0.3, -0.4]]),
    ("acosh", acosh_op, np.arccosh, 1.5, [[1.5, 2.0], [2.5, 3.0]]),
    ("atanh", atanh_op, np.arctanh, 0.2, [[0.1, 0.2], [0.3, 0.4]]),
)


@pytest.mark.parametrize("kind", UNARY_SCALAR_KINDS)
@pytest.mark.parametrize("label, op_func, np_func, scalar_value, _", TRIG_UNARY_CASES)
def test_trig_unary_scalar_operators(kind, label, op_func, np_func, scalar_value, _):
    """
    Validate trig unary operators for scalar-like operands.
    """
    operand = _operand_factory(kind, scalar_value)
    expr = op_func(operand)
    expected = np_func(_as_expected_array(operand))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_trig_unary_scalar_operators


@pytest.mark.parametrize("label, op_func, np_func, _, tensor_values", TRIG_UNARY_CASES)
def test_trig_unary_tensor_operators(label, op_func, np_func, _, tensor_values):
    """
    Validate trig unary operators for tensor operands.
    """
    tensor = _make_tensor_array(tensor_values, f"tensor_trig_{label}")
    expr = op_func(tensor)
    expected = np_func(tensor.eval())
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_trig_unary_tensor_operators


@pytest.mark.parametrize("lhs_kind", SCALAR_KINDS)
@pytest.mark.parametrize("rhs_kind", SCALAR_KINDS)
def test_atan2_scalar_combinations(lhs_kind, rhs_kind):
    """
    Validate atan2 for scalar-like operand combinations.
    """
    y = _operand_factory(lhs_kind, 0.5)
    x = _operand_factory(rhs_kind, 0.3)
    expr = atan2_op(y, x)
    expected = np.arctan2(_as_expected_array(y), _as_expected_array(x))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_atan2_scalar_combinations


def test_atan2_tensor_tensor_same_shape():
    """
    Validate atan2 on same-shape tensors.
    """
    y = _make_tensor_array([[0.1, 0.2], [0.3, 0.4]], "atan2_y")
    x = _make_tensor_array([[0.5, 0.6], [0.7, 0.8]], "atan2_x")
    expr = atan2_op(y, x)
    expected = np.arctan2(y.eval(), x.eval())
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_atan2_tensor_tensor_same_shape


@pytest.mark.parametrize("scalar_kind", TENSOR_SCALAR_KINDS)
def test_atan2_tensor_scalar_combinations(scalar_kind):
    """
    Validate atan2 mixing tensor and scalar operands.
    """
    y = _make_tensor_array([[0.1, 0.2], [0.3, 0.4]], "atan2_tensor")
    x_scalar = _operand_factory(scalar_kind, 0.5)
    expr = atan2_op(y, x_scalar)
    expected = np.arctan2(y.eval(), _as_expected_array(x_scalar))
    np.testing.assert_allclose(expr.eval(), expected)

    expr2 = atan2_op(x_scalar, y)
    expected2 = np.arctan2(_as_expected_array(x_scalar), y.eval())
    np.testing.assert_allclose(expr2.eval(), expected2)
# end test test_atan2_tensor_scalar_combinations
