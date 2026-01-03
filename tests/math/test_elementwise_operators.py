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

from pixelprism.math.functional.elementwise import add, sub, mul, div, neg
from pixelprism.math.math_expr import MathExpr
from pixelprism.math.tensor import Tensor


def _make_tensor(value: float, name: str) -> Tensor:
    return Tensor(name=name, data=value)
# end def _make_tensor


def _make_tensor_array(values, name: str) -> Tensor:
    return Tensor(name=name, data=np.array(values, dtype=np.float32))
# end def _make_tensor_array


def _as_expected_array(value):
    if isinstance(value, MathExpr):
        return value.eval()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, (int, float, np.number, bool, complex)):
        return np.asarray(value, dtype=np.float64)
    raise TypeError(f"Unsupported operand type: {type(value)}")
# end def _as_expected_array


def _operand_factory(kind: str, value: float):
    if kind == "math_expr":
        return _make_tensor(value, f"t_{value}")
    if kind == "float":
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "np_scalar":
        return np.float32(value)
    if kind == "np_array_scalar":
        return np.array(value, dtype=np.float32)
    if kind == "list":
        return [value]
    raise ValueError(f"Unknown operand kind: {kind}")
# end def _operand_factory


SCALAR_KINDS = (
    "math_expr",
    "float",
    "int",
    "np_scalar",
    "np_array_scalar",
)

OP_CASES = (
    ("add", add, np.add),
    ("sub", sub, np.subtract),
    ("mul", mul, np.multiply),
    ("div", div, np.divide),
)


@pytest.mark.parametrize("op_name, op_func, np_op", OP_CASES)
@pytest.mark.parametrize("lhs_kind", SCALAR_KINDS)
@pytest.mark.parametrize("rhs_kind", SCALAR_KINDS)
def test_arithmetic_operator_scalar_combinations(op_name, op_func, np_op, lhs_kind, rhs_kind):
    """
    Validate arithmetic operators for scalar-like operand combinations.

    Returns
    -------
    None
    """
    lhs = _operand_factory(lhs_kind, 6.0)
    rhs = _operand_factory(rhs_kind, 2.0)

    expr = op_func(lhs, rhs)
    expected = np_op(_as_expected_array(lhs), _as_expected_array(rhs))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_arithmetic_operator_scalar_combinations


NEG_SCALAR_KINDS = (
    "math_expr",
    "float",
    "int",
    "np_scalar",
    "np_array_scalar"
)


@pytest.mark.parametrize("kind", NEG_SCALAR_KINDS)
def test_neg_operator_scalar_combinations(kind):
    """
    Validate negation for scalar-like operand combinations.

    Returns
    -------
    None
    """
    operand = _operand_factory(kind, 3.5)
    expr = neg(operand)
    expected = np.negative(_as_expected_array(operand))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_neg_operator_scalar_combinations


def test_neg_dunder_on_math_expr():
    """
    Ensure MathExpr.__neg__ delegates to the neg operator.

    Returns
    -------
    None
    """
    tensor = _make_tensor(4.0, "value")
    expr = -tensor
    np.testing.assert_allclose(expr.eval(), np.array(-4.0, dtype=np.float32))
# end test test_neg_dunder_on_math_expr


TENSOR_SCALAR_KINDS = (
    "float",
    "int",
    "np_scalar",
    "np_array_scalar",
)


@pytest.mark.parametrize("op_name, op_func, np_op", OP_CASES)
def test_elementwise_tensor_tensor_same_shape(op_name, op_func, np_op):
    """
    Validate elementwise operators on same-shape tensors.

    Returns
    -------
    None
    """
    left = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "left")
    right = _make_tensor_array([[5.0, 6.0], [7.0, 8.0]], "right")

    expr = op_func(left, right)
    expected = np_op(left.eval(), right.eval())
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_elementwise_tensor_tensor_same_shape


@pytest.mark.parametrize("op_name, op_func, np_op", OP_CASES)
@pytest.mark.parametrize("scalar_kind", TENSOR_SCALAR_KINDS)
def test_elementwise_scalar_tensor_combinations(op_name, op_func, np_op, scalar_kind):
    """
    Validate elementwise operators with scalar left operand.

    Returns
    -------
    None
    """
    scalar = _operand_factory(scalar_kind, 2.0)
    tensor = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "tensor")

    expr = op_func(scalar, tensor)
    expected = np_op(_as_expected_array(scalar), tensor.eval())
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_elementwise_scalar_tensor_combinations


@pytest.mark.parametrize("op_name, op_func, np_op", OP_CASES)
@pytest.mark.parametrize("scalar_kind", TENSOR_SCALAR_KINDS)
def test_elementwise_tensor_scalar_combinations(op_name, op_func, np_op, scalar_kind):
    """
    Validate elementwise operators with scalar right operand.

    Returns
    -------
    None
    """
    tensor = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "tensor")
    scalar = _operand_factory(scalar_kind, 2.0)

    expr = op_func(tensor, scalar)
    expected = np_op(tensor.eval(), _as_expected_array(scalar))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_elementwise_tensor_scalar_combinations
