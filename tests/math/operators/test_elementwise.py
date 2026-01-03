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

from pixelprism.math.functional.elementwise import (
    add,
    sub,
    mul,
    div,
    pow as pow_op,
    exp as exp_op,
    log as log_op,
    log2 as log2_op,
    log10 as log10_op,
    sqrt as sqrt_op,
    neg,
)
from ._helpers import (
    SCALAR_KINDS,
    TENSOR_SCALAR_KINDS,
    UNARY_SCALAR_KINDS,
    as_expected_array as _as_expected_array,
    make_tensor as _make_tensor,
    make_tensor_array as _make_tensor_array,
    operand_factory as _operand_factory,
)

OP_CASES = (
    ("add", add, np.add),
    ("sub", sub, np.subtract),
    ("mul", mul, np.multiply),
    ("div", div, np.divide),
)

UNARY_CASES = (
    ("exp", exp_op, np.exp, 1.5),
    ("log", log_op, np.log, 2.5),
    ("log2", log2_op, np.log2, 2.5),
    ("log10", log10_op, np.log10, 2.5),
    ("sqrt", sqrt_op, np.sqrt, 4.0),
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


@pytest.mark.parametrize("lhs_kind", SCALAR_KINDS)
@pytest.mark.parametrize("rhs_kind", SCALAR_KINDS)
def test_pow_scalar_combinations(lhs_kind, rhs_kind):
    """
    Validate pow for scalar-like operand combinations.
    """
    base = _operand_factory(lhs_kind, 5.0)
    exponent = _operand_factory(rhs_kind, 2.0)

    expr = pow_op(base, exponent)
    expected = np.power(_as_expected_array(base), _as_expected_array(exponent))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_pow_scalar_combinations


def test_pow_tensor_tensor_same_shape():
    """
    Validate pow on same-shape tensors.
    """
    left = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "left_pow")
    right = _make_tensor_array([[2.0, 3.0], [1.0, 0.5]], "right_pow")

    expr = pow_op(left, right)
    expected = np.power(left.eval(), right.eval())
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_pow_tensor_tensor_same_shape


@pytest.mark.parametrize("scalar_kind", TENSOR_SCALAR_KINDS)
def test_pow_tensor_scalar_combinations(scalar_kind):
    """
    Validate pow when mixing tensor and scalar operands.
    """
    tensor = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "tensor_pow")
    scalar = _operand_factory(scalar_kind, 3.0)

    expr = pow_op(tensor, scalar)
    expected = np.power(tensor.eval(), _as_expected_array(scalar))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_pow_tensor_scalar_combinations


def test_pow_dunder_on_math_expr():
    """
    Ensure MathExpr.__pow__ delegates to the pow operator.
    """
    tensor = _make_tensor(3.0, "value_pow")
    expr = tensor ** 2.0
    np.testing.assert_allclose(expr.eval(), np.array(9.0, dtype=np.float32))
# end test test_pow_dunder_on_math_expr


@pytest.mark.parametrize("kind", UNARY_SCALAR_KINDS)
@pytest.mark.parametrize("label, op_func, np_func, value", UNARY_CASES)
def test_unary_scalar_operators(kind, label, op_func, np_func, value):
    """
    Validate exp/log/sqrt for scalar-like operands.
    """
    operand = _operand_factory(kind, value)
    expr = op_func(operand)
    expected = np_func(_as_expected_array(operand))
    np.testing.assert_allclose(expr.eval(), expected)
# end test test_unary_scalar_operators


def test_unary_tensor_operators():
    """
    Validate exp/log/sqrt for tensor inputs.
    """
    tensor = _make_tensor_array([[1.5, 2.5], [3.5, 4.5]], "tensor_unary")
    cases = (
        (exp_op, np.exp),
        (log_op, np.log),
        (sqrt_op, np.sqrt),
    )
    for op_func, np_func in cases:
        expr = op_func(tensor)
        expected = np_func(tensor.eval())
        np.testing.assert_allclose(expr.eval(), expected)
    # end for
# end test test_unary_tensor_operators
