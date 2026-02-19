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
from numpy.ma.core import equal

import pixelprism.math as pm
from pixelprism.math.dtype import to_numpy
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
    abs as abs_op,
    neg,
    exp2,
    expm1,
    cbrt,
    square,
    reciprocal,
    deg2rad,
    rad2deg,
    absolute,
)
from ._helpers import (
    SCALAR_KINDS,
    TENSOR_SCALAR_KINDS,
    UNARY_SCALAR_KINDS,
    assert_expr_allclose as _assert_expr_allclose,
    as_expected_array as _as_expected_array,
    eval_expr_value as _eval_expr_value,
    make_tensor as _make_tensor,
    make_tensor_array as _make_tensor_array,
operand_factory as _operand_factory,
)


def _const_expr(name, values, dtype):
    """Create a constant expression paired with its numpy array."""
    arr = np.asarray(values, dtype=to_numpy(dtype))
    return pm.const(name=name, data=arr, dtype=dtype), arr
# end def _const_expr


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
    _assert_expr_allclose(expr, expected)
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
    _assert_expr_allclose(expr, expected)
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
    _assert_expr_allclose(expr, np.array(-4.0, dtype=np.float32))
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
    expected = np_op(_eval_expr_value(left), _eval_expr_value(right))
    _assert_expr_allclose(expr, expected)
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
    expected = np_op(_as_expected_array(scalar), _eval_expr_value(tensor))
    _assert_expr_allclose(expr, expected)
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
    expected = np_op(_eval_expr_value(tensor), _as_expected_array(scalar))
    _assert_expr_allclose(expr, expected)
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
    _assert_expr_allclose(expr, expected)
# end test test_pow_scalar_combinations


def test_pow_tensor_tensor_same_shape():
    """
    Validate pow on same-shape tensors.
    """
    left = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "left_pow")
    right = _make_tensor_array([[2.0, 3.0], [1.0, 0.5]], "right_pow")

    expr = pow_op(left, right)
    expected = np.power(_eval_expr_value(left), _eval_expr_value(right))
    _assert_expr_allclose(expr, expected)
# end test test_pow_tensor_tensor_same_shape


@pytest.mark.parametrize("scalar_kind", TENSOR_SCALAR_KINDS)
def test_pow_tensor_scalar_combinations(scalar_kind):
    """
    Validate pow when mixing tensor and scalar operands.
    """
    tensor = _make_tensor_array([[1.0, 2.0], [3.0, 4.0]], "tensor_pow")
    scalar = _operand_factory(scalar_kind, 3.0)

    expr = pow_op(tensor, scalar)
    expected = np.power(_eval_expr_value(tensor), _as_expected_array(scalar))
    _assert_expr_allclose(expr, expected)
# end test test_pow_tensor_scalar_combinations


def test_pow_dunder_on_math_expr():
    """
    Ensure MathExpr.__pow__ delegates to the pow operator.
    """
    tensor = _make_tensor(3.0, "value_pow")
    expr = tensor ** 2.0
    _assert_expr_allclose(expr, np.array(9.0, dtype=np.float32))
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
    _assert_expr_allclose(expr, expected)
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
        expected = np_func(_eval_expr_value(tensor))
        _assert_expr_allclose(expr, expected)
    # end for
# end test test_unary_tensor_operators


def test_elementwise_tensor_tensor_mixed_dtype_and_shape():
    left, left_np = _const_expr(
        "left_mixed",
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]],
        pm.DType.R,
    )
    right, right_np = _const_expr(
        "right_mixed",
        [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
        pm.DType.R,
    )

    expr = add(left, right)
    expected = left_np + right_np

    _assert_expr_allclose(expr, expected)
    assert expr.dtype == pm.DType.R
    assert tuple(expr.shape.dims) == expected.shape
# end test_elementwise_tensor_tensor_mixed_dtype_and_shape


def test_elementwise_scalar_tensor_dtype_promotion():
    tensor_expr, tensor_np = _const_expr(
        "int_tensor",
        [[1, 2], [3, 4]],
        pm.DType.Z,
    )
    scalar_expr, scalar_np = _const_expr("float_scalar", 2.5, pm.DType.R)

    expr = mul(tensor_expr, scalar_expr)
    expected = tensor_np * scalar_np

    _assert_expr_allclose(expr, expected)
    assert expr.dtype == pm.DType.R
    assert expr.shape.dims == tensor_expr.shape.dims
# end test_elementwise_scalar_tensor_dtype_promotion


def test_elementwise_combined_expression_chain():
    a_expr, a_np = _const_expr(
        "chain_a",
        [[1.0, 2.0], [3.0, 4.0]],
        pm.DType.R,
    )
    b_expr, b_np = _const_expr(
        "chain_b",
        [[0.5, 1.5], [2.5, 3.5]],
        pm.DType.R,
    )
    c_expr, c_np = _const_expr(
        "chain_c",
        [[4.0, 5.0], [6.0, 7.0]],
        pm.DType.R,
    )
    d_expr, d_np = _const_expr(
        "chain_d",
        [[2.0, 3.0], [4.0, 5.0]],
        pm.DType.R,
    )

    combined = sqrt_op(add(mul(a_expr, b_expr), div(c_expr, d_expr)))
    expected = np.sqrt((a_np * b_np) + (c_np / d_np))

    _assert_expr_allclose(combined, expected)
    assert tuple(combined.shape.dims) == expected.shape
# end test_elementwise_combined_expression_chain


def test_element_wise_op_scalar():
    # Variables
    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())

    # Math equations
    z1 = add(x, y)
    z2 = sub(x, y)
    z3 = mul(x, y)
    z4 = div(x, y)
    z5 = pow(x, y)
    z6 = sqrt_op(x)
    z7 = abs_op(x)
    z8 = exp_op(x)
    z9 = exp2(x)
    z10 = expm1(x)
    z11 = log_op(x)
    z12 = log2_op(x)
    z13 = log10_op(x)
    z14 = cbrt(x)
    z15 = square(x)
    z16 = reciprocal(x)
    z17 = deg2rad(x)
    z18 = rad2deg(x)
    z20 = absolute(x)
    z21 = neg(x)
    z22 = sub(y, z4)
    z23 = mul(z1, z22)
    z24 = div(z1, z22)
    z25 = mul(z12, z9)

    # Set value and evaluate
    with pm.new_context():
        pm.set_value("x", 2.4)
        pm.set_value("y", 2.2)
        assert z1.eval().value == pytest.approx(4.6, abs=1e-6)
        assert z2.eval().value == pytest.approx(0.2, abs=1e-6)
        assert z3.eval().value == pytest.approx(5.28, abs=1e-6)
        assert z4.eval().value == pytest.approx(1.0909091234207153, abs=1e-6)
        assert z5.eval().value == pytest.approx(6.862222194671631, abs=1e-6)
        assert z6.eval().value == pytest.approx(1.5491933822631836, abs=1e-6)
        assert z7.eval().value == pytest.approx(2.4, abs=1e-6)
        assert z8.eval().value == pytest.approx(11.023177146911621, abs=1e-6)
        assert z9.eval().value == pytest.approx(5.278031826019287, abs=1e-6)
        assert z10.eval().value == pytest.approx(10.023177146911621, abs=1e-6)
        assert z11.eval().value == pytest.approx(0.8754687905311584, abs=1e-6)
        assert z12.eval().value == pytest.approx(1.263034462928772, abs=1e-6)
        assert z13.eval().value == pytest.approx(0.3802112638950348, abs=1e-6)
        assert z14.eval().value == pytest.approx(1.338865876197815, abs=1e-6)
        assert z15.eval().value == pytest.approx(5.760000228881836, abs=1e-6)
        assert z16.eval().value == pytest.approx(0.4166666567325592, abs=1e-6)
        assert z17.eval().value == pytest.approx(0.04188790172338486, abs=1e-6)
        assert z18.eval().value == pytest.approx(137.50987243652344, abs=1e-6)
        assert z20.eval().value == pytest.approx(2.4, abs=1e-6)
        assert z21.eval().value == pytest.approx(-2.4, abs=1e-6)
        assert z22.eval().value == pytest.approx(1.1090909242630005, abs=1e-6)
        assert z23.eval().value == pytest.approx(5.101818561553955, abs=1e-6)
        assert z24.eval().value == pytest.approx(4.147541046142578, abs=1e-6)
        assert z25.eval().value == pytest.approx(6.6663360595703125, abs=1e-6)
    # end with
# end test test_element_wise_op_scalar
