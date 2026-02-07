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

import numpy as np
import pytest
from itertools import count

import pixelprism.math as pm
from pixelprism.math.functional import discretization as D


_CONST_COUNTER = count()


def _make_value(values):
    """Create a Constant expression for the provided values."""
    idx = next(_CONST_COUNTER)
    return pm.const(f"disc_const_{idx}", data=values, dtype=pm.DType.R)
# end def _make_value


def _round_decimals_cases():
    """Build a variety of decimal parameter inputs for the round operator."""
    def _var_case():
        var = pm.var("round_dec_var", dtype=pm.DType.Z, shape=())

        def _ctx():
            pm.set_value("round_dec_var", 3)
        # end def _ctx

        return var, _ctx, 3
    # end def _var_case

    return (
        ("python_int", lambda: (1, None, 1)),
        ("constant", lambda: (
            pm.const("round_dec_const", data=2, dtype=pm.DType.Z),
            None,
            2
        )),
        ("variable", _var_case),
        ("math_expr", lambda: (
            pm.const("round_dec_a", data=1, dtype=pm.DType.Z) +
            pm.const("round_dec_b", data=2, dtype=pm.DType.Z) -
            pm.const("round_dec_c", data=1, dtype=pm.DType.Z),
            None,
            2
        )),
    )
# end def _round_decimals_cases


def _clip_bound_cases():
    """Build combinations of clip bounds with different parameter types."""
    def _var_bounds():
        vmin = pm.var("clip_min_var", dtype=pm.DType.R, shape=())
        vmax = pm.var("clip_max_var", dtype=pm.DType.R, shape=())

        def _ctx():
            pm.set_value("clip_min_var", -0.25)
            pm.set_value("clip_max_var", 0.6)
        # end def _ctx

        return vmin, vmax, _ctx, -0.25, 0.6
    # end def _var_bounds

    return (
        ("python_min", lambda: (-0.5, None, None, -0.5, None)),
        ("constant_max", lambda: (
            None,
            pm.const("clip_max_const", data=0.75, dtype=pm.DType.R),
            None,
            None,
            0.75
        )),
        ("variable_bounds", _var_bounds),
        ("math_expr_bounds", lambda: (
            pm.const("clip_min_expr_a", data=-1.0, dtype=pm.DType.R) +
            pm.const("clip_min_expr_b", data=0.1, dtype=pm.DType.R),
            pm.const("clip_max_expr_a", data=1.0, dtype=pm.DType.R) -
            pm.const("clip_max_expr_b", data=0.2, dtype=pm.DType.R),
            None,
            -0.9,
            0.8
        )),
    )
# end def _clip_bound_cases


def test_basic_discretization_ops_match_numpy():
    """
    Verify sign/floor/ceil/trunc/rint align with numpy for vector inputs.
    """
    data = _make_value([-1.5, -0.1, 0.0, 1.2, 3.7])
    checks = [
        (D.sign, np.sign),
        (D.floor, np.floor),
        (D.ceil, np.ceil),
        (D.trunc, np.trunc),
        (D.rint, np.rint),
    ]
    for func, np_func in checks:
        expr = func(data)
        np.testing.assert_allclose(expr.eval().value, np_func(data.eval().value))
    # end for
# end test_basic_discretization_ops_match_numpy


def test_round_operator_supports_decimals():
    """
    Ensure functional.round forwards decimal precision to Tensor.round.
    """
    data = _make_value([1.234, -1.266])
    expr = D.round(data, decimals=2)
    np.testing.assert_allclose(expr.eval().value, np.round(data.eval().value, 2))
# end test_round_operator_supports_decimals


def test_clip_operator_bounds():
    """
    Validate clip handles min-only, max-only, and combined bounds.
    """
    data = _make_value([-2.0, -0.5, 0.5, 2.0])
    mid = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)

    lower_expr = D.clip(data, min_value=-0.25)
    np.testing.assert_allclose(lower_expr.eval().value, np.clip(mid, -0.25, None))

    upper_expr = D.clip(data, max_value=0.75)
    np.testing.assert_allclose(upper_expr.eval().value, np.clip(mid, None, 0.75))

    both_expr = D.clip(data, min_value=-0.25, max_value=0.75)
    np.testing.assert_allclose(
        both_expr.eval().value,
        np.clip(mid, -0.25, 0.75)
    )
# end test_clip_operator_bounds


@pytest.mark.parametrize("case_name, case_builder", _round_decimals_cases())
def test_round_operator_parameter_types(case_name, case_builder):
    """
    Ensure round consumes decimals from ints, constants, variables, and generic expressions.
    """
    decimals_param, ctx_setup, expected_decimals = case_builder()
    data = _make_value([1.234, -1.266, 2.555])
    expr = D.round(data, decimals=decimals_param)

    with pm.new_context():
        if ctx_setup is not None:
            ctx_setup()
        np.testing.assert_allclose(
            expr.eval().value,
            np.round(data.eval().value, expected_decimals)
        )
    # end with
# end test_round_operator_parameter_types


_ROUND_INVALID_CASES = (
    ("string", lambda: "invalid", TypeError),
    ("float", lambda: 1.5, TypeError),
    ("vector_expr", lambda: pm.const(
        "round_bad_vector",
        data=[1, 2],
        dtype=pm.DType.Z
    ), ValueError),
)


@pytest.mark.parametrize("case_name, builder, expected_exc", _ROUND_INVALID_CASES)
def test_round_invalid_decimals_inputs(case_name, builder, expected_exc):
    data = _make_value([1.0, -1.5])
    with pytest.raises(expected_exc):
        expr = D.round(data, decimals=builder())
        expr.eval()
    # end with
# end test_round_invalid_decimals_inputs


@pytest.mark.parametrize("case_name, case_builder", _clip_bound_cases())
def test_clip_operator_parameter_types(case_name, case_builder):
    """
    Clip should accept ints, constants, variables, and scalar expressions for bounds.
    """
    min_param, max_param, ctx_setup, expected_min, expected_max = case_builder()
    data = _make_value([-1.5, -0.1, 0.2, 1.75])
    expr = D.clip(data, min_value=min_param, max_value=max_param)

    with pm.new_context():
        if ctx_setup is not None:
            ctx_setup()
        np.testing.assert_allclose(
            expr.eval().value,
            np.clip(data.eval().value, expected_min, expected_max)
        )
    # end with
# end test_clip_operator_parameter_types


def test_clip_invalid_bounds_raise():
    data = _make_value([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        D.clip(data)

    with pytest.raises(TypeError):
        D.clip(data, min_value="low")

    with pytest.raises(ValueError):
        bad_bound = pm.const("clip_bad_bound", data=[-1.0, 0.0], dtype=pm.DType.R)
        D.clip(data, min_value=bad_bound).eval()
# end test_clip_invalid_bounds_raise


def test_math_expr_discretization_ops_scalar():
    # Variables
    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())

    # Math equations
    z1 = D.ceil(x + y)
    z2 = D.floor(x + y)
    z3 = D.round(x + y, 1)
    z4 = D.trunc(x + y)
    z5 = D.sign(x + y)
    z6 = D.rint(x + y)
    z7 = D.clip(x + y, 0.0, 1.0)

    # Set value and evaluate
    with pm.new_context():
        pm.set_value("x", 2.4)
        pm.set_value("y", 2.2)
        assert z1.eval().value == 5.0
        assert z2.eval().value == 4.0
        assert z3.eval().value == 4.6
        assert z4.eval().value == 4.0
        assert z5.eval().value == 1.0
        assert z6.eval().value == 5.0
        assert z7.eval().value == 1.0
    # end with
# end test test_math_expr_discretization_ops


def test_math_expr_discretization_ops_vector_float64():
    x = pm.var("vx", dtype=pm.DType.R, shape=(3,))
    y = pm.var("vy", dtype=pm.DType.R, shape=(3,))

    floored = D.floor(x - y)
    ceiled = D.ceil(x + y)
    rounded = D.round(x + y, 2)
    clipped = D.clip(rounded, min_value=-0.25, max_value=1.5)
    rinted = D.rint(y - x)

    with pm.new_context():
        x_val = np.array([-1.25, 0.1, 2.4], dtype=np.float64)
        y_val = np.array([0.5, -0.3, 1.1], dtype=np.float64)
        pm.set_value("vx", x_val)
        pm.set_value("vy", y_val)

        np.testing.assert_allclose(floored.eval().value, np.floor(x_val - y_val))
        np.testing.assert_allclose(ceiled.eval().value, np.ceil(x_val + y_val))
        np.testing.assert_allclose(
            clipped.eval().value,
            np.clip(np.round(x_val + y_val, 2), -0.25, 1.5),
        )
        np.testing.assert_allclose(rinted.eval().value, np.rint(y_val - x_val))
    # end with
# end test test_math_expr_discretization_ops_vector_float64


def test_math_expr_discretization_ops_matrix_int32():
    a = pm.var("ma", dtype=pm.DType.Z, shape=(2, 2))
    b = pm.var("mb", dtype=pm.DType.Z, shape=(2, 2))

    sign_expr = D.sign(a - b)
    trunc_expr = D.trunc(a + b)
    clip_expr = D.clip(D.ceil(a - b), min_value=-2, max_value=2)
    rint_expr = D.rint(a + b)

    with pm.new_context():
        a_val = np.array(
            [
                [-2, -1],
                [0, 3],
            ],
            dtype=np.int32,
        )
        b_val = np.array(
            [
                [1, -2],
                [2, -1],
            ],
            dtype=np.int32,
        )
        pm.set_value("ma", a_val)
        pm.set_value("mb", b_val)

        np.testing.assert_array_equal(sign_expr.eval().value, np.sign(a_val - b_val))
        np.testing.assert_array_equal(trunc_expr.eval().value, np.trunc(a_val + b_val))
        np.testing.assert_array_equal(
            clip_expr.eval().value,
            np.clip(np.ceil(a_val - b_val), -2, 2),
        )
        np.testing.assert_array_equal(rint_expr.eval().value, np.rint(a_val + b_val))
    # end with
# end test test_math_expr_discretization_ops_matrix_int32
