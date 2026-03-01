import pytest

import pixelprism.math as pm
from pixelprism.math.math_exceptions import SymbolicMathRuntimeError
from pixelprism.math.tensor import tensor


def test_constants_namespace_is_exposed_from_pm():
    assert hasattr(pm, "constants")
    assert "PI" in pm.constants.ALL
    assert "E" in pm.constants.ALL
    assert pm.constants.PI.name == "pi"
    assert pm.constants.E.name == "e"
# end def test_constants_namespace_is_exposed_from_pm


def test_all_registered_constants_are_symbolic_and_locked():
    for const in pm.constants.ALL.values():
        assert isinstance(const, pm.SymbolicConstant)
        assert const.is_constant_leaf() is True
        assert const.is_foldable() is False
# end def test_all_registered_constants_are_symbolic_and_locked


def test_symbolic_constant_cannot_be_mutated():
    with pytest.raises(SymbolicMathRuntimeError):
        pm.constants.PI.set(tensor(data=0.0, dtype=pm.DType.R, mutable=False))
# end def test_symbolic_constant_cannot_be_mutated


def test_common_aliases_reference_same_objects():
    assert pm.constants.GOLDEN_RATIO is pm.constants.PHI
    assert pm.constants.GAMMA is pm.constants.EULER_GAMMA
    assert pm.constants.ZETA3 is pm.constants.APERY
# end def test_common_aliases_reference_same_objects


def test_symbolic_constants_work_in_expressions_and_matching():
    x = pm.var("x", dtype=pm.DType.R, shape=())
    expr = x + pm.constants.PI

    simplified = expr.simplify()

    with pm.new_context():
        pm.set_value("x", 2.0)
        expr_value = expr.eval().value.item()
        simplified_value = simplified.eval().value.item()

    assert expr_value == pytest.approx(2.0 + float(pm.constants.PI.eval().value.item()), rel=1e-6)
    assert simplified_value == pytest.approx(expr_value, rel=1e-6)

    pattern = pm.P.n(
        "add",
        pm.P.v("x", as_="x_var"),
        pm.P.c(value=float(pm.constants.PI.eval().value.item()), as_="pi_const"),
        comm=True,
        as_="root",
    )
    match_result = expr.match(pattern)

    assert match_result.matched is True
    assert "x_var" in match_result.bindings
    assert "pi_const" in match_result.bindings
    assert "root" in match_result.bindings
# end def test_symbolic_constants_work_in_expressions_and_matching


def test_symbolic_constant_does_not_break_sub_mul_simplify():
    x = pm.var("x", dtype=pm.DType.R, shape=())
    expr = (pm.constants.E * x) - pm.constants.PHI
    simplified = expr.simplify()

    with pm.new_context():
        pm.set_value("x", 3.0)
        expr_value = expr.eval().value.item()
        simplified_value = simplified.eval().value.item()

    assert simplified_value == pytest.approx(expr_value, rel=1e-6)
# end def test_symbolic_constant_does_not_break_sub_mul_simplify
