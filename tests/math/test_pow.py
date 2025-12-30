"""Tests for pixelprism.math.Pow."""

from pixelprism.math import Pow
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_pow_evaluates_correctly():
    """Ensure Pow exponentiates runtime values."""
    base = make_value(FakeTensor(2), (2, 2))
    exp = make_value(FakeTensor(3), (2, 2))
    expr = Pow(ValueExpr(base), ValueExpr(exp))
    result = expr.evaluate({})
    assert result.get().value == 8
# end def test_pow_evaluates_correctly

