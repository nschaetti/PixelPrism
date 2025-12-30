"""Tests for pixelprism.math.Div."""

from pixelprism.math import Div
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_div_evaluates_correctly():
    """Ensure Div divides runtime values elementwise."""
    left = make_value(FakeTensor(8), (2, 2))
    right = make_value(FakeTensor(2), (2, 2))
    expr = Div(ValueExpr(left), ValueExpr(right))
    result = expr.evaluate({})
    assert result.get().value == 4
# end def test_div_evaluates_correctly

