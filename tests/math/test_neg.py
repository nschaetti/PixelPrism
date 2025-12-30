"""Tests for pixelprism.math.Neg."""

from pixelprism.math import Neg
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_neg_evaluates_correctly():
    """Ensure Neg negates runtime values."""
    value = make_value(FakeTensor(5), (2, 2))
    expr = Neg(ValueExpr(value))
    result = expr.evaluate({})
    assert result.get().value == -5
# end def test_neg_evaluates_correctly

