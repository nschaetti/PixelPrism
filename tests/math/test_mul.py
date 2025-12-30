"""Tests for pixelprism.math.Mul."""

from pixelprism.math import Mul
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_mul_evaluates_correctly():
    """Ensure Mul multiplies runtime values elementwise."""
    left = make_value(FakeTensor(2), (2, 2))
    right = make_value(FakeTensor(4), (2, 2))
    expr = Mul(ValueExpr(left), ValueExpr(right))
    result = expr.evaluate({})
    assert result.get().value == 8
# end def test_mul_evaluates_correctly

