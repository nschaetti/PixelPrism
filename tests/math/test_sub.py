"""Tests for pixelprism.math.Sub."""

from pixelprism.math import Sub
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_sub_evaluates_correctly():
    """Ensure Sub subtracts runtime values."""
    left = make_value(FakeTensor(5), (2, 2))
    right = make_value(FakeTensor(3), (2, 2))
    expr = Sub(ValueExpr(left), ValueExpr(right))
    result = expr.evaluate({})
    assert result.get().value == 2
# end def test_sub_evaluates_correctly

