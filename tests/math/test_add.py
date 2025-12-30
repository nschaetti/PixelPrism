"""Tests for pixelprism.math.Add."""

from pixelprism.math import Add, Shape
from tests.math.utils import FakeTensor, ValueExpr, make_value


def test_add_evaluates_and_preserves_shape():
    """Ensure Add merges shapes and adds runtime values."""
    left = make_value(FakeTensor(2), (2, 2))
    right = make_value(FakeTensor(3), (2, 2))
    expr = Add(ValueExpr(left), ValueExpr(right))
    result = expr.evaluate({})
    assert isinstance(result.get(), FakeTensor)
    assert result.get().value == 5
    assert result.shape.as_tuple() == (2, 2)
    assert expr.graph()["nodes"]
# end def test_add_evaluates_and_preserves_shape

