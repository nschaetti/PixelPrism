"""Tests for pixelprism.math.Stack."""

from pixelprism.math import Stack
from tests.math.utils import ValueExpr, make_value


def test_stack_inserts_axis():
    """Ensure Stack inserts a new axis and stacks data."""
    tensors = [make_value([1, 2], (2,)), make_value([3, 4], (2,))]
    expr = Stack(tuple(ValueExpr(t) for t in tensors), axis=0)
    assert expr.shape.as_tuple() == (2, 2)
    result = expr.evaluate({})
    assert result.get() == [[1, 2], [3, 4]]
# end def test_stack_inserts_axis

