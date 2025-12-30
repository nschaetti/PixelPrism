"""Tests for pixelprism.math.Concat."""

from pixelprism.math import Concat, Shape
from tests.math.utils import ValueExpr, make_value


def test_concat_along_axis():
    """Ensure Concat validates shapes and concatenates python data."""
    left = make_value([[1, 2]], (1, 2))
    right = make_value([[3, 4]], (1, 2))
    expr = Concat((ValueExpr(left), ValueExpr(right)), axis=0)
    assert expr.shape.as_tuple() == (2, 2)
    result = expr.evaluate({})
    assert result.get() == [[1, 2], [3, 4]]
    graph = expr.graph()
    root = next(node for node in graph["nodes"] if node["id"] == graph["root"])
    assert root["params"]["axis"] == 0
# end def test_concat_along_axis
