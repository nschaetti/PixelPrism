"""Tests for pixelprism.math.Reshape."""

from pixelprism.math import Reshape, Shape
from tests.math.utils import ValueExpr, make_value


def test_reshape_changes_shape():
    """Ensure Reshape updates symbolic shape and rearranges data."""
    value = make_value([[1, 2], [3, 4]], (2, 2))
    target = Shape((4,))
    expr = Reshape(ValueExpr(value), target)
    result = expr.evaluate({})
    assert result.get() == [1, 2, 3, 4]
    graph = expr.graph()
    root = next(node for node in graph["nodes"] if node["id"] == graph["root"])
    assert root["params"]["target"] == (4,)
# end def test_reshape_changes_shape
