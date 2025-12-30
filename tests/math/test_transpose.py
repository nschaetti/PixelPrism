"""Tests for pixelprism.math.Transpose."""

from pixelprism.math import Transpose
from tests.math.utils import ValueExpr, make_value


def test_transpose_swaps_axes():
    """Ensure Transpose permutes axes using python fallbacks."""
    value = make_value([[1, 2], [3, 4]], (2, 2))
    expr = Transpose(ValueExpr(value), (1, 0))
    result = expr.evaluate({})
    assert result.get() == [[1, 3], [2, 4]]
    graph = expr.graph()
    root = next(node for node in graph["nodes"] if node["id"] == graph["root"])
    assert root["params"]["perm"] == (1, 0)
# end def test_transpose_swaps_axes
