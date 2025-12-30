"""Tests for pixelprism.math.Const."""

from pixelprism.math import Const, Shape, Value


def test_const_returns_copies():
    """Ensure Const evaluation returns deep copies."""
    shape = Shape((1,))
    const = Const(Value([1], shape, "int"))
    value = const.evaluate({})
    other = const.evaluate({})
    assert value.get() == other.get()
    assert value is not other

    graph = const.graph()
    node = next(node for node in graph["nodes"] if node["id"] == graph["root"])
    assert node["params"]["constant"] is True
# end def test_const_returns_copies

