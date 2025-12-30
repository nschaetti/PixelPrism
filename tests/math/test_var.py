"""Tests for pixelprism.math.Var."""

from pixelprism.math import Shape, Value, Var


def test_var_evaluation_and_graph():
    """Ensure Var pulls values from environment and exposes metadata."""
    shape = Shape((2,))
    var = Var("x", shape, "float")
    value = Value([1, 2], shape, "float")
    env = {var: value}
    assert var.evaluate(env) is value

    graph = var.graph()
    node = next(node for node in graph["nodes"] if node["id"] == graph["root"])
    assert node["params"]["name"] == "x"
# end def test_var_evaluation_and_graph

