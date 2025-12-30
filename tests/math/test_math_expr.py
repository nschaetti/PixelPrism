"""Tests for pixelprism.math.MathExpr."""

from __future__ import annotations

from typing import Mapping, Sequence

from pixelprism.math import MathExpr, Shape, Value


class DummyExpr(MathExpr):
    """Simple MathExpr implementation for testing."""

    def __init__(self, value: Value | None = None, children: Sequence[MathExpr] | None = None):
        """Initialize the dummy expression."""
        shape = value.shape if value is not None else Shape((1,))
        dtype = value.dtype if value is not None else "int"
        super().__init__(shape, dtype, children or ())
        self._value = value
    # end def __init__

    def evaluate(self, env: Mapping):  # type: ignore[override]
        """Return the stored value."""
        if self._value is None:
            raise ValueError("No value provided.")
        # end if
        return self._value
    # end def evaluate

    def _graph_params(self):
        """Expose metadata."""
        return {"dummy": True}
    # end def _graph_params
# end class DummyExpr


def test_math_expr_graph_structure():
    """Ensure graph traversal captures nodes and params."""
    leaf_value = Value([1], Shape((1,)), "int")
    leaf = DummyExpr(leaf_value)
    parent = DummyExpr(leaf_value, (leaf,))
    graph = parent.graph()
    assert graph["root"] in {node["id"] for node in graph["nodes"]}
    assert any(node.get("params") == {"dummy": True} for node in graph["nodes"])
    assert len(graph["edges"]) == 1
# end def test_math_expr_graph_structure

