"""
Backend-agnostic graph extraction for MathExpr objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from pixelprism.math import MathExpr


@dataclass(frozen=True)
class Node:
    """Graph node descriptor."""

    id: str
    label: str
    meta: Mapping[str, str]


@dataclass(frozen=True)
class Graph:
    """Minimal directed graph representation."""
    nodes: Mapping[str, Node]
    edges: Tuple[Tuple[str, str], ...]
    roots: Tuple[str, ...]
# end class Graph


def extract(expr: MathExpr) -> Graph:
    """
    Extract a backend-independent graph from a MathExpr.
    """
    visited: Dict[MathExpr, str] = {}
    nodes: Dict[str, Node] = {}
    edges: list[Tuple[str, str]] = []

    def _label(node: MathExpr) -> str:
        if node.name:
            return node.name
        # end if
        return node.__class__.__name__
    # end def _label

    def _meta(node: MathExpr) -> Dict[str, str]:
        meta: Dict[str, str] = {
            "type": node.__class__.__name__,
            "arity": str(node.arity),
            "is_leaf": node.is_leaf(),
            "is_node": node.is_node(),
            "mutable": node.mutable if hasattr(node, "mutable") else None,
        }
        if node.name:
            meta["name"] = node.name
        # end if
        if getattr(node, "dtype", None) is not None:
            meta["dtype"] = str(node.dtype)
        # end if
        if getattr(node, "shape", None) is not None:
            meta["shape"] = str(node.shape)
        # end if
        if node.op is not None:
            meta["op"] = getattr(node.op, "name", repr(node.op))
        # end if
        return meta
    # end def _meta

    def _visit(node: MathExpr) -> str:
        if node in visited:
            return visited[node]
        # end if
        node_id = f"n{len(visited)}"
        visited[node] = node_id
        nodes[node_id] = Node(id=node_id, label=_label(node), meta=_meta(node))
        for child in node.children:
            child_id = _visit(child)
            edges.append((node_id, child_id))
        # end for
        return node_id
    # end def _visit

    root_id = _visit(expr)

    return Graph(
        nodes=dict(nodes),
        edges=tuple(edges),
        roots=(root_id,)
    )
# end def extract
