"""
Console tree backend for MathExpr graphs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .extract import Graph


def _children(graph: Graph) -> Dict[str, List[str]]:
    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in graph.nodes}
    for parent, child in graph.edges:
        adjacency[parent].append(child)
    for child_list in adjacency.values():
        child_list.sort()
    return adjacency


def render(graph: Graph, indent: str = "  ") -> str:
    """
    Render a graph as an ASCII tree.
    """
    adjacency = _children(graph)
    lines: List[str] = []
    seen: set[str] = set()

    def _render(node_id: str, prefix: str) -> None:
        node = graph.nodes[node_id]
        marker = ""
        if node_id in seen:
            marker = " [ref]"
        lines.append(f"{prefix}{node.label}{marker}")
        if marker:
            return
        seen.add(node_id)
        for child_id in adjacency[node_id]:
            _render(child_id, prefix + indent)
    # end def _render

    for root in graph.roots:
        _render(root, "")
    return "\n".join(lines)
