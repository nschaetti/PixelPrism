"""
S-expression renderer for MathExpr graphs.
"""

from __future__ import annotations

from typing import Dict, List

from .extract import Graph


def _adjacency(graph: Graph) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {node_id: [] for node_id in graph.nodes}
    for parent, child in graph.edges:
        adj[parent].append(child)
    for children in adj.values():
        children.sort()
    return adj


def render(graph: Graph) -> str:
    """
    Render the graph as an S-expression string.
    """
    adj = _adjacency(graph)
    rendered: set[str] = set()

    def _sexpr(node_id: str) -> str:
        node = graph.nodes[node_id]
        if node_id in rendered or not adj[node_id]:
            return node.label
        rendered.add(node_id)
        children = " ".join(_sexpr(child) for child in adj[node_id])
        return f"({node.label} {children})" if children else f"({node.label})"
    # end def _sexpr

    return " ".join(_sexpr(root) for root in graph.roots)
