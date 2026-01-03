"""
Topological textual renderer for MathExpr graphs.
"""

from __future__ import annotations

import heapq
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
    Render a graph as a topologically ordered textual listing.
    """
    adj = _adjacency(graph)
    indegree: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    for _, child in graph.edges:
        indegree[child] += 1

    queue: List[str] = []
    for node_id, deg in indegree.items():
        if deg == 0:
            heapq.heappush(queue, node_id)

    order: List[str] = []
    while queue:
        node_id = heapq.heappop(queue)
        order.append(node_id)
        for child in adj[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, child)

    lines: List[str] = []
    for node_id in order:
        node = graph.nodes[node_id]
        meta = ", ".join(f"{k}={v}" for k, v in sorted(node.meta.items()))
        children = ", ".join(adj[node_id]) or "-"
        lines.append(f"{node_id}: {node.label} -> [{children}] {{{meta}}}")
    return "\n".join(lines)
