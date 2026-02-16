# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2026 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""Graph primitives and adapters for symbolic math expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dtype import DType
from .math_leaves import const
from .random import random_const_name
from .typing import MathExpr


Edge = Tuple[int, int]


__all__ = [
    "Graph",
    "ExpressionGraph",
    "graph",
    "digraph",
    "graph_from_edges",
    "graph_from_adjacency",
    "expression_to_graph",
    "expression_to_adjacency",
    "graph_to_dot",
]


@dataclass(frozen=True)
class Graph:
    """Simple graph representation (no multi-edges)."""

    n_nodes: int
    edges: Tuple[Edge, ...]
    directed: bool = True
    allow_cycles: bool = True
    labels: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.n_nodes < 0:
            raise ValueError("n_nodes must be >= 0")
        # end if

        normalized: List[Edge] = []
        seen: set[Edge] = set()
        for u, v in self.edges:
            u_i = int(u)
            v_i = int(v)
            if u_i < 0 or v_i < 0 or u_i >= self.n_nodes or v_i >= self.n_nodes:
                raise ValueError(f"Edge ({u_i}, {v_i}) out of range for n_nodes={self.n_nodes}")
            # end if
            key = (u_i, v_i) if self.directed else (min(u_i, v_i), max(u_i, v_i))
            if key not in seen:
                seen.add(key)
                normalized.append(key)
            # end if
        # end for
        normalized.sort()
        object.__setattr__(self, "edges", tuple(normalized))

        if self.labels is not None and len(self.labels) != self.n_nodes:
            raise ValueError("labels length must match n_nodes")
        # end if

        if not self.allow_cycles and self.is_cyclic():
            raise ValueError("Graph contains a cycle while allow_cycles=False")
        # end if
    # end def __post_init__

    @property
    def n_edges(self) -> int:
        return len(self.edges)
    # end def n_edges

    def has_edge(self, u: int, v: int) -> bool:
        if self.directed:
            return (u, v) in self.edges
        # end if
        return (min(u, v), max(u, v)) in self.edges
    # end def has_edge

    def neighbors(self, node: int) -> Tuple[int, ...]:
        node_i = int(node)
        if node_i < 0 or node_i >= self.n_nodes:
            raise ValueError(f"node {node_i} out of range")
        # end if
        neigh: List[int] = []
        for u, v in self.edges:
            if self.directed:
                if u == node_i:
                    neigh.append(v)
                # end if
            else:
                if u == node_i:
                    neigh.append(v)
                elif v == node_i and u != v:
                    neigh.append(u)
                # end if
            # end if
        # end for
        return tuple(sorted(set(neigh)))
    # end def neighbors

    def out_degree(self) -> np.ndarray:
        deg = np.zeros(self.n_nodes, dtype=np.int64)
        for u, v in self.edges:
            deg[u] += 1
            if not self.directed and u != v:
                deg[v] += 1
            # end if
        # end for
        return deg
    # end def out_degree

    def in_degree(self) -> np.ndarray:
        if not self.directed:
            return self.out_degree()
        # end if
        deg = np.zeros(self.n_nodes, dtype=np.int64)
        for _, v in self.edges:
            deg[v] += 1
        # end for
        return deg
    # end def in_degree

    def degree(self) -> np.ndarray:
        return self.out_degree() if not self.directed else self.in_degree() + self.out_degree()
    # end def degree

    def adjacency(self, dtype=np.int64) -> np.ndarray:
        matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=dtype)
        for u, v in self.edges:
            matrix[u, v] = 1
            if not self.directed:
                matrix[v, u] = 1
            # end if
        # end for
        return matrix
    # end def adjacency

    def to_adjacency_expr(self, name: Optional[str] = None, dtype: DType = DType.Z):
        node_name = name or random_const_name("graph-adj-")
        return const(name=node_name, data=self.adjacency(), dtype=dtype)
    # end def to_adjacency_expr

    def is_cyclic(self) -> bool:
        return self._is_cyclic_directed() if self.directed else self._is_cyclic_undirected()
    # end def is_cyclic

    def topological_sort(self) -> Tuple[int, ...]:
        if not self.directed:
            raise ValueError("topological_sort is only defined for directed graphs")
        # end if
        indeg = np.zeros(self.n_nodes, dtype=np.int64)
        outgoing: List[List[int]] = [[] for _ in range(self.n_nodes)]
        for u, v in self.edges:
            outgoing[u].append(v)
            indeg[v] += 1
        # end for
        queue = [i for i in range(self.n_nodes) if indeg[i] == 0]
        order: List[int] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neigh in outgoing[node]:
                indeg[neigh] -= 1
                if indeg[neigh] == 0:
                    queue.append(neigh)
                # end if
            # end for
        # end while
        if len(order) != self.n_nodes:
            raise ValueError("Graph contains a cycle; topological ordering is undefined")
        # end if
        return tuple(order)
    # end def topological_sort

    def _is_cyclic_directed(self) -> bool:
        state = np.zeros(self.n_nodes, dtype=np.int8)
        outgoing: List[List[int]] = [[] for _ in range(self.n_nodes)]
        for u, v in self.edges:
            outgoing[u].append(v)
        # end for

        def dfs(node: int) -> bool:
            state[node] = 1
            for neigh in outgoing[node]:
                if state[neigh] == 1:
                    return True
                # end if
                if state[neigh] == 0 and dfs(neigh):
                    return True
                # end if
            # end for
            state[node] = 2
            return False
        # end def dfs

        for node in range(self.n_nodes):
            if state[node] == 0 and dfs(node):
                return True
            # end if
        # end for
        return False
    # end def _is_cyclic_directed

    def _is_cyclic_undirected(self) -> bool:
        visited = np.zeros(self.n_nodes, dtype=np.int8)
        adj: List[List[int]] = [[] for _ in range(self.n_nodes)]
        for u, v in self.edges:
            adj[u].append(v)
            if u != v:
                adj[v].append(u)
            # end if
        # end for

        def dfs(node: int, parent: int) -> bool:
            visited[node] = 1
            for neigh in adj[node]:
                if visited[neigh] == 0:
                    if dfs(neigh, node):
                        return True
                    # end if
                elif neigh != parent:
                    return True
                # end if
            # end for
            return False
        # end def dfs

        for node in range(self.n_nodes):
            if visited[node] == 0 and dfs(node, -1):
                return True
            # end if
        # end for
        return False
    # end def _is_cyclic_undirected

# end class Graph


@dataclass(frozen=True)
class ExpressionGraph:
    """Graph view built from a symbolic expression."""

    graph: Graph
    index_to_expr: Mapping[int, MathExpr]
    expr_id_to_index: Mapping[int, int]
    root_index: int

# end class ExpressionGraph


def graph(n_nodes: int, edges: Iterable[Edge], allow_cycles: bool = True) -> Graph:
    return Graph(n_nodes=n_nodes, edges=tuple(edges), directed=False, allow_cycles=allow_cycles)
# end def graph


def digraph(n_nodes: int, edges: Iterable[Edge], allow_cycles: bool = True) -> Graph:
    return Graph(n_nodes=n_nodes, edges=tuple(edges), directed=True, allow_cycles=allow_cycles)
# end def digraph


def graph_from_edges(
        n_nodes: int,
        edges: Iterable[Edge],
        *,
        directed: bool = True,
        allow_cycles: bool = True,
        labels: Optional[Sequence[str]] = None,
) -> Graph:
    return Graph(
        n_nodes=n_nodes,
        edges=tuple(edges),
        directed=directed,
        allow_cycles=allow_cycles,
        labels=tuple(labels) if labels is not None else None,
    )
# end def graph_from_edges


def graph_from_adjacency(
        adjacency,
        *,
        directed: Optional[bool] = None,
        allow_cycles: bool = True,
        atol: float = 1e-8,
) -> Graph:
    if isinstance(adjacency, np.ndarray):
        matrix = np.asarray(adjacency)
    elif hasattr(adjacency, "eval"):
        matrix = np.asarray(adjacency.eval().value)
    elif hasattr(adjacency, "value"):
        matrix = np.asarray(adjacency.value)
    else:
        matrix = np.asarray(adjacency)
    # end if

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square rank-2, got shape={matrix.shape}")
    # end if

    n_nodes = int(matrix.shape[0])
    if directed is None:
        directed = not np.allclose(matrix, matrix.T, atol=atol)
    # end if
    if not directed and not np.allclose(matrix, matrix.T, atol=atol):
        raise ValueError("Undirected adjacency matrix must be symmetric")
    # end if

    edges: List[Edge] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if matrix[i, j] == 0:
                continue
            # end if
            if directed:
                edges.append((i, j))
            elif i <= j:
                edges.append((i, j))
            # end if
        # end for
    # end for
    return Graph(n_nodes=n_nodes, edges=tuple(edges), directed=directed, allow_cycles=allow_cycles)
# end def graph_from_adjacency


def expression_to_graph(expr: MathExpr, edge_direction: str = "child_to_parent") -> ExpressionGraph:
    if edge_direction not in {"child_to_parent", "parent_to_child"}:
        raise ValueError("edge_direction must be 'child_to_parent' or 'parent_to_child'")
    # end if

    visited: Dict[int, MathExpr] = {}
    raw_edges: List[Tuple[int, int]] = []
    stack: List[MathExpr] = [expr]

    while stack:
        current = stack.pop()
        current_id = int(current.identifier)
        if current_id not in visited:
            visited[current_id] = current
        # end if
        children = getattr(current, "children", ())
        for child in children:
            child_id = int(child.identifier)
            if edge_direction == "child_to_parent":
                raw_edges.append((child_id, current_id))
            else:
                raw_edges.append((current_id, child_id))
            # end if
            if child_id not in visited:
                stack.append(child)
            # end if
        # end for
    # end while

    ordered_ids = sorted(visited.keys())
    expr_id_to_index = {expr_id: idx for idx, expr_id in enumerate(ordered_ids)}
    index_to_expr = {idx: visited[expr_id] for expr_id, idx in expr_id_to_index.items()}
    labels = tuple(_expression_label(index_to_expr[idx]) for idx in range(len(index_to_expr)))

    edges = tuple((expr_id_to_index[u], expr_id_to_index[v]) for u, v in raw_edges)
    g = Graph(
        n_nodes=len(index_to_expr),
        edges=edges,
        directed=True,
        allow_cycles=True,
        labels=labels,
    )
    root_index = expr_id_to_index[int(expr.identifier)]
    return ExpressionGraph(graph=g, index_to_expr=index_to_expr, expr_id_to_index=expr_id_to_index, root_index=root_index)
# end def expression_to_graph


def expression_to_adjacency(expr: MathExpr, edge_direction: str = "child_to_parent", dtype: DType = DType.Z):
    view = expression_to_graph(expr, edge_direction=edge_direction)
    return view.graph.to_adjacency_expr(dtype=dtype)
# end def expression_to_adjacency


def graph_to_dot(g: Graph, graph_name: str = "G") -> str:
    connector = "->" if g.directed else "--"
    opening = "digraph" if g.directed else "graph"
    lines = [f"{opening} {graph_name} {{"]

    for node in range(g.n_nodes):
        label = g.labels[node] if g.labels is not None else str(node)
        lines.append(f"  {node} [label=\"{label}\"]; ")
    # end for

    for u, v in g.edges:
        lines.append(f"  {u} {connector} {v};")
    # end for
    lines.append("}")
    return "\n".join(lines)
# end def graph_to_dot


def _expression_label(expr: MathExpr) -> str:
    op = getattr(expr, "op", None)
    if op is not None:
        return f"{op.name}:{expr.name}" if expr.name is not None else op.name
    # end if
    expr_name = expr.name if expr.name is not None else expr.__class__.__name__
    return expr_name
# end def _expression_label
