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
"""Graph-theory operators on adjacency matrices."""

from __future__ import annotations

from abc import ABC
from typing import Optional, Sequence

import numpy as np

from ..dtype import DType, to_numpy
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import Tensor
from .base import Operands, OperatorBase, ParametricOperator, operator_registry


__all__ = [
    "GraphOperator",
    "Degree",
    "InDegree",
    "OutDegree",
    "Laplacian",
    "IsCyclic",
    "TopologicalSort",
]


class GraphOperator(OperatorBase, ParametricOperator, ABC):
    """Base class for adjacency-based graph operators."""

    ARITY = 1
    IS_VARIADIC = False

    def __init__(self, directed: Optional[bool] = None, **kwargs):
        super().__init__(directed=directed, **kwargs)
        self._directed = directed
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(cls, directed: Optional[bool] = None, **kwargs) -> bool:
        return directed is None or isinstance(directed, bool)
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        if len(operands) != 1:
            return False
        # end if
        shape = operands[0].shape
        if shape.rank != 2:
            return False
        # end if
        return shape[0] == shape[1]
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.R
    # end def infer_dtype

    def __str__(self) -> str:
        return f"{self.NAME}(directed={self._directed})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

    def _adjacency(self, operands: Operands) -> np.ndarray:
        if not self.check_shapes(operands):
            raise ValueError(f"{self.NAME} expects one square adjacency matrix operand")
        # end if
        return np.asarray(operands[0].eval().value)
    # end def _adjacency

    def _resolve_directed(self, matrix: np.ndarray) -> bool:
        if self._directed is not None:
            return self._directed
        # end if
        return not np.array_equal(matrix, matrix.T)
    # end def _resolve_directed

# end class GraphOperator


class Degree(GraphOperator):
    """Node degree vector from adjacency matrix."""

    NAME = "degree"

    def __init__(self, directed: Optional[bool] = None, mode: str = "total"):
        if mode not in {"in", "out", "total"}:
            raise ValueError("degree mode must be one of {'in', 'out', 'total'}")
        # end if
        super().__init__(directed=directed, mode=mode)
        self._mode = mode
    # end def __init__

    @classmethod
    def check_parameters(cls, directed: Optional[bool] = None, mode: str = "total", **kwargs) -> bool:
        return super().check_parameters(directed=directed) and mode in {"in", "out", "total"}
    # end def check_parameters

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(operands[0].shape[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.Z
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        adj = self._adjacency(operands)
        directed = self._resolve_directed(adj)

        if directed:
            out_deg = np.sum(adj != 0, axis=1)
            in_deg = np.sum(adj != 0, axis=0)
            if self._mode == "in":
                degree = in_deg
            elif self._mode == "out":
                degree = out_deg
            else:
                degree = in_deg + out_deg
            # end if
        else:
            degree = np.sum(adj != 0, axis=1)
        # end if

        return Tensor(data=np.asarray(degree, dtype=np.int64), dtype=DType.Z)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Degree does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(directed={self._directed}, mode={self._mode})"
    # end def __str__

# end class Degree


class InDegree(Degree):
    """Incoming degree vector for directed adjacency matrix."""

    NAME = "in_degree"

    def __init__(self, directed: Optional[bool] = None):
        super().__init__(directed=directed, mode="in")
    # end def __init__

# end class InDegree


class OutDegree(Degree):
    """Outgoing degree vector for directed adjacency matrix."""

    NAME = "out_degree"

    def __init__(self, directed: Optional[bool] = None):
        super().__init__(directed=directed, mode="out")
    # end def __init__

# end class OutDegree


class Laplacian(GraphOperator):
    """Graph Laplacian (unnormalized or symmetric normalized)."""

    NAME = "laplacian"

    def __init__(self, directed: Optional[bool] = None, normalized: bool = False, dtype: DType = DType.R):
        super().__init__(directed=directed, normalized=normalized, dtype=dtype)
        self._normalized = bool(normalized)
        self._dtype = dtype
    # end def __init__

    @classmethod
    def check_parameters(
            cls,
            directed: Optional[bool] = None,
            normalized: bool = False,
            dtype: DType = DType.R,
            **kwargs
    ) -> bool:
        return super().check_parameters(directed=directed) and isinstance(normalized, bool)
    # end def check_parameters

    def infer_shape(self, operands: Operands) -> Shape:
        n = operands[0].shape[0]
        return Shape.matrix(n, n)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        adj = self._adjacency(operands)
        directed = self._resolve_directed(adj)
        work_adj = np.asarray(adj != 0, dtype=np.float64)

        if directed:
            degree = np.sum(work_adj, axis=1)
        else:
            degree = np.sum(work_adj, axis=1)
            work_adj = np.maximum(work_adj, work_adj.T)
        # end if

        d_mat = np.diag(degree)
        lap = d_mat - work_adj

        if self._normalized:
            inv_sqrt = np.zeros_like(degree, dtype=np.float64)
            non_zero = degree > 0
            inv_sqrt[non_zero] = 1.0 / np.sqrt(degree[non_zero])
            d_inv_sqrt = np.diag(inv_sqrt)
            lap = d_inv_sqrt @ lap @ d_inv_sqrt
        # end if

        return Tensor(data=np.asarray(lap, dtype=to_numpy(self._dtype)), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Laplacian does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(directed={self._directed}, normalized={self._normalized}, dtype={self._dtype.name})"
    # end def __str__

# end class Laplacian


class IsCyclic(GraphOperator):
    """Return whether the graph represented by adjacency has a cycle."""

    NAME = "is_cyclic"

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.scalar()
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.B
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        adj = np.asarray(self._adjacency(operands) != 0, dtype=np.int8)
        directed = self._resolve_directed(adj)
        cyclic = _has_cycle_directed(adj) if directed else _has_cycle_undirected(adj)
        return Tensor(data=np.asarray(cyclic, dtype=np.bool_), dtype=DType.B)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("IsCyclic does not support backward propagation.")
    # end def _backward

# end class IsCyclic


class TopologicalSort(GraphOperator):
    """Topological ordering from a directed acyclic adjacency matrix."""

    NAME = "topological_sort"

    def __init__(self, directed: Optional[bool] = True):
        super().__init__(directed=directed)
    # end def __init__

    def infer_shape(self, operands: Operands) -> Shape:
        return Shape.vector(operands[0].shape[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.Z
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        adj = np.asarray(self._adjacency(operands) != 0, dtype=np.int8)
        directed = self._resolve_directed(adj)
        if not directed:
            raise ValueError("topological_sort requires a directed adjacency matrix")
        # end if
        order = _topological_sort(adj)
        return Tensor(data=np.asarray(order, dtype=np.int64), dtype=DType.Z)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("TopologicalSort does not support backward propagation.")
    # end def _backward

# end class TopologicalSort


def _has_cycle_directed(adj: np.ndarray) -> bool:
    n = int(adj.shape[0])
    state = np.zeros(n, dtype=np.int8)

    def dfs(node: int) -> bool:
        state[node] = 1
        for neigh in np.where(adj[node] != 0)[0]:
            neigh_i = int(neigh)
            if state[neigh_i] == 1:
                return True
            # end if
            if state[neigh_i] == 0 and dfs(neigh_i):
                return True
            # end if
        # end for
        state[node] = 2
        return False
    # end def dfs

    for node in range(n):
        if state[node] == 0 and dfs(node):
            return True
        # end if
    # end for
    return False
# end def _has_cycle_directed


def _has_cycle_undirected(adj: np.ndarray) -> bool:
    n = int(adj.shape[0])
    visited = np.zeros(n, dtype=np.int8)

    def dfs(node: int, parent: int) -> bool:
        visited[node] = 1
        for neigh in np.where(adj[node] != 0)[0]:
            neigh_i = int(neigh)
            if visited[neigh_i] == 0:
                if dfs(neigh_i, node):
                    return True
                # end if
            elif neigh_i != parent:
                return True
            # end if
        # end for
        return False
    # end def dfs

    for node in range(n):
        if visited[node] == 0 and dfs(node, -1):
            return True
        # end if
    # end for
    return False
# end def _has_cycle_undirected


def _topological_sort(adj: np.ndarray) -> np.ndarray:
    n = int(adj.shape[0])
    indeg = np.sum(adj != 0, axis=0).astype(np.int64)
    queue = [int(i) for i in np.where(indeg == 0)[0].tolist()]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neigh in np.where(adj[node] != 0)[0]:
            neigh_i = int(neigh)
            indeg[neigh_i] -= 1
            if indeg[neigh_i] == 0:
                queue.append(neigh_i)
            # end if
        # end for
    # end while

    if len(order) != n:
        raise ValueError("Graph contains a cycle; topological ordering is undefined")
    # end if

    return np.asarray(order, dtype=np.int64)
# end def _topological_sort


operator_registry.register(Degree)
operator_registry.register(InDegree)
operator_registry.register(OutDegree)
operator_registry.register(Laplacian)
operator_registry.register(IsCyclic)
operator_registry.register(TopologicalSort)
