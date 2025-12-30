"""Abstract base class for symbolic math expressions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING

from .shape import Shape
from .value import Value

if TYPE_CHECKING:
    from .var import Var
# end if

__all__ = ["MathExpr"]


class MathExpr(ABC):
    """Abstract base class for symbolic math expressions."""

    def __init__(self, shape: Shape, dtype: Any, children: Sequence["MathExpr"] | None = None):
        """Initialize a MathExpr.

        Args:
            shape: Symbolic output shape.
            dtype: Runtime dtype metadata.
            children: Child expressions.
        """
        self._shape = shape
        self._dtype = dtype
        self._children: Tuple[MathExpr, ...] = tuple(children) if children else tuple()
    # end def __init__

    @property
    def shape(self) -> Shape:
        """Return the expression shape.

        Returns:
            Shape: Expression output shape.
        """
        return self._shape
    # end def shape

    @property
    def dtype(self) -> Any:
        """Return the expression dtype.

        Returns:
            Any: Runtime dtype metadata.
        """
        return self._dtype
    # end def dtype

    @property
    def children(self) -> Tuple["MathExpr", ...]:
        """Return the child expressions.

        Returns:
            Tuple[MathExpr, ...]: Child expression nodes.
        """
        return self._children
    # end def children

    @abstractmethod
    def evaluate(self, env: Mapping["Var", Value]) -> Value:
        """Evaluate the expression in the provided environment.

        Args:
            env: Mapping from Var to runtime Value.

        Returns:
            Value: Resulting runtime Value.
        """
    # end def evaluate

    def graph(self) -> Dict[str, Any]:
        """Return a serializable computation graph for the expression tree.

        Returns:
            Dict[str, Any]: Graph description including nodes and edges.
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        visited: Dict[int, str] = {}

        def visit(expr: MathExpr) -> str:
            """Traverse the expression tree and record graph nodes.

            Args:
                expr: Expression node to process.

            Returns:
                str: Identifier assigned to the node.
            """
            expr_id = id(expr)
            if expr_id in visited:
                return visited[expr_id]
            # end if
            node_id = f"n{len(visited)}"
            visited[expr_id] = node_id
            child_refs: List[Tuple[int, str]] = []
            for idx, child in enumerate(expr.children):
                child_id = visit(child)
                child_refs.append((idx, child_id))
            # end for
            node_info = {
                "id": node_id,
                "type": expr.__class__.__name__,
                "shape": expr.shape.as_tuple(),
                "dtype": expr.dtype,
            }
            params = expr._graph_params()
            if params:
                node_info["params"] = params
            # end if
            nodes.append(node_info)
            for idx, child_id in child_refs:
                edges.append({"parent": node_id, "child": child_id, "index": idx})
            # end for
            return node_id
        # end def visit

        root_id = visit(self)
        return {"root": root_id, "nodes": nodes, "edges": edges}
    # end def graph

    def _graph_params(self) -> Dict[str, Any]:
        """Return additional parameters for graph visualization.

        Returns:
            Dict[str, Any]: Optional metadata to attach to graph nodes.
        """
        return {}
    # end def _graph_params
# end class MathExpr
