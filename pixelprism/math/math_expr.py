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
# Copyright (C) 2025 Pixel Prism
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
"""Core node abstraction for PixelPrism's symbolic math system."""

from __future__ import annotations

import weakref
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from .dtype import DType
from .graph_context import GraphContext
from .op import Op
from .source_info import SourceInfo
from .symbolic_dim import SymbolicDim

ShapeDim = int | SymbolicDim
ShapeDesc = Tuple[ShapeDim, ...]

__all__ = ["MathExpr", "ShapeDim", "ShapeDesc"]


class MathExpr:
    """
    Abstract base class for all symbolic math nodes.

    A MathExpr is a *structural* node in a computation DAG. It exposes
    read-only metadata required for evaluation and transformation passes:
    - identity (`id`),
    - operator (`op`) if any,
    - children (inputs),
    - static type information (`dtype`, `shape`), and
    - provenance (`source_info`, `tags`).

    Immutability & identity
    -----------------------
    The *structure* of a node (its operator, children, dtype, shape) is
    immutable after construction. Leaf *values* are not stored here; they
    are bound at evaluation time by the execution engine via a separate
    mapping from Variables to runtime tensors/arrays.

    Equality and hashing default to identity (node-identity semantics),
    making nodes suitable keys in caches within a given context. A separate
    structural equality may be provided by transformation utilities but is
    intentionally not part of the core API to avoid ambiguity.

    Thread-safety & memory
    ----------------------
    Parents are tracked as a weak set to avoid reference cycles. The class
    itself stores no mutable evaluation state, keeping instances safe to
    share across threads once constructed.

    Subclassing
    -----------
    Concrete subclasses MUST set all protected slots at construction and
    adhere to the invariants documented here. Typical subclasses include:
    - Constant (leaf)
    - Variable (leaf)
    - OpNode (non-leaf; wraps an `Op` and its children)

    This class intentionally contains *no* operator overloading. Operator
    syntax (e.g., `a + b`) is provided by thin façade functions or mixins
    outside the core to preserve a minimal, clean kernel.
    """

    __slots__ = (
        "_id",
        "_name",
        "_op",
        "_children",
        "_dtype",
        "_shape",
        "_source_info",
        "_tags",
        "_context",
        "_parents_weak",
        "_meta",
    )

    # region CONSTRUCTOR

    def __init__(
        self,
        *,
        id: int,
        name: Optional[str],
        op: Optional[Op],
        children: Tuple["MathExpr", ...],
        dtype: DType,
        shape: ShapeDesc,
        source_info: Optional[SourceInfo] = None,
        tags: Optional[FrozenSet[str]] = None,
        context: Optional[GraphContext] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Construct a MathExpr (used by subclasses/factories).

        Parameters
        ----------
        id : int
            Unique identifier within the owning context.
        name : str | None
            Optional user-friendly name for debugging/printing.
        op : Op | None
            Operator descriptor for non-leaf nodes; ``None`` for leaves.
        children : tuple[MathExpr, ...]
            Input nodes (empty for leaves).
        dtype : DType
            Element dtype.
        shape : ShapeDesc
            Symbolic shape tuple (ints or SymbolicDim).
        source_info : SourceInfo | None, optional
            Optional provenance data.
        tags : FrozenSet[str] | None, optional
            Optional immutable set of tags (e.g., for passes).
        context : GraphContext | None, optional
            Owning graph context if any.
        meta : Mapping[str, Any] | None, optional
            Arbitrary metadata map (shallow-copied).
        """
        self._id = id
        self._name = name
        self._op = op
        self._children = children
        self._dtype = dtype
        self._shape = shape
        self._source_info = source_info
        self._tags = tags if tags is not None else frozenset()
        self._context = context
        self._parents_weak: "weakref.WeakSet[MathExpr]" = weakref.WeakSet()
        self._meta = dict(meta) if meta is not None else {}
    # end __init__

    # endregion CONSTRUCTOR

    # region PROPERTIES

    @property
    def id(self) -> int:
        """
        Unique identifier (context-local).
        """
        return self._id

    @property
    def name(self) -> Optional[str]:
        """
        Optional user-friendly name.
        """
        return self._name

    @property
    def op(self) -> Optional[Op]:
        """
        Operator descriptor (None for leaves).
        """
        return self._op

    @property
    def children(self) -> Tuple["MathExpr", ...]:
        """
        Child nodes (empty for leaves).
        """
        return self._children

    @property
    def dtype(self) -> DType:
        """
        Element dtype for this expression.
        """
        return self._dtype

    @property
    def shape(self) -> ShapeDesc:
        """
        Symbolic shape of this expression.
        """
        return self._shape

    @property
    def source_info(self) -> Optional[SourceInfo]:
        """
        Source provenance if available.
        """
        return self._source_info

    @property
    def tags(self) -> FrozenSet[str]:
        """
        Immutable tag set.
        """
        return self._tags

    @property
    def context(self) -> Optional[GraphContext]:
        """
        Owning graph context if any.
        """
        return self._context

    @property
    def parents(self) -> FrozenSet["MathExpr"]:
        """
        Weak parent set (best-effort).
        """
        return frozenset(self._parents_weak)

    @property
    def meta(self) -> Mapping[str, Any]:
        """Free-form, read-only metadata map (shallow-copied on construction)."""
        return self._meta

    # endregion PROPERTIES

    # region API

    def is_leaf(self) -> bool:
        """
        True iff this node has no children (Constant/Variable).
        """
        return len(self._children) == 0
    # end is_leaf

    def is_opnode(self) -> bool:
        """
        True iff this node wraps an operator (i.e., `op is not None`).
        """
        return self._op is not None
    # end is_opnode

    def arity(self) -> int:
        """
        Declared arity (0 for leaves).
        """
        return len(self._children)
    # end arity

    def _register_as_parent_of(self, *children: "MathExpr") -> None:
        """
        (Internal) record weak parent links for inverse navigation.

        Subclasses/factories call this exactly once at construction time to
        attach back-references in children. This method is intentionally
        protected; user code should not manipulate graph structure.
        """
        for ch in children:
            try:
                ch._parents_weak.add(self)  # type: ignore[attr-defined]
            except Exception:
                # Parent tracking is best-effort; failure must not break core.
                pass
            # end try
        # end for
    # end _register_as_parent_of

    # endregion CONVENIENCE_API

    # region OVERRIDE

    def __repr__(self) -> str:  # pragma: no cover - formatting only
        op_name = self._op.name if self._op is not None else "Leaf"
        shape_str = "(" + ", ".join(
            d.name if isinstance(d, SymbolicDim) else str(d) for d in self._shape
        ) + ")"
        return f"<{self.__class__.__name__} #{self._id} {op_name} {self._dtype.value} {shape_str} c:{len(self._children)}>"
    # end __repr__

    def __hash__(self) -> int:
        # Identity hashing suitable for dict/set membership within a context.
        return hash((self._context, self._id))
    # end __hash__

    def __eq__(self, other: object) -> bool:
        # Identity semantics: nodes are equal iff they are the same instance
        # or share the same (context, id). Structural equality lives elsewhere.
        if self is other:
            return True
        # end if

        if not isinstance(other, MathExpr):
            return False
        # end if

        return (self._context is other._context) and (self._id == other._id)
    # end __eq__

    # endregion OVERRIDE

# end class MathExpr

