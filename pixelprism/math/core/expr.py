#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

#
# This file contains the Scalar class, which is used to represent a scalar value.
#

"""pixelprism.math.core.expr

Core node abstraction for PixelPrism's symbolic math system.

This module defines the **MathExpr** base class: an immutable
(compositionally immutable) node that lives in a directed acyclic
computation graph (DAG). A MathExpr carries *structure* (children,
associated operator), *static typing* (dtype/shape), and *provenance*
(source information, tags). It contains **no runtime values** and no
backend-specific logic. Evaluation and differentiation are handled by
separate subsystems (engine, autodiff) using the structural metadata
exposed by MathExpr.

Design goals
------------
- Clear separation of concerns: node (MathExpr) vs. operator semantics (Op).
- Immutability of graph structure after construction for thread-safety
  and reliable caching/interning.
- Early type/shape availability (or symbolic placeholders) for good UX.
- Minimal but strong introspection API (children, op, dtype, shape, ids).
- Lightweight; no heavy dependencies.

Notes
-----
- Variables/Constants are leaves (specializations of MathExpr).
- Operator applications create OpNodes (also a specialization of MathExpr),
  where `op` refers to a shared, stateless Op descriptor.
- `GraphContext` owns node-id assignment, optional interning, and cache
  invalidation policies. MathExpr keeps an opaque pointer to its context
  for identity stability but does not depend on its implementation.

"""

# Imports
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Tuple, FrozenSet, Protocol
import weakref

from pixelprism.math.ops import Op


# ---------------------------------------------------------------------------
# Lightweight shared types (minimal stubs to keep typing precise)
# ---------------------------------------------------------------------------

class DType(Enum):
    """
    Scalar/element dtypes supported by PixelPrism's math system.

    Extend as needed; backends may map these to their native types.
    """
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
# end DType


@dataclass(frozen=True)
class SymbolicDim:
    """
    A symbolic dimension (e.g., N, M) usable in shapes.

    Example
    -------
    >>> N = SymbolicDim("N")
    >>> N.name
    'N'
    """
    name: str
# end SymbolicDim


ShapeDim = int | SymbolicDim
ShapeDesc = Tuple[ShapeDim, ...]  # e.g., (), (N,), (N, M), (3, 224, 224)


@dataclass(frozen=True)
class SourceInfo:
    """
    Provenance for a node.

    Attributes
    ----------
    file : str | None
        Path-like location. May be None for synthetic nodes.
    line : int | None
        1-based line number if available.
    module : str | None
        Importable module path if available.
    """
    file: Optional[str] = None
    line: Optional[int] = None
    module: Optional[str] = None
# end SourceInfo


class GraphContext(Protocol):
    """
    Opaque owner of node identities and optional interning.

    MathExpr stores a pointer to its `context` for identity stability and
    optional services (e.g., weak parent tracking), but does not call into
    it from core logic.
    """

    def uid(self) -> int: ...  # monotonic or otherwise stable within context
# end class GraphContext


# ---------------------------------------------------------------------------
# Core abstraction: MathExpr
# ---------------------------------------------------------------------------

class MathExpr(ABC):
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
        context: GraphContext,
        op: Optional[Op],
        children: Tuple["MathExpr", ...],
        dtype: DType,
        shape: ShapeDesc,
        source_info: Optional[SourceInfo] = None,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # Identity & ownership
        self._id: int = int(id)
        self._context: GraphContext = context

        # Structure
        self._op: Optional[Op] = op
        self._children: Tuple[MathExpr, ...] = tuple(children)
        self._dtype: DType = dtype
        self._shape: ShapeDesc = tuple(shape)  # normalize to tuple

        # Provenance & annotations
        self._source_info: SourceInfo = source_info or SourceInfo()
        self._name: Optional[str] = name
        self._tags: FrozenSet[str] = frozenset(tags or ())
        self._meta: Mapping[str, Any] = dict(meta or {})

        # Weak parent tracking for inverse navigation (optional)
        self._parents_weak: "weakref.WeakSet[MathExpr]" = weakref.WeakSet()

        # Basic invariants (lightweight; deep validation is done by Op creation)
        if any(ch is None for ch in self._children):  # pragma: no cover
            raise ValueError("children must not contain None")
        # end if

        if isinstance(self._op, Op) and len(self._children) == 0:
            # Non-leaf must have inputs (arity validation is enforced elsewhere)
            pass
        # end if
    # end __init__

    # endregion CONSTRUCTOR

    # region PROPERTIES

    @property
    def id(self) -> int:
        """
        Opaque, context-scoped node identifier (stable within context).
        """
        return self._id

    @property
    def name(self) -> Optional[str]:
        """
        Human-friendly alias for logs/visualization (optional).
        """
        return self._name

    @property
    def op(self) -> Optional[Op]:
        """Operator descriptor if this node is an operator application.

        Returns None for leaf nodes (Variable/Constant).
        """
        return self._op

    @property
    def children(self) -> Tuple["MathExpr", ...]:
        """Direct input nodes (empty tuple for leaves)."""
        return self._children

    @property
    def parents(self) -> Iterable["MathExpr"]:
        """Best-effort iterable of parent nodes using weak references."""
        return iter(self._parents_weak)

    @property
    def dtype(self) -> DType:
        """Element dtype determined (or inferred) at build time."""
        return self._dtype

    @property
    def shape(self) -> ShapeDesc:
        """Static shape tuple (may include symbolic dimensions)."""
        return self._shape

    @property
    def source_info(self) -> SourceInfo:
        """Provenance descriptor (file/line/module), possibly empty."""
        return self._source_info

    @property
    def tags(self) -> FrozenSet[str]:
        """Arbitrary user or system annotations (immutable view)."""
        return self._tags

    @property
    def context(self) -> GraphContext:
        """Owning context; identity and interning scope live here."""
        return self._context

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
