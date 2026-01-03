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

"""Core abstractions for PixelPrism's symbolic math system.

This module documents the structural contract implemented by every
``MathExpr`` node as well as the :class:`~pixelprism.math.shape.Shape`
descriptor used to carry symbolic tensor shapes.  Even though the core
implementation currently lives in generated code, this docstring captures
the canonical interface in a single place for reference by contributors,
type-checkers, and the documentation build.

MathExpr Contract
-----------------
``MathExpr`` instances represent nodes in the symbolic computation DAG.
Each node owns immutable metadata:

* ``name`` – optional string identifier used for debugging traces.
* ``children`` – tuple of input nodes (empty for leaves).
* ``dtype`` – :class:`~pixelprism.math.dtype.DType` describing element type.
* ``shape`` – :class:`~pixelprism.math.shape.Shape` describing tensor extent.

Parameters
----------
name : str, optional
    Display label used in debug UIs and logging outputs.
children : tuple[MathExpr, ...]
    Positional dependencies for the node.  Leaves supply an empty tuple.
dtype : DType
    Scalar element type that downstream passes use for validation.
shape : Shape
    Symbolic tensor shape.  See *Shape descriptor* below for the complete
    contract implemented by :class:`~pixelprism.math.shape.Shape`.

Shape descriptor
----------------
The :class:`~pixelprism.math.shape.Shape` class is a thin immutable wrapper
around a tuple of dimensions.  A dimension is either an ``int`` (known size)
or ``None`` (symbolic degree of freedom).  Shape instances offer convenience
properties that are frequently consumed by higher-level code:

``dims``
    Tuple of the stored dimensions.
``rank``
    Number of axes in the tensor (``len(dims)``).
``size``
    Product of all non-symbolic dimensions or ``None`` when any dimension is
    symbolic.
``is_elementwise_compatible(other)``
    Returns ``True`` when ``self`` and ``other`` have the same rank and each
    axis is either equal or symbolic.
``merge_elementwise(other)``
    Produces a new shape where each axis merges the compatible dimensions
    from the two operands, raising :class:`ValueError` when they conflict.

Examples
--------
The snippets below illustrate the expected API for both ``MathExpr`` nodes
and the ``Shape`` helper.

Create a shape, inspect metadata, and merge it with another compatible shape.

>>> from pixelprism.math import Shape
>>> activations = Shape((32, 128))
>>> activations.rank
2
>>> activations.size
4096
>>> merged = activations.merge_elementwise(Shape((32, 128)))
>>> merged.dims
(32, 128)

Use a shape while instantiating a hypothetical ``MathExpr`` leaf node.

>>> from pixelprism.math import DType
>>> weight = MathExpr(  # created by a concrete subclass/factory
...     name="weight",
...     children=(),
...     dtype=DType.FLOAT32,
...     shape=Shape((128, 256)),
... )
>>> weight.dtype
<DType.FLOAT32: 'float32'>
>>> weight.shape.dims
(128, 256)
"""

# Imports
from __future__ import annotations
import weakref
from abc import ABC, abstractmethod
from typing import Any, FrozenSet, Mapping, Optional, Tuple
from .dtype import DType
from .shape import Shape
from .operators import Operator


__all__ = ["MathExpr", "MathLeaf"]


class MathExpr:
    """
    Abstract base class for all symbolic math nodes.

    A MathExpr is a *structural* node in a computation DAG. It exposes
    read-only metadata required for evaluation and transformation passes:
    - identity name (`name`),
    - children (inputs),
    - static type information (`dtype`, `shape`)

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

    Keeping this logic encapsulated means ``MathExpr`` nodes never have to
    duplicate validation code and guarantees that any shape a node advertises
    has already passed the invariants enforced by :mod:`pixelprism.math.shape`.
    """

    __slots__ = (
        "_id",
        "_name",
        "_op",
        "_children",
        "_dtype",
        "_shape",
        "_parents_weak",
    )

    # Global counter
    _next_id = 0

    # region CONSTRUCTOR

    def __init__(
            self,
            *,
            name: Optional[str],
            op: Optional[Operator],
            children: Tuple["MathExpr", ...],
            dtype: DType,
            shape: Shape
    ) -> None:
        """
        Construct a MathExpr (used by subclasses/factories).

        Parameters
        ----------
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
        """
        self._id: int = MathExpr._next_id
        self._name: str = name
        self._op: Operator = op
        self._children: Tuple["MathExpr", ...] = children
        self._dtype: DType = dtype
        self._shape: Shape = shape
        self._parents_weak: "weakref.WeakSet[MathExpr]" = weakref.WeakSet()
        self._check_operator()
        MathExpr._next_id += 1
    # end __init__

    # endregion CONSTRUCTOR

    # region PROPERTIES

    @property
    def identifier(self):
        """Unique identifier for this expression."""
        return self._id
    # end def identifier

    @property
    def name(self) -> Optional[str]:
        """
        Optional user-friendly name.
        """
        return self._name
    # end def name

    @property
    def op(self) -> Optional[Any]:
        """
        Operator descriptor for non-leaf nodes; ``None`` for leaves.
        """
        return self._op
    # end op

    @property
    def children(self) -> Tuple["MathExpr", ...]:
        """
        Child nodes (empty for leaves).
        """
        return self._children
    # end def children

    @property
    def dtype(self) -> DType:
        """
        Element dtype for this expression.
        """
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """
        Symbolic shape of this expression.
        """
        return self._shape
    # end def shape

    @property
    def rank(self):
        """Rank of the tensor."""
        return self._shape.rank
    # end def rank

    @property
    def parents(self) -> FrozenSet["MathExpr"]:
        """
        Weak parent set (best-effort).
        """
        return frozenset(self._parents_weak)
    # end def parents

    @property
    def arity(self):
        """Number of children."""
        return len(self._children)
    # end def arity

    # endregion PROPERTIES

    # region PUBLIC

    def eval(self, **kwargs):
        """Evaluate this expression in the current context."""
        values = [c.eval(**kwargs) for c in self._children]
        return self._op.eval(values=values)
    # end def eval

    def is_node(self) -> bool:
        """
        True iff this node is a non-leaf.
        """
        return self._op is not None
    # end def is_node

    def is_leaf(self) -> bool:
        """
        True iff this node has no children (Constant/Variable).
        """
        return len(self._children) == 0
    # end is_leaf

    # endregion PUBLIC

    # region PRIVATE

    def _check_operator(self):
        """Check that the operator is valid for this node type.
        """
        if self._op is not None and self.arity != self._op.arity:
            raise TypeError(
                f"Operator and arity mismatch: "
                f"{self._op.name}({self.arity}) != {self.__class__.__name__}({self._op.arity})"
            )
        # end if
    # end _def_check_operator

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

    # endregion PRIVATE

    # region OPERATORS

    @staticmethod
    def add(operand1: MathExpr, operand2: MathExpr) -> MathExpr:
        """Addition operator
        """
        from .functional.elementwise import add
        return add(operand1, operand2)
    # end def add

    @staticmethod
    def sub(operand1: MathExpr, operand2: MathExpr) -> MathExpr:
        """Substraction operator
        """
        from .functional.elementwise import sub
        return sub(operand1, operand2)
    # end def sub

    @staticmethod
    def mul(operand1: MathExpr, operand2: MathExpr) -> MathExpr:
        """Multiplication operator"""
        from .functional.elementwise import mul
        return mul(operand1, operand2)
    # end def mul

    @staticmethod
    def div(operand1: MathExpr, operand2: MathExpr) -> MathExpr:
        """Division operator"""
        from .functional.elementwise import div
        return div(operand1, operand2)
    # end def div

    @staticmethod
    def neg(operand: MathExpr) -> MathExpr:
        """Negation operator"""
        from .functional.elementwise import neg
        return neg(operand)
    # end def neg

    @staticmethod
    def pow(operand1: MathExpr, operand2: MathExpr) -> MathExpr:
        """Power operator"""
        from .functional.elementwise import pow as elementwise_pow
        return elementwise_pow(operand1, operand2)
    # end def pow

    @staticmethod
    def exp(operand: MathExpr) -> MathExpr:
        """Exponential operator"""
        from .functional.elementwise import exp as elementwise_exp
        return elementwise_exp(operand)
    # end def exp

    @staticmethod
    def log(operand: MathExpr) -> MathExpr:
        """Natural logarithm operator"""
        from .functional.elementwise import log as elementwise_log
        return elementwise_log(operand)
    # end def log

    @staticmethod
    def sqrt(operand: MathExpr) -> MathExpr:
        """Square root operator"""
        from .functional.elementwise import sqrt as elementwise_sqrt
        return elementwise_sqrt(operand)
    # end def sqrt

    @staticmethod
    def log2(operand: MathExpr) -> MathExpr:
        """Base-2 logarithm operator"""
        from .functional.elementwise import log2 as elementwise_log2
        return elementwise_log2(operand)
    # end def log2

    @staticmethod
    def log10(operand: MathExpr) -> MathExpr:
        """Base-10 logarithm operator"""
        from .functional.elementwise import log10 as elementwise_log10
        return elementwise_log10(operand)
    # end def log10

    # endregion OPERATORS

    # region STATIC

    @staticmethod
    def next_id():
        """Get the next available node ID."""
        return MathExpr._next_id
    # end def next_id

    # endregion STATIC

    # region OVERRIDE

    def __repr__(self) -> str:
        """Return a string representation of the expression.

        Returns
        -------
        str
            A string representation of the expression.
        """
        op_name = self._op.name if self._op is not None else "Leaf"
        shape_str = str(self._shape)
        return f"<{self.__class__.__name__} #{self._id} {op_name} {self._dtype.value} {shape_str} c:{len(self._children)}>"
    # end __repr__

    def __hash__(self) -> int:
        # Identity hashing suitable for dict/set membership within a context.
        return hash(self._id)
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

        return self._id == other._id
    # end __eq__

    def __ne__(self, other):
        """
        """
        return not self.__eq__(other)
    # end __ne__

    # Operator overloading
    def __add__(self, other) -> MathExpr:
        """
        ...
        """
        return MathExpr.add(self, other)
    # end __add__

    def __radd__(self, other) -> MathExpr:
        """
        Add another value to this Scalar (reverse addition).
        """
        return MathExpr.add(other, self)
    # end __radd__

    def __sub__(self, other) -> MathExpr:
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        return MathExpr.sub(self, other)
    # end __sub__

    def __rsub__(self, other) -> MathExpr:
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        return MathExpr.sub(other, self)
    # end __rsub__

    def __mul__(self, other) -> MathExpr:
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        return MathExpr.mul(self, other)
    # end __mul__

    def __rmul__(self, other) -> MathExpr:
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        return MathExpr.mul(other, self)
    # end __rmul__

    def __truediv__(self, other) -> MathExpr:
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        return MathExpr.div(self, other)
    # end __truediv__

    def __rtruediv__(self, other) -> MathExpr:
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        return MathExpr.div(other, self)
    # end __rtruediv__

    def __pow__(self, other) -> MathExpr:
        """
        Power operator.
        """
        return MathExpr.pow(self, other)
    # end __pow__

    def __rpow__(self, other) -> MathExpr:
        """
        Reverse power operator.
        """
        return MathExpr.pow(other, self)
    # end __rpow__

    def __neg__(self) -> MathExpr:
        """
        Negation operator.
        """
        return MathExpr.neg(self)
    # end def __neg__

    # Override less
    def __lt__(self, other):
        """
        Check if the scalar value is less than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        pass
    # end __lt__

    # Override less or equal
    def __le__(self, other):
        """
        Check if the scalar value is less than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        pass
    # end __le__

    # Override greater
    def __gt__(self, other):
        """
        Check if the scalar value is greater than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        pass
    # end __gt__

    # Override greater or equal
    def __ge__(self, other):
        """
        Check if the scalar value is greater than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        pass
    # end __ge__

    # endregion OVERRIDE

# end class MathExpr


# An expression which does not contain sub-expressions
class MathLeaf(MathExpr, ABC):
    """
    Abstract base class for leaf nodes in the expression tree.

    MathLeaf represents a terminal node in a computational graph (CG) that
    stores a value directly. Unlike operator nodes, it does not compute from
    children. Typical implementations include constants, variables, and other
    primitive values that may be combined to form larger expressions.
    """

    # Arity - always 0 for leaf nodes as they don't have child expressions
    arity = 0

    def __init__(
            self,
            *,
            name: str,
            data: Any,
            dtype: DType,
            shape: Shape,
            mutable: bool = True
    ):
        """
        Initialize a new leaf node with the specified value.

        Parameters
        ----------
        data : Any
            Initial value to store. The concrete type is determined by the
            subclass.
        mutable : bool, optional
            If ``False``, the value cannot be modified after initialization.
        """
        # Init
        super(MathLeaf, self).__init__(
            name=name,
            op=None,
            children=(),
            dtype=dtype,
            shape=shape
        )
        self._mutable = mutable
        self._data: Any = data
    # end __init__

    # region PROPERTIES

    @property
    def data(self) -> Any:
        """Get the value of this leaf node."""
        return self._data
    # end def data

    @property
    def mutable(self) -> bool:
        """True iff the value can be modified."""
        return self._mutable
    # end def mutable

    # endregion PROPERTIES

    # region PUBLIC

    def eval(self, **kwargs) -> Any:
        """
        """
        return self._eval(**kwargs)
    # end _get

    def set(self, value: Any):
        """
        Set the value of this leaf node.
        """
        if not self._mutable:
            raise RuntimeError("Cannot set immutable leaf node.")
        # end if
        self._set(value)
    # end def set

    # endregion PUBLIC

    # region PRIVATE

    @abstractmethod
    def _set(self, data: Any) -> None:
        """
        Set the internal value of this leaf node.
        """
        raise NotImplementedError("Leaf nodes must implement _set.")
    # end _set

    def _eval(self, **kwargs) -> Any:
        """Evaluate this leaf node in the current context.
        """
        if self.name in kwargs:
            return kwargs[self.name]
        else:
            return self._data
        # end if
    # end def _eval

    # endregion PRIVATE

# end MathLeaf
