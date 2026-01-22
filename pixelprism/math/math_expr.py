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

"""
Symbolic math primitives used throughout Pixel Prism.

The :mod:`pixelprism.math.math_expr` module defines the canonical interfaces
implemented by every symbolic expression node in the system.  These nodes form
the immutable directed acyclic graphs that power algebraic manipulation,
automatic differentiation, and execution planning.  Each node advertises a
``DType`` and ``Shape`` describing the element type and tensor extent of its
value, and exposes helpers for structured traversal.

The documentation below mirrors the runtime implementation.  Every public
method follows the NumPy docstring convention to make the API consumable by the
interactive help system, by static tooling, and by our reference docs.
"""

# Imports
from __future__ import annotations
import weakref
from abc import ABC, abstractmethod
from typing import Any, FrozenSet, List, Optional, Tuple, Union, Dict
import numpy as np

from .dtype import DType
from .shape import Shape
from .tensor import Tensor
from .context import get_value

__all__ = [
    "MathNode",
    "MathLeaf",
    "Variable",
    "Constant",
    "SliceExpr",
    "MathExprError",
    "MathExprOperatorError",
    "MathExprValidationError",
    "MathExprLookupError",
    "MathExprNotImplementedError",
]


class MathExprError(RuntimeError):
    """
    Base exception for symbolic math expression failures.

    Subclasses capture finer grain failure modes (validation, lookup, etc.)
    to keep error handling explicit without exposing the internals of this
    module to callers.
    """


class MathExprOperatorError(MathExprError):
    """Raised when an operator declaration does not match a node definition."""


class MathExprValidationError(MathExprError):
    """Raised when runtime metadata (dtype, shape, etc.) violates invariants."""


class MathExprLookupError(MathExprError):
    """Raised when an expression, variable, or constant cannot be located."""


class MathExprNotImplementedError(MathExprError):
    """Raised when a subclass fails to implement an abstract requirement."""


class MathExpr(ABC):

    @abstractmethod
    def eval(self) -> Tensor:
        """
        Evaluate this expression in the active math context.

        Returns
        -------
        Tensor
            Result of executing ``self.op`` with evaluated children.
        """
    # end def eval

    @abstractmethod
    def variables(self) -> List[MathExpr]:
        """
        Enumerate variable leaves reachable from this node.

        Returns
        -------
        list[MathExpr]
            List of :class:`Variable` instances (duplicates possible).
        """
    # end def variables

    @abstractmethod
    def constants(self) -> List[MathExpr]:
        """
        Enumerate constant leaves reachable from this node.

        Returns
        -------
        list[MathExpr]
            List of :class:`Constant` instances (duplicates possible).
        """
    # end def constants

    @abstractmethod
    def contains(
            self,
            leaf: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        """
        Test whether ``var`` appears in the expression tree.

        Parameters
        ----------
        leaf : str or MathNode
            Reference searched for within the tree.  Strings are matched
            against node names; ``MathExpr`` instances are matched either by
            identity or by their ``name``.
        by_ref : bool, default False
            When ``True`` the search compares identities instead of names.
        check_operator : bool, default True
            When ``True`` the search also queries the operator to determine if
            it captures ``var`` internally.
        look_for : Optional[str], default None
            Can be None, "var", or "const"

        Returns
        -------
        bool
            ``True`` when ``var`` was located in ``self`` or any child.
        """
    # end def contains

    @abstractmethod
    def contains_variable(
            self,
            variable: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a variable `variable`"""
    # end def contains_variable

    @abstractmethod
    def contains_constant(
            self,
            constant: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a constant `constant`"""
    # end def contains_constant

    @abstractmethod
    def replace(self, old_m: MathNode, new_m: MathNode):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
    # end def replace

    @abstractmethod
    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
    # end rename

# end class MathBase


class MathNode(MathExpr):
    """
    Canonical symbolic expression node.

    Each node represents a typed tensor value produced by a symbolic
    computation.  Nodes are immutable once constructed: they expose their
    operator, ordered children, dtype, and shape as read-only metadata so that
    graph transformations can reason about the structure deterministically.
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
            op: Optional,
            children: Tuple["MathNode", ...],
            dtype: DType,
            shape: Shape
    ) -> None:
        """
        Build a symbolic node (typically invoked by subclasses).

        Parameters
        ----------
        name : str or None
            Optional debugging label for the node.
        op : Operator or None
            Operator descriptor.  ``None`` indicates a leaf node.
        children : tuple[MathExpr, ...]
            Ordered child expressions consumed by ``op``.
        dtype : DType
            Element dtype advertised by the node.
        shape : Shape
            Symbolic tensor shape describing the node's extent.
        """
        self._id: int = MathNode._next_id
        self._name: str = name
        self._op = op
        self._children: Tuple["MathNode", ...] = children
        self._dtype: DType = dtype
        self._shape: Shape = shape
        self._parents_weak: "weakref.WeakSet[MathNode]" = weakref.WeakSet()
        self._check_operator()
        self._register_as_parent_of(*children)
        MathNode._next_id += 1
    # end __init__

    # endregion CONSTRUCTOR

    # region PROPERTIES

    @property
    def identifier(self) -> int:
        """
        Returns
        -------
        int
            Unique identifier for the node.
        """
        return self._id
    # end def identifier

    @property
    def name(self) -> Optional[str]:
        """
        Returns
        -------
        str or None
            Optional user-provided display name.
        """
        return self._name
    # end def name

    @property
    def op(self) -> Optional[Any]:
        """
        Returns
        -------
        Operator or None
            Operator descriptor for non-leaf nodes.
        """
        return self._op
    # end op

    @property
    def children(self) -> Tuple["MathNode", ...]:
        """
        Returns
        -------
        tuple[MathExpr, ...]
            Child nodes (empty tuple for leaves).
        """
        return self._children
    # end def children

    @property
    def dtype(self) -> DType:
        """
        Returns
        -------
        DType
            Element dtype for this expression.
        """
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """
        Returns
        -------
        Shape
            Symbolic shape describing the tensor extent.
        """
        return self._shape
    # end def shape

    @property
    def rank(self) -> int:
        """
        Returns
        -------
        int
            Rank of the advertised tensor.
        """
        return self._shape.rank
    # end def rank

    @property
    def parents(self) -> FrozenSet["MathNode"]:
        """
        Returns
        -------
        frozenset[MathNode]
            Best-effort view of nodes that consume this expression.
        """
        return frozenset(self._parents_weak)
    # end def parents

    @property
    def arity(self) -> int:
        """
        Returns
        -------
        int
            Number of children supplied to the node.
        """
        return len(self._children)
    # end def arity

    # endregion PROPERTIES

    # region PUBLIC

    def eval(self) -> Tensor:
        """
        Evaluate this node in the active math context.

        Returns
        -------
        Tensor
            Result of executing ``self.op`` with evaluated children.
        """
        return self._op.eval(operands=self._children)
    # end def eval

    def is_node(self) -> bool:
        """
        Returns
        -------
        bool
            ``True`` when the expression has an operator (non-leaf).
        """
        return self._op is not None
    # end def is_node

    def is_leaf(self) -> bool:
        """
        Returns
        -------
        bool
            ``True`` when the expression has no children.
        """
        return len(self._children) == 0
    # end is_leaf

    def is_scalar(self) -> bool:
        """
        Is the expression a scalar?

        Returns:
            ``True`` when the expression is a leaf and its shape is (1,)
        """
        return self._shape.rank == 0
    # end def is_scalar

    def is_vector(self) -> bool:
        """
        Is the expression a vector?

        Returns:
            ``True`` when the expression is a leaf and its shape is (n,)
        """
        return self._shape.rank == 1
    # end def is_vector

    def is_matrix(self) -> bool:
        """
        Is the expression a matrix?

        Returns:
            ``True`` when the expression is a leaf and its shape is (m,n)
        """
        return self._shape.rank == 2
    # end is_matrix

    def is_constant(self):
        """Does the expression contain only constant values?"""
        rets = [o.is_constant() for o in self._children]
        return all(rets)
    # end def is_constant

    def is_variable(self):
        """Does the expression contain a variable?"""
        rets = [o.is_variable() for o in self._children]
        return any(rets)
    # end def is_variable

    def is_higher_order(self):
        """
        Is the expression higher order?

        Returns:
            ``True`` when the expression is a leaf and its shape is (m,n,...)
        """
        return self._shape.rank > 2
    # end def is_higher_order

    def add_parent(self, parent: "MathNode"):
        """
        Register ``parent`` as a consumer of this node.

        Parameters
        ----------
        parent : MathNode
            Expression that references ``self`` as an input.
        """
        self._parents_weak.add(parent)
    # end def parent

    def variables(self) -> List:
        """
        Enumerate variable leaves reachable from this node.

        Returns
        -------
        list[MathNode]
            List of :class:`Variable` instances (duplicates possible).
        """
        vars: List = list()
        for c in self._children:
            vars.extend(c.variables())
        # end for
        return vars
    # end def variables

    def constants(self) -> List:
        """
        Enumerate constant leaves reachable from this node.

        Returns
        -------
        list[MathNode]
            List of :class:`Constant` instances (duplicates possible).
        """
        constants: List = list()
        for c in self._children:
            constants.extend(c.constants())
        # end for
        return constants
    # end def constants

    def leaves(self) -> List:
        """
        Collect all leaves reachable from the node.

        Returns
        -------
        list[MathNode]
            Combined list of variables and constants.
        """
        return self.variables() + self.constants()
    # end def leaves

    def contains(
            self,
            leaf: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        """
        Test whether ``var`` appears in the expression tree.

        Parameters
        ----------
        leaf : str or MathNode
            Reference searched for within the tree.  Strings are matched
            against node names; ``MathExpr`` instances are matched either by
            identity or by their ``name``.
        by_ref : bool, default False
            When ``True`` the search compares identities instead of names.
        check_operator : bool, default True
            When ``True`` the search also queries the operator to determine if
            it captures ``var`` internally.
        look_for : Optional[str], default None
            Can be None, "var", or "const"

        Returns
        -------
        bool
            ``True`` when ``var`` was located in ``self`` or any child.
        """
        rets = [
            c.contains(leaf, by_ref=by_ref, check_operator=check_operator, look_for=look_for)
            for c in self._children
        ]
        return (
                any(rets) or
                (by_ref and self == leaf) or
                (check_operator and self._op.contains(leaf, by_ref=by_ref, look_for=look_for))
        )
    # end def contains

    def contains_variable(
            self,
            variable: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a variable `variable`"""
        return self.contains(variable, by_ref=by_ref, check_operator=check_operator, look_for="var")
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a constant `constant`"""
        return self.contains(constant, by_ref=by_ref, check_operator=check_operator, look_for="const")
    # end def contains_constant

    def replace(self, old_m: MathNode, new_m: MathNode):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
        if self is old_m:
            raise ValueError("Cannot replace a node with itself.")
        # end if

        new_children = [
            new_m if child is old_m else child
            for child in self._children
        ]
        self._children = tuple(new_children)
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        rename_dict = {}
        for child in self._children:
            rn_out = child.rename(old_name, new_name)
            rename_dict.update(rn_out)
        # end for
        return rename_dict
    # end rename

    # endregion PUBLIC

    # region PRIVATE

    def _check_operator(self) -> None:
        """
        Validate that the supplied operator matches the node signature.

        Raises
        ------
        MathExprOperatorError
            If the operator arity does not match ``self.arity``.
        """
        if self._op is not None and self.arity != self._op.arity:
            raise MathExprOperatorError(
                f"Operator and arity mismatch: "
                f"{self._op.name}({self.arity}) != {self.__class__.__name__}({self._op.arity})"
            )
        # end if
    # end _def_check_operator

    def _register_as_parent_of(self, *children: "MathNode") -> None:
        """
        Record weak parent links for inverse traversal.

        Parameters
        ----------
        *children : MathNode
            Child expressions that should keep a weak reference to ``self``.
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
    def add(operand1: MathNode, operand2: MathNode) -> MathNode:
        """
        Create an elementwise addition node.

        Parameters
        ----------
        operand1 : MathNode
            Left operand.
        operand2 : MathNode
            Right operand.

        Returns
        -------
        MathNode
            Symbolic addition of ``operand1`` and ``operand2``.
        """
        from .functional.elementwise import add
        return add(operand1, operand2)
    # end def add

    @staticmethod
    def sub(operand1: MathNode, operand2: MathNode) -> MathNode:
        """
        Create an elementwise subtraction node.

        Parameters
        ----------
        operand1 : MathNode
            Left operand.
        operand2 : MathNode
            Right operand to subtract.

        Returns
        -------
        MathNode
            Symbolic subtraction ``operand1 - operand2``.
        """
        from .functional.elementwise import sub
        return sub(operand1, operand2)
    # end def sub

    @staticmethod
    def mul(operand1: MathNode, operand2: MathNode) -> MathNode:
        """
        Create an elementwise multiplication node.

        Parameters
        ----------
        operand1 : MathNode
            Left operand.
        operand2 : MathNode
            Right operand.

        Returns
        -------
        MathNode
            Symbolic product of the operands.
        """
        from .functional.elementwise import mul
        return mul(operand1, operand2)
    # end def mul

    @staticmethod
    def div(operand1: MathNode, operand2: MathNode) -> MathNode:
        """
        Create an elementwise division node.

        Parameters
        ----------
        operand1 : MathNode
            Numerator expression.
        operand2 : MathNode
            Denominator expression.

        Returns
        -------
        MathNode
            Symbolic quotient ``operand1 / operand2``.
        """
        from .functional.elementwise import div
        return div(operand1, operand2)
    # end def div

    @staticmethod
    def neg(operand: MathNode) -> MathNode:
        """
        Create an elementwise negation node.

        Parameters
        ----------
        operand : MathNode
            Expression to negate.

        Returns
        -------
        MathNode
            Symbolic negation of ``operand``.
        """
        from .functional.elementwise import neg
        return neg(operand)
    # end def neg

    @staticmethod
    def pow(operand1: MathNode, operand2: MathNode) -> MathNode:
        """
        Create an elementwise power node.

        Parameters
        ----------
        operand1 : MathNode
            Base expression.
        operand2 : MathNode
            Exponent expression.

        Returns
        -------
        MathNode
            Symbolic representation of ``operand1 ** operand2``.
        """
        from .functional.elementwise import pow as elementwise_pow
        return elementwise_pow(operand1, operand2)
    # end def pow

    @staticmethod
    def exp(operand: MathNode) -> MathNode:
        """
        Create an elementwise exponential node.

        Parameters
        ----------
        operand : MathNode
            Expression whose entries will be exponentiated.

        Returns
        -------
        MathNode
            Node computing ``exp(operand)``.
        """
        from .functional.elementwise import exp as elementwise_exp
        return elementwise_exp(operand)
    # end def exp

    @staticmethod
    def log(operand: MathNode) -> MathNode:
        """
        Create an elementwise natural-logarithm node.

        Parameters
        ----------
        operand : MathNode
            Expression whose entries will be transformed.

        Returns
        -------
        MathNode
            Node computing ``log(operand)``.
        """
        from .functional.elementwise import log as elementwise_log
        return elementwise_log(operand)
    # end def log

    @staticmethod
    def sqrt(operand: MathNode) -> MathNode:
        """
        Create an elementwise square-root node.

        Parameters
        ----------
        operand : MathNode
            Expression whose entries will be square-rooted.

        Returns
        -------
        MathNode
            Node computing ``sqrt(operand)``.
        """
        from .functional.elementwise import sqrt as elementwise_sqrt
        return elementwise_sqrt(operand)
    # end def sqrt

    @staticmethod
    def log2(operand: MathNode) -> MathNode:
        """
        Create an elementwise base-2 logarithm node.

        Parameters
        ----------
        operand : MathNode
            Expression whose entries will be transformed.

        Returns
        -------
        MathNode
            Node computing ``log2(operand)``.
        """
        from .functional.elementwise import log2 as elementwise_log2
        return elementwise_log2(operand)
    # end def log2

    @staticmethod
    def log10(operand: MathNode) -> MathNode:
        """
        Create an elementwise base-10 logarithm node.

        Parameters
        ----------
        operand : MathNode
            Expression whose entries will be transformed.

        Returns
        -------
        MathNode
            Node computing ``log10(operand)``.
        """
        from .functional.elementwise import log10 as elementwise_log10
        return elementwise_log10(operand)
    # end def log10

    @staticmethod
    def matmul(operand1: MathNode, operand2: MathNode) -> MathNode:
        """
        Create a matrix multiplication node.
        """
        from .functional.linear_algebra import matmul
        return matmul(operand1, operand2)
    # end def matmul

    @staticmethod
    def getitem(operand: MathNode, index: Union[int, slice]) -> MathNode:
        """
        Create a getitem node.
        """
        from .functional.structure import getitem
        return getitem(operand, index)
    # end def getitem

    # endregion OPERATORS

    # region STATIC

    @staticmethod
    def next_id() -> int:
        """
        Returns
        -------
        int
            Next identifier that will be assigned to a :class:`MathExpr`.
        """
        return MathNode._next_id
    # end def next_id

    # endregion STATIC

    # region OVERRIDE

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            Readable representation containing type, id, dtype, and shape.
        """
        op_name = self._op.name if self._op is not None else "Leaf"
        shape_str = str(self._shape)
        return f"<{self.__class__.__name__} #{self._id} {op_name} {self._dtype.value} {shape_str} c:{len(self._children)}>"
    # end __repr__

    def __hash__(self) -> int:
        """
        Returns
        -------
        int
            Identity hash suitable for storing nodes in sets/dicts.
        """
        return hash(self._id)
    # end __hash__

    def __eq__(self, other: object) -> bool:
        """
        Compare node identities.

        Parameters
        ----------
        other : object
            Expression being compared.

        Returns
        -------
        bool
            ``True`` when both operands share the same identifier.
        """
        if self is other:
            return True
        # end if

        if not isinstance(other, MathNode):
            return False
        # end if

        return self._id == other._id
    # end __eq__

    def __ne__(self, other: object) -> bool:
        """
        Logical negation of :meth:`__eq__`.
        """
        return not self.__eq__(other)
    # end __ne__

    # Operator overloading
    def __add__(self, other) -> MathNode:
        """
        Convenience forward addition operator.
        """
        return MathNode.add(self, other)
    # end __add__

    def __radd__(self, other) -> MathNode:
        """
        Reverse addition operator.
        """
        return MathNode.add(other, self)
    # end __radd__

    def __sub__(self, other) -> MathNode:
        """
        Convenience forward subtraction operator.
        """
        return MathNode.sub(self, other)
    # end __sub__

    def __rsub__(self, other) -> MathNode:
        """
        Reverse subtraction operator.
        """
        return MathNode.sub(other, self)
    # end __rsub__

    def __mul__(self, other) -> MathNode:
        """
        Convenience forward multiplication operator.
        """
        return MathNode.mul(self, other)
    # end __mul__

    def __rmul__(self, other) -> MathNode:
        """
        Reverse multiplication operator.
        """
        return MathNode.mul(other, self)
    # end __rmul__

    def __truediv__(self, other) -> MathNode:
        """
        Convenience forward division operator.
        """
        return MathNode.div(self, other)
    # end __truediv__

    def __rtruediv__(self, other) -> MathNode:
        """
        Reverse division operator.
        """
        return MathNode.div(other, self)
    # end __rtruediv__

    def __pow__(self, other) -> MathNode:
        """
        Convenience power operator.
        """
        return MathNode.pow(self, other)
    # end __pow__

    def __rpow__(self, other) -> MathNode:
        """
        Reverse power operator.
        """
        return MathNode.pow(other, self)
    # end __rpow__

    def __neg__(self) -> MathNode:
        """
        Negation operator.
        """
        return MathNode.neg(self)
    # end def __neg__

    def __matmul__(self, other):
        return MathNode.matmul(self, other)
    # end def __matmul__

    def __rmatmul__(self, other):
        return MathNode.matmul(other, self)
    # end def __rmatmul__

    # Override less
    def __lt__(self, other) -> MathNode:
        """
        Placeholder less-than operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        raise MathExprNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end __lt__

    # Override less or equal
    def __le__(self, other) -> MathNode:
        """
        Placeholder less-or-equal operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        raise MathExprNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end __le__

    # Override greater
    def __gt__(self, other) -> MathNode:
        """
        Placeholder greater-than operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        raise MathExprNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end __gt__

    # Override greater or equal
    def __ge__(self, other) -> MathNode:
        """
        Placeholder greater-or-equal operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        raise MathExprNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end __ge__

    # Get item
    def __getitem__(self, item: Union[int, slice]):
        return MathNode.getitem(item)
    # end def __getitem__

    # endregion OVERRIDE

# end class MathExpr


# An expression which does not contain sub-expressions
class MathLeaf(MathNode, ABC):
    """
    Abstract base for terminal expressions.

    Leaf nodes do not own operators or children; instead they store metadata
    needed to look up runtime tensors (variables) or to hold literal values
    (constants).  Subclasses must implement ``_eval`` and the traversal helpers
    that expose any nested variables/constants they reference.
    """

    # Arity - always 0 for leaf nodes as they don't have child expressions
    arity = 0

    def __init__(
            self,
            *,
            name: str,
            dtype: DType,
            shape: Shape
    ):
        """
        Initialize a new leaf node.

        Parameters
        ----------
        name : str
            Identifier assigned to the node.
        dtype : DType
            Element dtype advertised by the leaf.
        shape : Shape
            Symbolic tensor shape of the stored value.
        """
        # Init
        super(MathLeaf, self).__init__(
            name=name,
            op=None,
            children=(),
            dtype=dtype,
            shape=shape
        )
    # end __init__

    # region PUBLIC

    def eval(self) -> Tensor:
        """
        Evaluate the leaf using its subclass implementation.

        Returns
        -------
        Tensor
            Runtime tensor value.
        """
        return self._eval()
    # end _get

    @abstractmethod
    def variables(self) -> list:
        """
        Returns
        -------
        list
            Variable leaves referenced by the node.
        """
    # end def variables

    @abstractmethod
    def constants(self) -> List:
        """
        Returns
        -------
        list
            Constant leaves referenced by the node.
        """
    # end def constants

    def is_constant(self):
        """Does the expression contain only constant values?"""
        raise NotImplementedError("Leaf nodes do not support is_constant.")
    # end def is_constant

    def is_variable(self):
        """Does the expression contain a variable?"""
        raise NotImplementedError("Leaf nodes do not support is_variable.")
    # end def is_variable

    def contains(
            self,
            leaf: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        """
        Check if ``var`` references this leaf.

        Parameters
        ----------
        leaf : str or MathNode
            Variable name or expression to look for.
        by_ref : bool, default False
            When ``True`` only identity matches are considered.
        check_operator : bool, default True
            Ignored for leaves; included for API compatibility.
        look_for : Optional[str], default None
            Can be None (all), "var" or "const"

        Returns
        -------
        bool
            ``True`` when the leaf matches ``var``.

        Raises
        ------
        MathExprValidationError
            If ``by_ref`` is ``True`` while ``var`` is provided as a string.
        """
        if by_ref:
            if type(leaf) is str:
                raise MathExprValidationError(f"Cannot find by reference if string given: var={leaf}.")
            # end if
            if isinstance(leaf, MathNode) and leaf is self:
                return True
            # end if
        else:
            if type(leaf) is str and leaf == self.name:
                return True
            elif isinstance(leaf, MathNode) and leaf.name == self.name:
                return True
            # end if
        # end if
        return False
    # en def contains

    def replace(self, old_m: MathNode, new_m: MathNode):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
        raise MathExprNotImplementedError("Leaf nodes do not support replace.")
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        raise MathExprNotImplementedError("Leaf nodes do not support rename.")
    # end rename

    # endregion PUBLIC

    # region PRIVATE

    @abstractmethod
    def _eval(self) -> Tensor:
        """
        Evaluate this leaf node in the current context.

        Raises
        ------
        MathExprNotImplementedError
            Always raised unless overridden.
        """
        raise MathExprNotImplementedError("Leaf nodes must implement _eval.")
    # end def _eval

    # endregion PRIVATE

# end MathLeaf


class Variable(MathLeaf):
    """
    Leaf node bound to a runtime tensor via the active context.
    """

    def __init__(
            self,
            *,
            name: str,
            dtype: DType,
            shape: Shape
    ):
        """
        Parameters
        ----------
        name : str
            Identifier used to look up the variable in the context.
        dtype : DType
            Element dtype advertised by the variable.
        shape : Shape
            Symbolic tensor shape of the variable.
        """
        # Super
        super(Variable, self).__init__(
            name=name,
            dtype=dtype,
            shape=shape
        )
    # end __init__

    # region PROPERTIES

    @property
    def value(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            Tensor currently stored in the runtime context.
        """
        return get_value(self._name)
    # end def value

    @property
    def dtype(self) -> DType:
        """
        Returns
        -------
        DType
            Element dtype advertised by the variable.
        """
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """
        Returns
        -------
        Shape
            Symbolic shape of the variable.
        """
        return self._shape
    # end def shape

    @property
    def dims(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        tuple[int, ...]
            Tuple of shape dimensions.
        """
        return self._shape.dims
    # end def dims

    @property
    def ndim(self) -> int:
        """
        Returns
        -------
        int
            Rank (number of axes) of the variable.
        """
        return self._shape.rank
    # end def dim

    @property
    def rank(self) -> int:
        """
        Returns
        -------
        int
            Rank (number of axes) of the variable.
        """
        return self._shape.rank
    # end def rank

    @property
    def size(self) -> int:
        """
        Returns
        -------
        int
            Total number of elements (when fully defined).
        """
        return self._shape.size
    # end def size

    @property
    def n_elements(self) -> int:
        """
        Returns
        -------
        int
            Total number of elements (alias for ``size``).
        """
        return self._shape.size
    # end def n_elements

    # endregion PROPERTIES

    # region PUBLIC

    def variables(self) -> list:
        """
        Returns
        -------
        list
            List containing ``self``.
        """
        return [self]
    # end def variable

    def constants(self) -> List:
        """
        Returns
        -------
        list
            Empty list because variables do not reference constants.
        """
        return []
    # end def constants

    def is_constant(self):
        """Does the expression contain only constant values?"""
        return False
    # end def is_constant

    def is_variable(self):
        """Does the expression contain a variable?"""
        return True
    # end def is_variable

    def contains(
            self,
            leaf: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        """
        Check if ``var`` references this leaf.

        Parameters
        ----------
        leaf : str or MathNode
            Variable name or expression to look for.
        by_ref : bool, default False
            When ``True`` only identity matches are considered.
        check_operator : bool, default True
            Ignored for leaves; included for API compatibility.
        look_for : Optional[str], default None
            Can be None (all), "var" or "const"

        Returns
        -------
        bool
            ``True`` when the leaf matches ``var``.

        Raises
        ------
        MathExprValidationError
            If ``by_ref`` is ``True`` while ``var`` is provided as a string.
        """
        if look_for is None or look_for == "var":
            return super(Variable, self).contains(
                leaf=leaf,
                by_ref=by_ref,
                check_operator=check_operator,
                look_for=None
            )
        # end if
        return False
    # en def contains

    def copy(
            self,
            name: str
    ) -> 'Variable':
        """
        Create a shallow copy with a new name.

        Parameters
        ----------
        name : str
            Name assigned to the clone.

        Returns
        -------
        Variable
            New variable sharing dtype/shape metadata.
        """
        return Variable(
            name=name,
            dtype=self._dtype.copy(),
            shape=self._shape.copy(),
        )
    # end def copy

    def replace(self, old_m: MathNode, new_m: MathNode):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
        pass
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        if self._name == old_name:
            self._name = new_name
        # end if
        return {old_name: new_name}
    # end rename

    # endregion PUBLIC

    # region STATIC

    @staticmethod
    def create(
            *,
            name: Optional[str],
            dtype: DType,
            shape: Shape
    ) -> 'Variable':
        """
        Create a new variable node.

        Parameters
        ----------
        name : str or None
            Optional explicit name.  Auto-generated when ``None``.
        dtype : DType
            Element dtype.
        shape : Shape
            Symbolic shape.

        Returns
        -------
        Variable
            Variable leaf bound to the provided metadata.
        """
        name = name or f"var_{Variable._next_id}"
        return Variable(
            name=name,
            dtype=dtype,
            shape=shape
        )
    # end def create

    # endregion STATIC

    # region PRIVATE

    def _eval(self) -> Tensor:
        """
        Evaluate this leaf node in the current context.

        Returns
        -------
        Tensor
            Tensor retrieved from the active context.

        Raises
        ------
        MathExprLookupError
            If the variable cannot be found.
        MathExprValidationError
            If the context provides a tensor with mismatched dtype/shape.
        """
        var_val: Tensor = get_value(self._name)
        if var_val is None:
            raise MathExprLookupError(f"Variable {self._name} not found in context.")
        # end if
        if var_val.dtype != self._dtype:
            var_val = var_val.astype(self._dtype)
        # end if
        if var_val.shape != self._shape:
            raise MathExprValidationError(
                f"Variable {self._name} shape mismatch: {var_val.shape} != {self._shape}"
            )
        # end if
        return var_val
    # end def _eval

    # endregion PRIVATE

    # region OVERRIDE

    def __str__(self):
        """
        Returns
        -------
        str
            Human-readable description of the variable.
        """
        return f"variable({self.name}, dtype={self._dtype}, shape={self._shape})"
    # end __str__

    def __repr__(self):
        """
        Returns
        -------
        str
            Debug representation identical to :meth:`__str__`.
        """
        return self.__str__()
    # end __repr__

    # endregion OVERRIDE

# end class Variable


class Constant(MathLeaf):
    """
    Leaf node that stores an immutable tensor value.
    """

    def __init__(
            self,
            *,
            name: str,
            data: Tensor
    ):
        """
        Parameters
        ----------
        name : str
            Identifier assigned to the constant.
        data : Tensor
            Tensor value stored by the constant.
        """
        # Super
        super(Constant, self).__init__(
            name=name,
            dtype=data.dtype,
            shape=data.shape
        )
        if data.shape != self._shape:
            raise MathExprValidationError(f"Constant shape mismatch: {data.shape} != {self._shape}")
        # end if
        if data.dtype != self._dtype:
            raise MathExprValidationError(f"Constant dtype mismatch: {data.dtype} != {self._dtype}")
        # end if
        self._data = data
    # end __init__

    # region PROPERTIES

    @property
    def value(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            Stored tensor value.
        """
        return self._data
    # end def value

    @property
    def dtype(self) -> DType:
        """
        Returns
        -------
        DType
            Element dtype.
        """
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """
        Returns
        -------
        Shape
            Symbolic shape of the tensor.
        """
        return self._shape
    # end def shape

    @property
    def dims(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        tuple[int, ...]
            Tuple of dimensions.
        """
        return self._shape.dims
    # end def dims

    @property
    def ndim(self) -> int:
        """
        Returns
        -------
        int
            Rank (number of axes).
        """
        return self._shape.rank
    # end def dim

    @property
    def rank(self) -> int:
        """
        Returns
        -------
        int
            Rank (number of axes).
        """
        return self._shape.rank
    # end def rank

    @property
    def size(self) -> int:
        """
        Returns
        -------
        int
            Total number of elements.
        """
        return self._shape.size
    # end def size

    @property
    def n_elements(self) -> int:
        """
        Returns
        -------
        int
            Total number of elements.
        """
        return self._shape.size
    # end def n_elements

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, data: Tensor) -> None:
        """
        Replace the stored tensor.

        Parameters
        ----------
        data : Tensor
            Replacement data.

        Raises
        ------
        MathExprValidationError
            If dtype or shape mismatches occur.
        """
        if data.shape != self._shape:
            raise MathExprValidationError(f"Constant shape mismatch: {data.shape} != {self._shape}")
        # end if
        if data.dtype != self._dtype:
            raise MathExprValidationError(f"Constant dtype mismatch: {data.dtype} != {self._dtype}")
        # end if
        self._data = data
    # end def set

    def variables(self) -> list:
        """
        Returns
        -------
        list
            Empty list because constants do not reference variables.
        """
        return []
    # end def variable

    def constants(self) -> List:
        """
        Returns
        -------
        list
            List containing ``self``.
        """
        return [self]
    # end def constants

    def is_constant(self):
        """Does the expression contain only constant values?"""
        return True
    # end def is_constant

    def is_variable(self):
        """Does the expression contain a variable?"""
        return False
    # end def is_variable

    def contains(
            self,
            leaf: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        """
        Check if ``var`` references this leaf.

        Parameters
        ----------
        leaf : str or MathNode
            Variable name or expression to look for.
        by_ref : bool, default False
            When ``True`` only identity matches are considered.
        check_operator : bool, default True
            Ignored for leaves; included for API compatibility.
        look_for : Optional[str], default None
            Can be None (all), "var" or "const"

        Returns
        -------
        bool
            ``True`` when the leaf matches ``var``.

        Raises
        ------
        MathExprValidationError
            If ``by_ref`` is ``True`` while ``var`` is provided as a string.
        """
        if look_for is None or look_for == "const":
            super(Constant, self).contains(
                leaf=leaf,
                by_ref=by_ref,
                check_operator=check_operator,
                look_for=None
            )
        # end if
        return False
    # en def contains

    def copy(
            self,
            name: str
    ) -> 'Constant':
        """
        Create a copy of the constant with a new name.

        Parameters
        ----------
        name : str
            Identifier assigned to the clone.

        Returns
        -------
        Constant
            New constant carrying a copy of the tensor.
        """
        return Constant(
            name=name,
            data=self._data.copy()
        )
    # end def copy

    def replace(self, old_m: MathNode, new_m: MathNode):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
        pass
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        if self._name == old_name:
            self._name = new_name
        # end if
        return {old_name: new_name}
    # end rename

    # endregion PUBLIC

    # region PRIVATE

    def _eval(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            Copy of the stored tensor.
        """
        return self._data.copy()
    # end def _eval

    # endregion PRIVATE

    # region STATIC

    @staticmethod
    def create(name: str, data: Tensor):
        """Create a new constant node."""
        return Constant(
            name=name,
            data=data
        )
    # end def create

    # endregion STATIC

    # region OVERRIDE

    def __str__(self):
        """
        Returns
        -------
        str
            Human-readable description of the constant.
        """
        return f"constant({self.name}, dtype={self._dtype}, shape={self._shape})"
    # end __str__

    def __repr__(self):
        """
        Returns
        -------
        str
            Debug representation identical to :meth:`__str__`.
        """
        return self.__str__()
    # end __repr__

    # endregion OVERRIDE

# end class Constant


class SliceExpr(MathExpr):
    """
    Immutable representation of a Python-style slice using :class:`MathNode` bounds.

    The object mirrors the behavior of Python's native ``slice`` (and thereby
    NumPy's slicing convention) but stores each bound as an expression so
    transformations can reason about those values symbolically.  Bounds must be
    constant integer expressions so they can be materialized where necessary.

    Parameters
    ----------
    start : MathNode or int or None, optional
        Inclusive lower bound of the slice. ``None`` matches Python/NumPy semantics.
    stop : MathNode or int or None, optional
        Exclusive upper bound of the slice. ``None`` matches Python/NumPy semantics.
    step : MathNode or int or None, optional
        Slice stride. ``None`` matches Python/NumPy semantics.

    Attributes
    ----------
    start : Optional[MathNode]
        Symbolic representation of the ``start`` bound.
    stop : Optional[MathNode]
        Symbolic representation of the ``stop`` bound.
    step : Optional[MathNode]
        Symbolic representation of the ``step`` bound.

    Notes
    -----
    The evaluation rules match ``slice`` and ``numpy.s_`` exactly: ``start`` is
    inclusive, ``stop`` is exclusive, and ``step`` cannot evaluate to zero.  All
    bounds must evaluate to scalar integer tensors so their Python equivalents
    can be emitted where NumPy-compatible indexing is required.
    """

    __slots__ = ("_start", "_stop", "_step")

    def __init__(
            self,
            start: Optional[Union[MathNode, int]] = None,
            stop: Optional[Union[MathNode, int]] = None,
            step: Optional[Union[MathNode, int]] = None
    ):
        """
        Initialize a new :class:`SliceExpr`.

        Parameters
        ----------
        start : MathNode or int or None, optional
            Inclusive lower bound of the slice.
        stop : MathNode or int or None, optional
            Exclusive upper bound of the slice.
        step : MathNode or int or None, optional
            Slice stride.
        """
        self._start: Optional[MathNode] = self._coerce_bound("start", start)
        self._stop: Optional[MathNode] = self._coerce_bound("stop", stop)
        self._step: Optional[MathNode] = self._coerce_bound("step", step)

        if self._step.eval() == 0:
            raise MathExprValidationError(f"Slice step cannot be zero.")
        # end if
    # end def __init__

    # region PROPERTIES

    @property
    def start(self) -> Optional[MathNode]:
        """
        Returns
        -------
        Optional[MathNode]
            Symbolic expression for the ``start`` bound.
        """
        return self._start
    # end def start

    @property
    def stop(self) -> Optional[MathNode]:
        """
        Returns
        -------
        Optional[MathNode]
            Symbolic expression for the ``stop`` bound.
        """
        return self._stop
    # end def stop

    @property
    def step(self) -> Optional[MathNode]:
        """
        Returns
        -------
        Optional[MathNode]
            Symbolic expression for the ``step`` bound.
        """
        return self._step
    # end def step

    @property
    def start_value(self) -> Optional[int]:
        """
        Returns
        -------
        Optional[int]
            Concrete Python value for ``start`` when defined.
        """
        return self._expr_to_python(self._start)
    # end def start_value

    @property
    def stop_value(self) -> Optional[int]:
        """
        Returns
        -------
        Optional[int]
            Concrete Python value for ``stop`` when defined.
        """
        return self._expr_to_python(self._stop)
    # end def stop_value

    @property
    def step_value(self) -> Optional[int]:
        """
        Returns
        -------
        Optional[int]
            Concrete Python value for ``step`` when defined.
        """
        return self._expr_to_python(self._step)
    # end def step_value

    @property
    def as_slice(self) -> slice:
        """
        Returns
        -------
        slice
            Native ``slice`` object mirroring ``self``.
        """
        start_val = self.start_value if self.start is not None else None
        stop_val = self.stop_value if self.stop is not None else None
        step_val = self.step_value if self.step is not None else None

        if start_val is not None and type(start_val) != int:
            raise MathExprValidationError(f"Invalid start value: {start_val}")
        # end if

        if stop_val is not None and type(stop_val) != int:
            raise MathExprValidationError(f"Invalid stop value: {stop_val}")
        # end if

        if step_val is not None and type(step_val) != int:
            raise MathExprValidationError(f"Invalid step value: {step_val}")
        # end if

        return slice(start_val, stop_val, step_val)
    # end def as_slice

    # endregion PROPERTIES

    # region PUBLIC

    def to_slice(self) -> slice:
        """
        Returns
        -------
        slice
            Native ``slice`` object mirroring ``self``.
        """
        return self.as_slice
    # end def to_slice

    def eval(self) -> Tensor:
        """
        Evaluate the slice expression in the current context.
        """
        raise MathExprValidationError(f"Cannot evaluate slice expression: {self}")
    # end def eval

    def variables(self) -> List[MathExpr]:
        return []
    # end def variables

    def constants(self) -> List[MathExpr]:
        slice_const = list()
        slice_const.extend(self._start.constants() if self._start else [])
        slice_const.extend(self._stop.constants() if self._stop else [])
        slice_const.extend(self._step.constants() if self._step else [])
        return slice_const
    # end def constantes

    def contains(
            self,
            leaf: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool:
        children = [o for o in [self._start, self._stop, self._step] if o is not None]
        return any([
            o.contains(leaf, by_ref, check_operator, look_for)
            for o in children
        ])
    # end def contains

    def contains_variable(
            self,
            variable: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(variable, by_ref, check_operator, "var")
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, MathNode],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(constant, by_ref, check_operator, "const")
    # end def contains_constant

    def replace(self, old_m: MathNode, new_m: MathNode):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence."""
        if self._start is not None: self._start.replace(old_m, new_m)
        if self._stop is not None: self._stop.replace(old_m, new_m)
        if self._step is not None: self._step.replace(old_m, new_m)
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        if self._start is not None: self._start.rename(old_name, new_name)
        if self._stop is not None: self._stop.rename(old_name, new_name)
        if self._step is not None: self._step.rename(old_name, new_name)
        return {old_name: new_name}
    # end def rename

    # endregion PUBLIC

    # region STATIC

    @staticmethod
    def create(
            start: Optional[Union[MathNode, int]] = None,
            stop: Optional[Union[MathNode, int]] = None,
            step: Optional[Union[MathNode, int]] = None
    ) -> "SliceExpr":
        """
        Instantiate a :class:`SliceExpr` from Python or symbolic bounds.

        Parameters
        ----------
        start : MathExpr or int or None, optional
            Inclusive lower bound of the slice.
        stop : MathExpr or int or None, optional
            Exclusive upper bound of the slice.
        step : MathExpr or int or None, optional
            Stride applied between slice elements.

        Returns
        -------
        SliceExpr
            Newly constructed slice expression.
        """
        return SliceExpr(start=start, stop=stop, step=step)
    # end def create

    @staticmethod
    def from_slice(py_slice: slice) -> "SliceExpr":
        """
        Instantiate a :class:`SliceExpr` from a Python ``slice``.

        Parameters
        ----------
        py_slice : slice
            Python slice whose bounds should be mirrored.

        Returns
        -------
        SliceExpr
            Symbolic slice equivalent to ``py_slice``.
        """
        return SliceExpr(
            start=py_slice.start,
            stop=py_slice.stop,
            step=py_slice.step
        )
    # end def from_slice

    @staticmethod
    def _coerce_bound(
            name: str,
            value: Optional[Union[MathNode, int]]
    ) -> Optional[MathNode]:
        """
        Normalize Python and symbolic bounds into :class:`MathExpr` instances.

        Parameters
        ----------
        name : str
            Bound name for diagnostics.
        value : Optional[Union[MathNode, int]]
            User-provided bound.

        Returns
        -------
        Optional[MathNode]
            Normalized bound expression, or ``None``.

        Raises
        ------
        TypeError
            If ``value`` is not ``None``, ``int``, or :class:`MathExpr`.
        MathExprValidationError
            If ``value`` violates constant scalar integer requirements.
        """
        if value is None:
            return None
        # end if
        if isinstance(value, int):
            tensor = Tensor(data=value, dtype=DType.INT64)
            value = Constant(
                name=f"slice_{name}_{MathNode.next_id()}",
                data=tensor
            )
        # end if
        if not isinstance(value, MathNode):
            raise TypeError(f"{name} must be a MathExpr or int, got {type(value)}")
        # end if
        SliceExpr._validate_bound(name, value)
        return value
    # end def _coerce_bound

    @staticmethod
    def _validate_bound(name: str, expr: MathNode) -> None:
        """
        Validate that ``expr`` is a scalar integer constant expression.

        Parameters
        ----------
        name : str
            Bound identifier for error messages.
        expr : MathNode
            Expression to validate.

        Raises
        ------
        MathExprValidationError
            If ``expr`` is not constant, not scalar, or not integer typed.
        """
        if not expr.is_constant():
            raise MathExprValidationError(f"{name} must be composed of constants.")
        # end if
        if expr.dtype not in {DType.INT32, DType.INT64}:
            raise MathExprValidationError(f"{name} must be an integer expression, got {expr.dtype}")
        # end if
        if expr.shape.dims != ():
            raise MathExprValidationError(f"{name} must be scalar, got shape {expr.shape}")
        # end if
    # end def _validate_bound

    @staticmethod
    def _expr_to_python(expr: Optional[MathNode]) -> Optional[int]:
        """
        Convert a symbolic bound to its Python integer value.

        Parameters
        ----------
        expr : Optional[MathNode]
            Symbolic expression representing a slice bound.

        Returns
        -------
        Optional[int]
            Python integer bound or ``None`` when ``expr`` is ``None``.
        """
        if expr is None:
            return None
        # end if
        tensor = expr.eval()
        return int(tensor.value.item())
    # end def _expr_to_python

    # endregion STATIC

    # region OVERRIDE

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            Human-readable representation showing resolved Python bounds.
        """
        return f"SliceExpr(start={self.start_value}, stop={self.stop_value}, step={self.step_value})"
    # end def __repr__

    # endregion OVERRIDE

# end class SliceExpr
