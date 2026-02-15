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

# Imports
from __future__ import annotations
import weakref
from typing import Any, FrozenSet, List, Optional, Tuple, Union, Dict, Sequence, TYPE_CHECKING

from .math_base import MathBase
from .math_exceptions import (
    SymbolicMathOperatorError,
    SymbolicMathNotImplementedError
)
from .math_slice import SliceExpr
from .mixins import DifferentiableMixin, PredicateMixin
from .dtype import DType
from .shape import Shape
from .tensor import Tensor
from .typing import Index, MathExpr, Operands


if TYPE_CHECKING:
    from .math_leaves import Constant, Variable
# end if


__all__ = [
    "MathNode",
]


class MathNode(
    MathBase,
    DifferentiableMixin,
    PredicateMixin,
    MathExpr
):
    """
    Canonical symbolic expression node.

    Each node represents a typed tensor value produced by a symbolic
    computation.  Nodes are immutable once constructed: they expose their
    operator, ordered children, dtype, and shape as read-only metadata so that
    graph transformations can reason about the structure deterministically.
    """

    __slots__ = (
        "_op",
        "_children",
        "_dtype",
        "_shape",
        "_parents_weak",
    )

    # region CONSTRUCTOR

    def __init__(
            self,
            name: Optional[str],
            *,
            op: Optional,
            children: Operands,
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
        super(MathNode, self).__init__(name=name, dtype=dtype, shape=shape)
        self._op = op
        self._children: Operands = children
        self._parents_weak: weakref.WeakSet[MathExpr] = weakref.WeakSet()
        self._check_operator()
        self._register_as_parent_of(*children)
    # end __init__

    # endregion CONSTRUCTOR

    # region PROPERTIES

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
    def children(self) -> Tuple[MathExpr, ...]:
        """
        Returns
        -------
        tuple[MathExpr, ...]
            Child nodes (empty tuple for leaves).
        """
        return self._children
    # end def children

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
    def parents(self) -> FrozenSet[MathExpr]:
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

    @property
    def shape(self) -> Shape:
        """
        Returns
        -------
        'Shape'
            Shape of the advertised tensor.
        """
        return self._shape
    # end def shape

    # endregion PROPERTIES

    # region MATH_EXPR

    def eval(self) -> Tensor:
        """
        Evaluate this node in the active math context.

        Returns
        -------
        'Tensor'
            Result of executing ``self.op`` with evaluated children.
        """
        return self._op.eval(operands=self._children)
    # end def eval

    def diff(
            self,
             wrt: "Variable"
    ) -> MathExpr:
        """
        Compute the derivative of this expression with respect to ``wrt``.

        Parameters
        ----------
        wrt : "Variable"
            The variable to differentiate with respect to.

        Returns
        -------
        'MathExpr'
            The derivative of ``self`` with respect to ``wrt``.
        """
        return self._op.diff(wrt=wrt, operands=self._children)
    # end def diff

    def variables(self) -> Sequence['Variable']:
        """
        Enumerate variable leaves reachable from this node.

        Returns
        -------
        Sequence['Variable']
            List of :class:`Variable` instances (duplicates possible).
        """
        _vars: List = list()
        for c in self._children:
            _vars.extend(c.variables())
        # end for
        return _vars
    # end def variables

    def constants(self) -> Sequence['Constant']:
        """
        Enumerate constant leaves reachable from this node.

        Returns
        -------
        Sequence['Constant']
            List of :class:`Constant` instances (duplicates possible).
        """
        constants: List = list()
        for c in self._children:
            constants.extend(c.constants())
        # end for
        return constants
    # end def constants

    def contains(
            self,
            leaf: Union[str, MathExpr],
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
            c.contains(
                leaf,
                by_ref=by_ref,
                check_operator=check_operator,
                look_for=look_for
            )
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
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a variable `variable`"""
        return self.contains(variable, by_ref=by_ref, check_operator=check_operator, look_for="var")
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a constant `constant`"""
        return self.contains(constant, by_ref=by_ref, check_operator=check_operator, look_for="const")
    # end def contains_constant

    def renamed(self, old_name: str, new_name: str) -> Dict[str, str]:
        """
        Rename all variables/constants named ``old_name`` with ``new_name`` in the tree.
        The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        rename_dict = {}
        for child in self._children:
            rn_out = child.renamed(old_name, new_name)
            rename_dict.update(rn_out)
        # end for
        return rename_dict
    # end rename

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

    def is_node(self) -> bool:
        """
        Returns
        -------
        'bool'
            ``True`` when the expression has an operator (non-leaf).
        """
        return self._op is not None
    # end def is_node

    def is_leaf(self) -> bool:
        """
        Returns
        -------
        'bool'
            ``True`` when the expression has no children.
        """
        return len(self._children) == 0
    # end is_leaf

    def depth(self) -> int:
        """Return the depth of the node in the tree"""
        return max([c.depth() for c in self._children]) + 1
    # end def depth

    def copy(self, deep: bool = False):
        """
        TODO: define a stable copy contract for MathNode.

        Current implementation is intentionally conservative: deep copies of
        DAG nodes are not yet specified (shared children, parent weakrefs,
        operator-owned parameters).
        """
        raise SymbolicMathNotImplementedError(
            "TODO: implement MathNode.copy(deep=...) once graph copy semantics are finalized."
        )
    # end def copy

    def __str__(self) -> str:
        """
        TODO: provide a dedicated human-readable string format.
        """
        return self.__repr__()
    # end def __str__

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

    # endregion MATH_EXPR

    # region PUBLIC

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

    def is_higher_order(self):
        """
        Is the expression higher order?

        Returns:
            ``True`` when the expression is a leaf and its shape is (m,n,...)
        """
        return self._shape.rank > 2
    # end def is_higher_order

    def add_parent(self, parent: MathExpr):
        """
        Register ``parent`` as a consumer of this node.

        Parameters
        ----------
        parent : MathNode
            Expression that references ``self`` as an input.
        """
        self._parents_weak.add(parent)
    # end def parent

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
        if self._op is None:
            return
        # end if
        if not self._op.is_variadic:
            if self.arity != self._op.arity:
                raise SymbolicMathOperatorError(
                    f"Operator and arity mismatch: "
                    f"{self._op.name}({self._op.arity}) != {self.__class__.__name__}({self.arity})"
                )
            # end if
        else:
            self._op.arity = self.arity
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
            ch._parents_weak.add(self)  # type: ignore[attr-defined]
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
    def getitem(operand: MathNode, index: Index) -> MathNode:
        """
        Create a getitem node.
        """
        from .functional.structure import getitem
        indices: List[Union[int, SliceExpr]] = [
            SliceExpr.from_slice(s) if isinstance(s, slice) else s
            for s in index
        ]
        return getitem(op1=operand, indices=indices)
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
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
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
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
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
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
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
        raise SymbolicMathNotImplementedError("Ordering comparisons are not implemented for MathExpr.")
    # end __ge__

    # Get item
    def __getitem__(self, item: Index):
        return MathNode.getitem(self, item)
    # end def __getitem__

    # endregion OVERRIDE

# end class MathNode
