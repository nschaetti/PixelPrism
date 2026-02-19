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
from typing import Any, FrozenSet, List, Optional, Union, Sequence, TYPE_CHECKING, Mapping

from .math_base import MathBase
from .math_exceptions import (
    SymbolicMathOperatorError,
    SymbolicMathNotImplementedError
)
from .mixins import PredicateMixin, ExprOperatorsMixin
from .dtype import DType
from .shape import Shape
from .tensor import Tensor
from .typing import MathExpr, Operands, Operator, LeafKind, SimplifyOptions

if TYPE_CHECKING:
    from .math_leaves import Constant, Variable
# end if


__all__ = [
    "MathNode",
]


class MathNode(
    MathBase,
    PredicateMixin,
    ExprOperatorsMixin,
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
            op: Optional[Operator],
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
    def children(self) -> Sequence[MathExpr]:
        """
        Returns
        -------
        Sequence['MathExpr']
            Child nodes (empty tuple for leaves).
        """
        return self._children
    # end def children

    @property
    def parents(self) -> FrozenSet[MathExpr]:
        """
        Returns
        -------
        frozenset['MathNode']
            Best-effort view of nodes that consume this expression.
        """
        return frozenset(self._parents_weak)
    # end def parents

    @property
    def arity(self) -> int:
        """
        Returns
        -------
        'int'
            Number of children supplied to the node.
        """
        return len(self._children)
    # end def arity

    # endregion PROPERTIES

    # region MATH_EXPR

    #
    # Properties
    #

    @property
    def rank(self) -> int:
        """
        Returns
        -------
        'int'
            Rank of the advertised tensor.
        """
        return self._shape.rank
    # end def rank

    #
    # Evaluate and differentiate
    #

    def eval(self) -> Tensor:
        """
        Evaluate this node in the active math context.

        The node delegates evaluation to its operator after child expressions
        have been resolved to concrete tensors. Semantics follow NumPy-style
        operator conventions (broadcasting, dtype promotion, shape rules), so
        the result matches what the equivalent NumPy operation would produce.

        Returns
        -------
        'Tensor'
            Result of executing ``self.op`` with evaluated children.

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> x = pm.Variable(name="x", dtype=pm.R, shape=pm.S.scalar)
        >>> y = pm.Variable(name="y", dtype=pm.R, shape=pm.S.scalar)
        >>> expr = x + y
        >>> with pm.new_context() as scope:
        ...     scope.set("x", 2.0)
        ...     scope.set("y", 5.0)
        ...     out = expr.eval()
        >>> out.value.item()
        7.0
        >>> np.add(2.0, 5.0)
        7.0
        """
        return self._op.eval(operands=self._children)
    # end def eval

    def diff(
            self,
             wrt: 'Variable'
    ) -> MathExpr:
        """
        Compute the derivative of this expression with respect to ``wrt``.

        The derivative is returned as a symbolic expression tree (not a numeric
        tensor). The resulting expression can be evaluated later in a context
        where all required variables are bound. Operator-level differentiation
        follows standard calculus rules while preserving NumPy-style tensor
        semantics for shapes and dtypes.

        Parameters
        ----------
        wrt : "Variable"
            The variable to differentiate with respect to.

        Returns
        -------
        'MathExpr'
            The derivative of ``self`` with respect to ``wrt``.

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> x = pm.Variable(name="x", dtype=pm.R, shape=pm.S.scalar)
        >>> y = pm.Variable(name="y", dtype=pm.R, shape=pm.S.scalar)
        >>> expr = x * y
        >>> dexpr_dx = expr.diff(x)
        >>> with pm.new_context() as scope:
        ...     scope.set("x", 2.0)
        ...     scope.set("y", 5.0)
        ...     out = dexpr_dx.eval()
        >>> out.value.item()
        5.0
        >>> np.multiply(1.0, 5.0)
        5.0
        """
        return self._op.diff(wrt=wrt, operands=self._children)
    # end def diff

    #
    # Structure
    #

    def variables(self) -> Sequence['Variable']:
        """
        List variable leaves reachable from this node.

        Traversal follows the expression children order and concatenates each
        child's contribution. Returned items are symbolic variable leaves, not
        concrete arrays. The method is useful to inspect dependencies before
        evaluating an expression against NumPy-like runtime values.

        Returns
        -------
        Sequence['Variable']
            List of :class:`Variable` instances (duplicates possible).

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> x = pm.Variable(name="x", dtype=pm.R, shape=pm.S.scalar)
        >>> y = pm.Variable(name="y", dtype=pm.R, shape=pm.S.scalar)
        >>> expr = x + y
        >>> [my_var.name for my_var in expr.variables()]
        ['x', 'y']
        >>> with pm.new_context() as scope:
        ...     scope.set("x", np.array([1.0, 2.0]))
        ...     scope.set("y", np.array([3.0, 4.0]))
        ...     out = expr.eval()
        >>> out.value.tolist()
        [4.0, 6.0]
        >>> np.add(np.array([1.0, 2.0]), np.array([3.0, 4.0])).tolist()
        [4.0, 6.0]
        """
        _vars: List = list()
        for c in self._children:
            _vars.extend(c.variables())
        # end for
        return list(set(_vars))
    # end def variables

    def constants(self) -> Sequence['Constant']:
        """
        List constant leaves reachable from this node.

        Traversal follows the expression children order and concatenates each
        child's contribution. Returned items are symbolic constant leaves, not
        raw NumPy values. This method is useful to inspect literal dependencies
        embedded in an expression before evaluation.

        Returns
        -------
        Sequence['Constant']
            List of :class:`Constant` instances (duplicates possible).

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> x = pm.Variable(name="x", dtype=pm.R, shape=pm.S.scalar)
        >>> c = pm.Constant.create(name="c", data=pm.tensor([10.0, 20.0]))
        >>> expr = x + c
        >>> [k.name for k in expr.constants()]
        ['c']
        >>> with pm.new_context() as scope:
        ...     scope.set("x", np.array([1.0, 2.0]))
        ...     out = expr.eval()
        >>> out.value.tolist()
        [11.0, 22.0]
        >>> np.add(np.array([1.0, 2.0]), np.array([10.0, 20.0])).tolist()
        [11.0, 22.0]
        """
        constants: List = list()
        for c in self._children:
            constants.extend(c.constants())
        # end for
        return list(set(constants))
    # end def constants

    def contains(
            self,
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """
        Test whether ``var`` appears in the expression tree.

        Search traverses all child expressions and can optionally inspect
        operator-owned symbolic parameters when ``check_operator=True``. The
        lookup target may be provided either by name (string) or by expression
        instance. With ``by_ref=True``, matching is identity-based; otherwise it
        uses symbolic naming rules implemented by leaf/node types.

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
        look_for : LeafKind, default LeafKind.ANY
            Restrict lookup scope to variables, constants, or both.

        Returns
        -------
        bool
            ``True`` when ``var`` was located in ``self`` or any child.

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> from pixelprism.math import DType, Shape, LeafKind
        >>> x = pm.var(name="x", dtype=DType.R, shape=Shape((2,)))
        >>> y = pm.var(name="y", dtype=DType.R, shape=Shape((2,)))
        >>> c = pm.const(name="c", data=[1.0, 1.0])
        >>> expr = x + (y + c)
        >>> expr.contains("x")
        True
        >>> expr.contains("c", look_for=pm.CONSTANT)
        True
        >>> expr.contains("c", look_for=pm.VARIABLE)
        False
        >>> with pm.new_context() as scope:
        ...     scope.set("x", np.array([2.0, 3.0]))
        ...     scope.set("y", np.array([4.0, 5.0]))
        ...     out = expr.eval()
        >>> out.value.tolist()
        [7.0, 9.0]
        >>> np.add(np.array([2.0, 3.0]), np.add(np.array([4.0, 5.0]), np.array([1.0, 1.0]))).tolist()
        [7.0, 9.0]
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
        """Return ``True`` if the expression contains variable ``variable``.

        This is a convenience wrapper over :meth:`contains` with
        ``look_for=LeafKind.VARIABLE``.

        Parameters
        ----------
        variable : str or Variable
            Variable name or variable expression to locate.
        by_ref : bool, default False
            When ``True``, require object identity match.
        check_operator : bool, default True
            When ``True``, include operator-owned symbolic parameters.

        Returns
        -------
        bool
            ``True`` if ``variable`` is present in this expression.

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> from pixelprism.math import DType, Shape
        >>> x = pm.Variable(name="x", dtype=DType.R, shape=pm.S.vector(2))
        >>> y = pm.Variable(name="y", dtype=DType.R, shape=pm.S.vector(2))
        >>> c = pm.Constant.create(name="c", data=pm.tensor([1.0, 1.0]))
        >>> expr = x + (y + c)
        >>> expr.contains_variable("x")
        True
        >>> expr.contains_variable("c")
        False
        >>> with pm.new_context() as scope:
        ...     scope.set("x", np.array([2.0, 3.0]))
        ...     scope.set("y", np.array([4.0, 5.0]))
        ...     out = expr.eval()
        >>> out.value.tolist()
        [7.0, 9.0]
        >>> np.add(np.array([2.0, 3.0]), np.add(np.array([4.0, 5.0]), np.array([1.0, 1.0]))).tolist()
        [7.0, 9.0]
        """
        return self.contains(
            leaf=variable,
            by_ref=by_ref,
            check_operator=check_operator,
            look_for=LeafKind.VARIABLE
        )
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return ``True`` if the expression contains constant ``constant``.

        This is a convenience wrapper over :meth:`contains` with
        ``look_for=LeafKind.CONSTANT``.

        Parameters
        ----------
        constant : str or Constant
            Constant name or constant expression to locate.
        by_ref : bool, default False
            When ``True``, require object identity match.
        check_operator : bool, default True
            When ``True``, include operator-owned symbolic parameters.

        Returns
        -------
        bool
            ``True`` if ``constant`` is present in this expression.

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> from pixelprism.math import DType, Shape
        >>> x = pm.Variable(name="x", dtype=DType.R, shape=pm.S.vector(2))
        >>> c = pm.Constant.create(name="c", data=pm.tensor([10.0, 20.0]))
        >>> expr = x + c
        >>> expr.contains_constant("c")
        True
        >>> expr.contains_constant("x")
        False
        >>> with pm.new_context() as scope:
        ...     scope.set("x", np.array([1.0, 2.0]))
        ...     out = expr.eval()
        >>> out.value.tolist()
        [11.0, 22.0]
        >>> np.add(np.array([1.0, 2.0]), np.array([10.0, 20.0])).tolist()
        [11.0, 22.0]
        """
        return self.contains(
            leaf=constant,
            by_ref=by_ref,
            check_operator=check_operator,
            look_for=LeafKind.CONSTANT
        )
    # end def contains_constant

    #
    # Immutable transforms
    #

    def simplify(self, options: SimplifyOptions | None = None) -> MathExpr:
        """
        Apply symbolic rewrite rules and return a simplified expression.
        This operation is pure: it never mutates the current tree.

        Intended behavior (implementation contract):
        1) Recursively simplify children first.
        2) Delegate operator-local rewrites to ``self.op.simplify(...)``.
        3) If the operator returns a full-node ``replacement``, return it.
        4) Otherwise rebuild a node with simplified operands when needed.
        5) Preserve semantic equivalence with NumPy-style evaluation.

        Simplification should honor :class:`SimplifyOptions`:
        - ``enabled=None`` means all rules are enabled by default.
        - ``disabled`` rules always take precedence.
        - No in-place mutation of existing nodes/subtrees.

        Parameters
        ----------
        options : SimplifyOptions or None, default None
            Rule selection for this simplify pass.

        Returns
        -------
        MathExpr
            A semantically equivalent, potentially reduced expression.

        Notes
        -----
        Typical target rewrites include:
        - ``x + 0 -> x``
        - ``x - 0 -> x``
        - ``x * 1 -> x``
        - ``x * 0 -> 0``
        - constant folding for constant-only subtrees

        The rewritten tree should evaluate to the same tensor as the original
        expression for any valid context bindings, following NumPy broadcasting
        and dtype semantics.

        Examples
        --------
        >>> import numpy as np
        >>> import pixelprism.math as pm
        >>> from pixelprism.math import DType, Shape, Tensor, SimplifyOptions, SimplifyRule
        >>> x = pm.Variable(name="x", dtype=DType.R, shape=pm.S.vector(2))
        >>> z = pm.Constant(name="z", data=pm.tensor([0.0, 0.0], dtype=DType.R))
        >>> expr = x + z
        >>> simplified = expr.simplify()  # doctest: +SKIP
        >>> str(simplified)  # doctest: +SKIP
        'x'

        >>> with pm.new_context() as scope:  # doctest: +SKIP
        ...     scope.set("x", np.array([3.0, 4.0]))
        ...     expr.eval().value.tolist() == np.add(np.array([3.0, 4.0]), np.array([0.0, 0.0])).tolist()
        True

        >>> opts = SimplifyOptions(disabled=frozenset({SimplifyRule.ADD_ZERO}))
        >>> kept = expr.simplify(options=opts)  # doctest: +SKIP
        >>> str(kept)  # doctest: +SKIP
        '(x + z)'
        """
        # Simplify children first
        children = [child.simplify(options=options) for child in self._children]

        # Apply operator rules
        simplify_result = self._op.simplify(operands=children, options=options)

        if simplify_result.replacement is not None:
            return simplify_result.replacement
        # end if

        self._children = simplify_result.operands

        return self
    # end def simplify

    def canonicalize(self) -> "MathExpr":
        """
        Normalize an expression form without changing semantics.
        Typical effects: associative flattening, deterministic operand ordering, etc.

        Returns
        -------
        'MathExpr'
            Normalized expression.
        """
        pass
    # end def canonicalize

    def fold_constants(self) -> "MathExpr":
        """
        Fold constant-only subexpressions into constant leaves.
        This is a focused transform and may be used independently of full simplifying.

        Returns
        -------
        'MathExpr'
            Expression with constant folding applied.
        """
        # TODO: to implement
        pass
    # end def fold_constants

    # Replace matching subexpressions using `mapping` and return a new tree.
    # - `by_ref=True`: match by object identity.
    # - `by_ref=False`: match by symbolic/tree equality policy.
    def substitute(
            self,
            mapping: Mapping[MathExpr, MathExpr],
            *,
            by_ref: bool = True
    ) -> MathExpr:
        """
        Replace matching subexpressions using `mapping` and return a new tree.
        - `by_ref=True`: match by object identity.
        - `by_ref=False`: match by symbolic/tree equality policy.

        Args:
            mapping:
            by_ref:

        Returns:

        """
        # TODO: to implement
        pass
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
        """
        Rename all variables/constants named ``old_name`` with ``new_name`` in the tree.
        The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.

        Returns
        -------
        MathExpr
            Expression with renamed variables/constants.
        """
        # rename_dict = {}
        # for child in self._children:
        #     rn_out = child.renamed(old_name, new_name)
        #     rename_dict.update(rn_out)
        # # end for
        # return rename_dict
        # TODO: to implement
        pass
    # end rename

    #
    # Comparison
    #

    # Strict symbolic tree equality.
    # Returns True only if both expressions have the same structure
    # (same node kinds/operators, same operand order, same leaf content).
    def eq_tree(self, other: "MathExpr") -> bool:
        # TODO: to implement
        pass
    # end def eq_tree

    # Mathematical symbolic equivalence.
    # Returns True when both expressions represent the same symbolic meaning,
    # even if their trees differ (e.g., after canonicalization/simplification).
    # This check is symbolic only (no numeric/probabilistic fallback).
    def equivalent(self, other: "MathExpr") -> bool:
        # TODO: to implement
        pass
    # end def equivalent

    #
    # Boolean predicates
    #

    def is_constant(self) -> bool:
        """Does the expression contain only constant values?"""
        rets = [o.is_constant() for o in self._children]
        return all(rets)
    # end def is_constant

    def is_variable(self) -> bool:
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

    #
    # Structure
    #

    def depth(self) -> int:
        """
        Return the maximum depth of the subtree rooted at this node.
        Convention: leaves have depth 1.

        Returns
        -------
        'int'
            Maximum depth of the subtree rooted at this node.
        """
        return max([c.depth() for c in self._children]) + 1
    # end def depth

    def copy(self, deep: bool = False) -> MathExpr:
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

    #
    # Representation
    #

    def __str__(self) -> str:
        """
        Human-readable representation intended for users/logs.
        TODO: provide a dedicated human-readable string format.
        """
        return self.__repr__()
    # end def __str__

    def __repr__(self) -> str:
        """
        Developer-oriented representation intended for debugging.
        Should be unambiguous and include key structural identifiers.

        Returns
        -------
        'str'
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
        list['MathNode']
            Combined list of variables and constants.
        """
        return list(self.variables()) + list(self.constants())
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
        if not self._op.IS_VARIADIC:
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

# end class MathNode
