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
from lib2to3.pytree import LeafPattern
from typing import Any, FrozenSet, List, Optional, Union, Sequence, TYPE_CHECKING, Mapping, Dict, Tuple

from .math_base import MathBase
from .math_exceptions import (
    SymbolicMathOperatorError,
    SymbolicMathNotImplementedError
)
from .mixins import ExpressionMixin, AlgebraicMixin
from .dtype import DType
from .shape import Shape
from .tensor import Tensor
from .typing import (
    ExprPattern,
    NodePattern,
    VariablePattern,
    ConstantPattern,
    AnyPattern,
    MathExpr,
    Operands,
    Operator,
    LeafKind,
    SimplifyOptions,
    ExprKind,
    ExprDomain,
    OperatorSpec,
    AlgebraicExpr
)
from .typing_expr import FoldPolicy, OpSimplifyResult, MatchResult, EllipsisPattern

if TYPE_CHECKING:
    from .math_leaves import Constant, Variable
# end if


__all__ = [
    "MathNode",
]


class MathNode(
    MathBase,
    ExpressionMixin,
    AlgebraicMixin,
    AlgebraicExpr
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
        super(MathNode, self).__init__(
            name=name,
            kind=ExprKind.NODE,
            domain=ExprDomain.ALGEBRAIC,
            dtype=dtype,
            shape=shape
        )
        self._op = op
        self._children: Operands = children
        self._parents_weak: weakref.WeakSet[MathExpr] = weakref.WeakSet()
        self._check_operator()
        self._register_as_parent_of(*children)
    # end __init__

    # endregion CONSTRUCTOR

    # region MATH_EXPR

    #
    # Properties
    #

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

    @property
    def op(self) -> Optional[Operator]:
        """
        Returns
        -------
        Operator or None
            Operator descriptor for non-leaf nodes.
        """
        return self._op
    # end op

    @property
    def op_name(self) -> Optional[str]:
        """
        Returns
        -------
        str or None
            Name of the operator for non-leaf nodes.
        """
        return self._op.name if self._op is not None else None
    # end def op_name

    @property
    def spec(self) -> Optional[OperatorSpec]:
        """
        Returns
        -------
        OperatorSpec or None
            Operator specification for non-leaf nodes.
        """
        return self._op.spec if self._op is not None else None
    # end def spec

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

    def _check_operator_min_operands(self):
        """Check that the number of operands is still ok for the operator"""
        if len(self._children) < self._op.arity.min_operands:
            raise RuntimeError(
                f"Operator '{self._op.name}' requires at least {self._op.arity.min_operands} operands, "
                f"got {len(self._children)}"
            )
        # end if
    # end if

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
        simplify_result: OpSimplifyResult = self._op.simplify(operands=children, options=options)

        # If remplacement available, return it
        if simplify_result.replacement is not None:
            return simplify_result.replacement
        # end if

        # Otherwise rebuild a new node with simplified operands
        self._children = simplify_result.operands

        # Check that the number of operands is still ok for the operator
        self._check_operator_min_operands()

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
        # Canonicalize children first
        children = [child.canonicalize() for child in self._children]

        # Ask the operator to canonicalize its operands
        canon_result: OpSimplifyResult = self._op.canonicalize(operands=children)

        # if remplacement available, return it
        if canon_result.replacement is not None:
            return canon_result.replacement
        # end if

        # Otherwise rebuild a new node with canonicalized operands
        self._children = canon_result.operands

        # Check that the number of operands is still ok for the operator
        self._check_operator_min_operands()

        return self
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
        # if this node is foldable
        if not self.is_foldable():
            return self
        # end if

        # Ask the operator to fold constants
        fold_result: OpSimplifyResult = self._op.fold_constants(operands=self._children)

        # if replacement available, return it
        if fold_result.replacement is not None:
            return fold_result.replacement
        # end if

        # Otherwise rebuild a new node with folded operands
        self._children = fold_result.operands

        # Check that the number of operands is still ok for the operator
        self._check_operator_min_operands()

        return self
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
        if len(mapping) == 0:
            return self
        # end if

        # Check if we are in the substitution scope
        if self in mapping:
            return mapping[self]
        # end if

        # Substitute children
        new_children = list()
        for child in self._children:
            new_children.append(child.substitute(mapping, by_ref=by_ref))
        # end for
        self._children = new_children
        return self
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> List[str]:
        """
        Rename all variables/constants/node named ``old_name`` with ``new_name`` in the tree.
        The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant/node to rename.
        new_name: str
            New name for the variable/constant/node.

        Returns
        -------
        List[str]
            List of renamed variables/constants.
        """
        if self.name == old_name:
            self._name = new_name
        # end if
        rename_list = list()
        for child in self.children:
            c_renamed = child.renamed(old_name, new_name)
            rename_list.extend(c_renamed)
        # end for
        return rename_list
    # end rename

    #
    # Comparison
    #

    # Strict symbolic tree equality.
    # Returns True only if both expressions have the same structure
    # (same node kinds/operators, same operand order, same leaf content).
    def eq_tree(self, other: "MathExpr") -> bool:
        """
        Strict symbolic tree equality.

        Parameters
        ----------
        other : MathExpr
            The other expression to compare with.

        Returns
        -------
        bool
            True if the trees are equal, False otherwise.
        """
        if not isinstance(other, MathNode):
            return False
        # end if
        return (
            self.__class__ == other.__class__ and
            self._dtype == other.dtype and
            self._shape == other.shape and
            self._kind == other.kind and
            self._domain == other.domain and
            len(self._children) == len(other.children) and
            all(c1.eq_tree(c2) for c1, c2 in zip(self.children, other.children)) and
            self._op == other.op
        )
    # end def eq_tree

    # Mathematical symbolic equivalence.
    # Returns True when both expressions represent the same symbolic meaning,
    # even if their trees differ (e.g., after canonicalization/simplification).
    # This check is symbolic only (no numeric/probabilistic fallback).
    def equivalent(self, other: "MathExpr") -> bool:
        """Return True if both expressions are equivalent."""
        # Copy both expressions
        a = self.copy(deep=True)
        b = other.copy(deep=True)

        # Simplify both expressions
        a = a.simplify()
        b = b.simplify()

        # Check if the simplified trees are equal
        return a.eq_tree(b)
    # end def equivalent

    def _match_operator(self, op: Operator, match_op: Optional[str] = None) -> bool:
        return match_op is None or op.name == match_op
    # end def _match_operator

    def _match_error_operands_length(self, len_operands: int, len_match_operands: int):
        raise RuntimeError(f"Expected {len_operands} operands, got {len_match_operands}")
    # end if

    def _pattern_contains_ellipsis(self, patterns: List[ExprPattern]) -> bool:
        """Check if any pattern contains Ellipsis or EllipsisPattern"""
        pattern_type = [op is Ellipsis or isinstance(op, EllipsisPattern) for op in patterns]
        return any(pattern_type)
    # end def _pattern_contains_ellipsis

    def _match_operands(
            self,
            operands: List[MathExpr],
            match_operands: Optional[List[ExprPattern]] = None
    ) -> MatchResult:
        has_ellipsis = self._pattern_contains_ellipsis(match_operands)
        if not match_operands:
            return MatchResult.success({})
        elif len(operands) != len(match_operands) and not has_ellipsis:
            self._match_error_operands_length(len(operands), len(match_operands))
        # end if

        first_m_op = match_operands[0]
        last_m_op = match_operands[-1]
        additional_expr = {}

        if not has_ellipsis:
            match_results = [
                op.match(op_pattern) for op, op_pattern in zip(operands, match_operands)
            ]
        # x, y, ...
        elif last_m_op is Ellipsis or isinstance(last_m_op, EllipsisPattern):
            match_results = list()
            matched_operands = list()
            for op, op_pattern in zip(operands, match_operands[:-1]):
                m = op.match(op_pattern)
                match_results.append(m)
                if m.matched:
                    matched_operands.append(op)
                # end if
            # end for
            if isinstance(last_m_op, EllipsisPattern) and last_m_op.name is not None:
                additional_expr = {last_m_op.name: [op for op in operands if op not in matched_operands]}
            # end if
        elif first_m_op is Ellipsis or isinstance(first_m_op, EllipsisPattern):
            match_results = list()
            matched_operands = list()
            for op, op_pattern in zip(operands[::-1], match_operands[::-1][:-1]):
                m = op.match(op_pattern)
                match_results.append(m)
                if m.matched:
                    matched_operands.append(op)
                # end if
            # end for
            if isinstance(first_m_op, EllipsisPattern) and first_m_op.name is not None:
                additional_expr = {first_m_op.name: [op for op in operands if op not in matched_operands]}
            # end if
        else:
            raise RuntimeError(f"Ellipsis must be at the end or at the beginning: {match_operands}")
        # end if

        match_bool = [m.matched for m in match_results]
        match_expr = [m.bindings for m in match_results]

        match_dict = {k:v for d in match_expr for k,v in d.items()}
        match_dict.update(additional_expr)

        if all(match_bool):
            return MatchResult.success(match_dict)
        else:
            return MatchResult.failed()
        # end if
    # end def _match_operands

    def _get_ellipsis(self, match_operands: List[ExprPattern]):
        """Get the ellipsis pattern if it exists in the match_operands"""
        for op_pattern in match_operands:
            if op_pattern is Ellipsis or isinstance(op_pattern, EllipsisPattern):
                return op_pattern
            # end if
        # end for
        return None
    # end def _get_ellipsis

    def _match_operands_comm(
            self,
            operands: List[MathExpr],
            match_operands: Optional[List[ExprPattern]] = None
    ) -> MatchResult:
        has_ellipsis = self._pattern_contains_ellipsis(match_operands)
        if not match_operands:
            return MatchResult.success({})
        elif len(operands) != len(match_operands) and not has_ellipsis:
            self._match_error_operands_length(len(operands), len(match_operands))
        # end if

        op_copy = list(match_operands)
        op_copy = [op for op in op_copy if not op is Ellipsis and not isinstance(op, EllipsisPattern)]
        matched_exprs = {}
        matched_operands = []
        for op in operands:
            match_found = False
            for op_pattern in op_copy:
                op_match_result = op.match(op_pattern)
                if op_match_result.matched:
                    match_found = True
                    op_copy.remove(op_pattern)
                    matched_exprs.update(op_match_result.bindings)
                    matched_operands.append(op)
                    break
                # end if
            # end for
            if has_ellipsis and len(op_copy) == 0:
                break
            # end if
        # end for

        if not has_ellipsis and len(op_copy) > 0:
            return MatchResult.failed()
        # end if

        # No ... -> we need all operands to be matched
        additional_expr = {}
        if len(matched_operands) != len(operands) and not has_ellipsis:
            return MatchResult.failed()
        elif has_ellipsis:
            ellipsis_pattern = self._get_ellipsis(match_operands)
            if isinstance(ellipsis_pattern, EllipsisPattern) and ellipsis_pattern.name is not None:
                additional_expr = {ellipsis_pattern.name: [op for op in operands if op not in matched_operands]}
            # end if
        # end if

        matched_exprs.update(additional_expr)

        return MatchResult.success(matched_exprs)
    # end def _match_operands_comm

    def _match_arity(self, arity: int, match_arity: Optional[int] = None) -> bool:
        return match_arity is None or arity == match_arity
    # end def _match_arity

    # Match a symbolic expression to a pattern.
    # - `pattern` is a string or a regular expression.
    def match(
            self,
            pattern: ExprPattern
    ) -> MatchResult:
        """
        Match a symbolic expression to a pattern.

        Parameters
        ----------
        pattern : ExprPattern
            The pattern to match against.

        Returns
        -------
        MatchResult
            The result of the match.
        """
        # Any, but check shape and dtype
        any_match = self._match_any_pattern(pattern)
        if any_match:
            return any_match
        # end if

        # Node, check kind and operator
        if isinstance(pattern, NodePattern):
            # Operator
            if not self._match_operator(self.op, pattern.op):
                return MatchResult.failed()
            # end if
            # Operands
            if pattern.commutative:
                operands_match = self._match_operands_comm(list(self.children), pattern.operands)
                if not operands_match.matched:
                    return MatchResult.failed()
                # end if
            else:
                operands_match = self._match_operands(list(self.children), pattern.operands)
                if not operands_match.matched:
                    return MatchResult.failed()
                # end if
            # end if
            if pattern.variable is not None and pattern.variable and not self.is_variable():
                return MatchResult.failed()
            # end if
            if pattern.constant is not None and pattern.constant and not self.is_constant():
                return MatchResult.failed()
            # end if
            if not self._match_shape(self.shape, pattern.shape):
                return MatchResult.failed()
            # end if
            if not self._match_dtype(self.dtype, pattern.dtype):
                return MatchResult.failed()
            # end if
            if not self._match_arity(len(self.children), pattern.arity):
                return MatchResult.failed()
            # end if
            return MatchResult.success({**self._match_return(pattern), **operands_match.bindings})
        # end if
        return MatchResult.failed()
    # end def match

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

    def is_constant_leaf(self) -> bool:
        """
        Returns
        -------
        'bool'
            ``True`` when the expression is a leaf node and represents a constant value.
        """
        return False
    # end def is_constant_leaf

    def is_variable_leaf(self) -> bool:
        """
        Returns
        -------
        'bool'
            ``True`` when the expression is a leaf node and represents a variable.
        """
        return False
    # end def is_variable_leaf

    def is_defined_constant(self) -> bool:
        return False
    # end def is_defined_constant

    def is_symbolic_constant(self) -> bool:
        return False
    # end def is_symbolic_constant

    def is_pure(self) -> bool:
        """
        Check if the expression is pure, i.e. has no side effects.
        """
        return all(child.is_pure() for child in self._children)
    # end def is_pure

    # True if behave like a scalar (rank = 1)
    def is_scalar(self) -> bool:
        """
        Check if the expression behaves like a scalar (rank = 1).
        """
        return self.rank == 0
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

    def is_tensor(self) -> bool:
        """
        Is the expression a tensor?

        Returns:
            ``True`` when the expression is a leaf and its shape is (m,n,...)
        """
        return self._shape.rank > 2
    # end def is_tensor

    def has_operator(self, name: str) -> bool:
        """
        Check if the expression as an operator with the given name.

        Parameters
        ----------
        name : 'str'
            Name of the operator to check for.

        Returns
        -------
        'bool'
            ``True`` when the expression has an operator with the given name.
        """
        return self._op is not None and self._op.name == name
    # end if

    # Check if the expression has children.
    def has_children(self) -> bool:
        """
        Check if the expression has children.
        """
        if self._children is None:
            return False
        # end if
        return len(self._children) > 0
    # end def has_children

    # Check the number of children of the expression.
    def num_children(self) -> int:
        """
        Check the number of children of the expression.
        """
        return len(self._children) if self._children is not None else 0
    # end def num_children

    #
    # Rules and policy
    #

    def is_foldable(self) -> bool:
        """
        Check if the expression can be folded into a single node.

        Returns
        -------
        'bool'
            ``True`` when the expression can be folded.
        """
        return all(child.is_foldable() for child in self._children)
    # end def is_foldable

    def fold_policy(self) -> "FoldPolicy":
        """
        Get the fold policy for this expression.

        Returns
        -------
        'FoldPolicy'
            The fold policy for this expression.
        """
        return FoldPolicy.FOLDABLE
    # end def fold_policy

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
        return max([c.depth() for c in self.children]) + 1
    # end def depth

    def _copy_deep(self) -> MathExpr:
        """
        Create a deep copy of the expression tree.
        """
        new_expr = MathNode(
            name=self._name,
            op=self._op.copy(deep=True),
            children=[c.copy(deep=True) for c in self.children],
            dtype=self._dtype.copy(),
            shape=self._shape.copy(),
        )
        return new_expr
    # end def _copy_deep

    def _copy_shallow(self) -> MathExpr:
        """Create a shallow copy of the expression tree."""
        new_expr = MathNode(
            name=self._name,
            op=self._op.copy(),
            children=self._children,
            dtype=self._dtype.copy(),
            shape=self._shape.copy(),
        )
        return new_expr
    # end def _copy_shallow

    def copy(self, deep: bool = False) -> MathExpr:
        """
        Create a deep or shallow copy of the expression tree.

        Parameters
        ----------
        deep : bool, optional
            If True, create a deep copy of the tree (default is False).

        Returns
        -------
        MathExpr
            A copy of the expression tree.
        """
        if deep:
            return self._copy_deep()
        else:
            return self._copy_shallow()
        # end if
    # end def copy

    #
    # Comparison
    #

    def __eq__(self, other: object) -> bool:
        return self is other
    # end def __eq__

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    # end def __ne__

    __hash__ = MathBase.__hash__

    #
    # Representation
    #

    def __str__(self) -> str:
        """
        Human-readable representation intended for users/logs.
        """
        return self._op.print(self._children)
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
        if not self._op.arity.variadic:
            if self.arity != self._op.spec.arity.exact:
                raise SymbolicMathOperatorError(
                    f"Operator and arity mismatch: "
                    f"{self._op.name}({self._op.spec.arity.exact}) != {self.__class__.__name__}({self.arity})"
                )
            # end if
        else:
            if self.arity < self._op.spec.arity.min_operands:
                raise SymbolicMathOperatorError(
                    f"Operator and arity mismatch: "
                    f"{self._op.name}({self._op.spec.arity.min_operands}, ...) != {self.__class__.__name__}({self.arity})"
                )
            # end if
        # end if
        self._op.set_parent(self)
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
