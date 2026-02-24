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
from abc import ABC, abstractmethod
from typing import Union, Dict, List, TYPE_CHECKING, Mapping, cast, Tuple, Callable, Optional
import numpy as np

from .typing_expr import LeafKind, MathExpr, ExprLike, Index, OpSimplifyResult
from .typing_rules import SimplifyRule, SimplifyOptions, SimplifyRuleType, RuleSpec

if TYPE_CHECKING:
    from .tensor import Tensor
    from .math_leaves import Variable, Constant
    from .math_node import MathNode
# end if


__all__ = [
    "ExpressionMixin",
    "AlgebraicMixin",
    "SimplifyRuleMixin",
]


class ExpressionMixin(ABC):
    """
    TODO: Add documentation.
    """

    #
    # Evaluate and differentiate
    #

    # Evaluate the expression in the active runtime context and return a tensor.
    # The returned value must match the expression contract (`dtype`, `shape`).
    @abstractmethod
    def eval(self) -> "Tensor":
        """Evaluate the expression in the active runtime context."""
    # end def eval

    # Symbolic derivative of the expression with respect to one variable.
    # Returns a new expression tree (not a numeric value), suitable for further rewrites.
    @abstractmethod
    def diff(self, wrt: "Variable") -> "MathExpr":
        """Symbolic derivative of the expression with respect to one variable."""
    # end def diff

    def equals(
            self: MathExpr,
            other: MathExpr | Tuple | List,
            *,
            rtol: float = 1e-6,
            atol: float = 1e-9,
            equal_nan: bool = False,
            require_same_shape: bool = True,
            require_same_dtype: bool = False
    ) -> bool:
        """
        Compare the numerical values of two expressions in a given context.

        Notes
        -----
        - This method compares a numerical evaluation (not symbolic equivalence).
        - `bindings` provides the values of variables used by the two expressions.
        """
        from . import as_expr

        # To math expression
        other_expr = as_expr(other)

        # Evaluate
        left_val = self.eval()
        right_val = other_expr.eval()

        # To numpy array
        left_arr = np.asarray(left_val)
        right_arr = np.asarray(right_val)

        if require_same_shape and left_arr.shape != right_arr.shape:
            return False
        # end if

        if require_same_dtype:
            if left_val.dtype != right_val.dtype:
                return False
            # end if
        # end if

        try:
            return bool(np.allclose(left_arr, right_arr, rtol=rtol, atol=atol, equal_nan=equal_nan))
        except ValueError:
            # Shapes non broadcastables (si require_same_shape=False)
            return False
        # end try
    # end def equals

    #
    # Structure
    #

    @abstractmethod
    def variables(self) -> List['Variable']:
        """
        List variable leaves reachable from this node.

        Returns
        -------
        list['MathNode']
            List of :class:`Variable` instances (duplicates possible).
        """
    # end def variables

    @abstractmethod
    def constants(self) -> List['Constant']:
        """
        List constant leaves reachable from this node.

        Returns
        -------
        list['MathNode']
            List of :class:`Constant` instances (duplicates possible).
        """
    # end def constants

    @abstractmethod
    def contains(
            self,
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
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
        look_for : LeafKind, default LeafKind.ANY
            Restrict lookup to variables/constants, or keep full lookup.

        Returns
        -------
        bool
            ``True`` when ``var`` was located in ``self`` or any child.
        """
    # end def contains

    @abstractmethod
    def contains_variable(
            self,
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """
        Return True if the expression contains a variable `variable`
        """

    # end def contains_variable

    @abstractmethod
    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """Return True if the expression contains a constant `constant`"""

    # end def contains_constant

    #
    # Immutable transforms
    #

    @abstractmethod
    def simplify(self, options: SimplifyOptions | None = None) -> "MathExpr":
        """Return a simplified version of the expression."""
    # end def simplify

    @abstractmethod
    def canonicalize(self) -> "MathExpr":
        """Return a canonicalized version of the expression."""
    # end def canonicalize

    @abstractmethod
    def fold_constants(self) -> "MathExpr":
        """Return a version of the expression with constants folded."""
    # end def fold_constants

    def substitute(
            self,
            mapping: Mapping[MathExpr, MathExpr],
            *,
            by_ref: bool = True
    ) -> MathExpr:
        """Return an expression with substitutions applied.

        Parameters
        ----------
        mapping : Mapping[MathExpr, MathExpr]
            Mapping from old subexpressions to replacement expressions.
        by_ref : bool, default True
            ``True`` for identity matching, ``False`` for symbolic matching.
        """
        if type(self).replace is not ExpressionMixin.replace:
            if not by_ref:
                raise NotImplementedError(
                    "Legacy replace(...) bridge supports only by_ref=True."
                )
            # end if
            for old_m, new_m in mapping.items():
                self.replace(old_m, new_m)
            # end for
            return cast(MathExpr, cast(object, self))
        # end if

        raise NotImplementedError("substitute(...) must be implemented by subclasses.")
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
        """Return an expression where ``old_name`` is replaced by ``new_name``.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        if type(self).rename is not ExpressionMixin.rename:
            self.rename(old_name, new_name)
            return cast(MathExpr, cast(object, self))
        # end if

        raise NotImplementedError("renamed(...) must be implemented by subclasses.")
    # end def renamed

    def replace(self, old_m: MathExpr, new_m: MathExpr):
        """Compatibility shim for mutable replacement APIs."""
        return self.substitute({old_m: new_m}, by_ref=True)
    # end def replace

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
        """Compatibility shim for mutable rename APIs."""
        self.renamed(old_name, new_name)
        return {old_name: new_name}
    # end def rename

    #
    # Boolean check
    #

    @abstractmethod
    def is_constant(self) -> bool:
        """Does the expression contain only constant values?"""
    # end def is_constant

    @abstractmethod
    def is_variable(self) -> bool:
        """Does the expression contain a variable?"""
    # end def is_variable

    @abstractmethod
    # True for operator-based internal tree nodes.
    def is_node(self) -> bool:
        """Is the expression an operator-based internal tree node?"""
    # end def is_node

    @abstractmethod
    # True for terminal tree elements (variables/constants) with no children.
    def is_leaf(self) -> bool:
        """Is the expression a terminal tree element (variables/constants) with no children?"""
    # end def is_leaf

    @abstractmethod
    # True if leaf and constant
    def is_constant_leaf(self) -> bool:
        """Is the expression a terminal tree element (variables/constants) with no children and constant value?"""
    # end def is_constant_leaf

    @abstractmethod
    # True if leaf and variable
    def is_variable_leaf(self) -> bool:
        """Is the expression a terminal tree element (variables/constants) with no children and variable name?"""
    # end def is_variable_leaf

    @abstractmethod
    # True if the expression is deterministically evaluable (no randomness or context side effects).
    def is_pure(self) -> bool:
        """Is the expression deterministically evaluable (no randomness or context side effects)?"""
    # end def is_pure

    @abstractmethod
    # True if behave like a scalar (rank = 1)
    def is_scalar(self) -> bool:
        """Is the expression behave like a scalar (rank = 1)?"""
    # end def is_scalar

    @abstractmethod
    # Check if the expression as an operator with the given name.
    def has_operator(self, name: str) -> bool:
        """Check if the expression as an operator with the given name."""
    # end def has_operator

    @abstractmethod
    # Check if the expression has children.
    def has_children(self) -> bool:
        """Check if the expression has children."""
    # end def has_children

# end class PredicateMixin


class AlgebraicMixin:
    """
    Shared arithmetic operator overloads for symbolic expressions.
    """

    # region OPERATORS

    @staticmethod
    def add(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Create an elementwise addition node.

        Parameters
        ----------
        operand1 : ExprLike
            Left operand.
        operand2 : ExprLike
            Right operand.

        Returns
        -------
        MathExpr
            Symbolic addition of ``operand1`` and ``operand2``.
        """
        from .functional.elementwise import add
        return add(operand1, operand2)
    # end def add

    @staticmethod
    def sub(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Create an elementwise subtraction node.

        Parameters
        ----------
        operand1 : ExprLike
            Left operand.
        operand2 : ExprLike
            Right operand to subtract.

        Returns
        -------
        MathExpr
            Symbolic subtraction ``operand1 - operand2``.
        """
        from .functional.elementwise import sub
        return sub(operand1, operand2)
    # end def sub

    @staticmethod
    def mul(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Create an elementwise multiplication node.

        Parameters
        ----------
        operand1 : ExprLike
            Left operand.
        operand2 : ExprLike
            Right operand.

        Returns
        -------
        MathExpr
            Symbolic product of the operands.
        """
        from .functional.elementwise import mul
        return mul(operand1, operand2)
    # end def mul

    @staticmethod
    def div(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Create an elementwise division node.

        Parameters
        ----------
        operand1 : ExprLike
            Numerator expression.
        operand2 : ExprLike
            Denominator expression.

        Returns
        -------
        MathExpr
            Symbolic quotient ``operand1 / operand2``.
        """
        from .functional.elementwise import div
        return div(operand1, operand2)
    # end def div

    @staticmethod
    def neg(operand: ExprLike) -> MathExpr:
        """
        Create an elementwise negation node.

        Parameters
        ----------
        operand : ExprLike
            Expression to negate.

        Returns
        -------
        MathExpr
            Symbolic negation of ``operand``.
        """
        from .functional.elementwise import neg
        return neg(operand)
    # end def neg

    @staticmethod
    def pow(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Create an elementwise power node.

        Parameters
        ----------
        operand1 : ExprLike
            Base expression.
        operand2 : ExprLike
            Exponent expression.

        Returns
        -------
        MathExpr
            Symbolic representation of ``operand1 ** operand2``.
        """
        from .functional.elementwise import pow as elementwise_pow
        return elementwise_pow(operand1, operand2)
    # end def pow

    @staticmethod
    def exp(operand: ExprLike) -> MathExpr:
        """
        Create an elementwise exponential node.

        Parameters
        ----------
        operand : ExprLike
            Expression whose entries will be exponentiated.

        Returns
        -------
        MathExpr
            Node computing ``exp(operand)``.
        """
        from .functional.elementwise import exp as elementwise_exp
        return elementwise_exp(operand)
    # end def exp

    @staticmethod
    def log(operand: ExprLike) -> MathExpr:
        """
        Create an elementwise natural-logarithm node.

        Parameters
        ----------
        operand : ExprLike
            Expression whose entries will be transformed.

        Returns
        -------
        MathExpr
            Node computing ``log(operand)``.
        """
        from .functional.elementwise import log as elementwise_log
        return elementwise_log(operand)
    # end def log

    @staticmethod
    def sqrt(operand: ExprLike) -> MathExpr:
        """
        Create an elementwise square-root node.

        Parameters
        ----------
        operand : ExprLike
            Expression whose entries will be square-rooted.

        Returns
        -------
        MathExpr
            Node computing ``sqrt(operand)``.
        """
        from .functional.elementwise import sqrt as elementwise_sqrt
        return elementwise_sqrt(operand)
    # end def sqrt

    @staticmethod
    def log2(operand: ExprLike) -> MathExpr:
        """
        Create an elementwise base-2 logarithm node.

        Parameters
        ----------
        operand : ExprLike
            Expression whose entries will be transformed.

        Returns
        -------
        MathExpr
            Node computing ``log2(operand)``.
        """
        from .functional.elementwise import log2 as elementwise_log2
        return elementwise_log2(operand)
    # end def log2

    @staticmethod
    def log10(operand: ExprLike) -> MathExpr:
        """
        Create an elementwise base-10 logarithm node.

        Parameters
        ----------
        operand : ExprLike
            Expression whose entries will be transformed.

        Returns
        -------
        MathExpr
            Node computing ``log10(operand)``.
        """
        from .functional.elementwise import log10 as elementwise_log10
        return elementwise_log10(operand)
    # end def log10

    @staticmethod
    def matmul(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Create a matrix multiplication node.
        """
        from .functional.linear_algebra import matmul
        return matmul(operand1, operand2)
    # end def matmul

    @staticmethod
    def getitem(operand: ExprLike, index: Index) -> MathExpr:
        """
        Create a getitem node.
        """
        from .functional.structure import getitem
        from .math_slice import SliceExpr
        indices: List[Union[int, SliceExpr]] = [
            SliceExpr.from_slice(s) if isinstance(s, slice) else s
            for s in index
        ]
        return getitem(op1=operand, indices=indices)
    # end def getitem

    @staticmethod
    def eq(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        from .functional.boolean import eq
        from . import as_expr
        return eq(as_expr(operand1), as_expr(operand2))
    # end def eq

    @staticmethod
    def ne(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        from .functional.boolean import ne
        from . import as_expr
        return ne(as_expr(operand1), as_expr(operand2))
    # end def ne

    @staticmethod
    # Override less
    def lt(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Placeholder less-than operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        from .functional.boolean import lt
        from . import as_expr
        return lt(as_expr(operand1), as_expr(operand2))
    # end def lt

    # Override less or equal
    @staticmethod
    def le(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Placeholder less-or-equal operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        from .functional.boolean import le
        from . import as_expr
        return le(as_expr(operand1), as_expr(operand2))
    # end def le

    # Override greater
    @staticmethod
    def gt(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Placeholder greater-than operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        from .functional.boolean import gt
        from . import as_expr
        return gt(as_expr(operand1), as_expr(operand2))
    # end def gt

    # Override greater or equal
    @staticmethod
    def ge(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        """
        Placeholder greater-or-equal operator.

        Raises
        ------
        MathExprNotImplementedError
            Always raised; ordering is not defined for ``MathExpr``.
        """
        from .functional.boolean import ge
        from . import as_expr
        return ge(as_expr(operand1), as_expr(operand2))
    # end def ge

    @staticmethod
    def logical_not(operand: ExprLike) -> MathExpr:
        from .functional.boolean import logical_not
        from . import as_expr
        return logical_not(as_expr(operand))
    # end def logical_not

    @staticmethod
    def logical_any(operand: ExprLike) -> MathExpr:
        from .functional.boolean import logical_any
        from . import as_expr
        return logical_any(as_expr(operand))
    # end def logical_any

    @staticmethod
    def logical_all(operand: ExprLike) -> MathExpr:
        from .functional.boolean import logical_all
        from . import as_expr
        return logical_all(as_expr(operand))
    # end def logical_all

    @staticmethod
    def logical_and(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        from .functional.boolean import logical_and
        from . import as_expr
        return logical_and(as_expr(operand1), as_expr(operand2))
    # end def logical_and

    @staticmethod
    def logical_or(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        from .functional.boolean import logical_or
        from . import as_expr
        return logical_or(as_expr(operand1), as_expr(operand2))
    # end def logical_or

    @staticmethod
    def logical_xor(operand1: ExprLike, operand2: ExprLike) -> MathExpr:
        from .functional.boolean import logical_xor
        from . import as_expr
        return logical_xor(as_expr(operand1), as_expr(operand2))
    # end def logical_xor

    # endregion OPERATORS

    # region BINARY

    def __add__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import add
        from . import as_expr
        return add(as_expr(self), as_expr(other))
    # end def __add__

    def __radd__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import add
        from . import as_expr
        return add(as_expr(other), as_expr(self))
    # end def __radd__

    def __sub__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import sub
        from . import as_expr
        return sub(as_expr(self), as_expr(other))
    # end def __sub__

    def __rsub__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import sub
        from . import as_expr
        return sub(as_expr(other), as_expr(self))
    # end def __rsub__

    def __mul__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import mul
        from . import as_expr
        return mul(as_expr(self), as_expr(other))
    # end def __mul__

    def __rmul__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import mul
        from . import as_expr
        return mul(as_expr(other), as_expr(self))
    # end def __rmul__

    def __truediv__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import div
        from . import as_expr
        return div(as_expr(self), as_expr(other))
    # end def __truediv__

    def __rtruediv__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import div
        from . import as_expr
        return div(as_expr(other), as_expr(self))
    # end def __rtruediv__

    def __pow__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import pow
        from . import as_expr
        return pow(as_expr(self), as_expr(other))
    # end def __pow__

    def __rpow__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.elementwise import pow
        from . import as_expr
        return pow(as_expr(other), as_expr(self))
    # end def __rpow__

    # endregion BINARY

    # region UNARY

    def __neg__(self: MathExpr) -> "MathExpr":
        from .functional.elementwise import neg
        from . import as_expr
        return neg(as_expr(self))
    # end def __neg__

    def __matmul__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.linear_algebra import matmul
        from . import as_expr
        return matmul(as_expr(self), as_expr(other))
    # end def __matmul__

    def __rmatmul__(self: MathExpr, other: ExprLike) -> "MathExpr":
        from .functional.linear_algebra import matmul
        from . import as_expr
        return matmul(as_expr(other), as_expr(self))
    # end def __rmatmul__

    # endregion UNARY

    def __invert__(self: MathExpr) -> MathExpr:
        from .functional.boolean import logical_not
        from . import as_expr
        return logical_not(as_expr(self))
    # end __invert__

    def __and__(self: MathExpr, other: ExprLike) -> MathExpr:
        from .functional.boolean import logical_and
        from . import as_expr
        return logical_and(as_expr(self), as_expr(other))
    # end __and__

    def __rand__(self: MathExpr, other: ExprLike) -> MathExpr:
        from .functional.boolean import logical_and
        from . import as_expr
        return logical_and(as_expr(other), as_expr(self))
    # end __rand__

    def __or__(self: MathExpr, other: ExprLike) -> MathExpr:
        from .functional.boolean import logical_or
        from . import as_expr
        return logical_or(as_expr(self), as_expr(other))
    # end __or__

    def __ror__(self: MathExpr, other: ExprLike) -> MathExpr:
        from .functional.boolean import logical_or
        from . import as_expr
        return logical_or(as_expr(other), as_expr(self))
    # end __ror__

    def __xor__(self: MathExpr, other: ExprLike) -> MathExpr:
        from .functional.boolean import logical_xor
        from . import as_expr
        return logical_xor(as_expr(self), as_expr(other))
    # end __xor__

    def __rxor__(self: MathExpr, other: ExprLike) -> MathExpr:
        from .functional.boolean import logical_xor
        from . import as_expr
        return logical_xor(as_expr(other), as_expr(self))
    # end __rxor__

    # endregion COMPARISON

    # Get item
    def __getitem__(self: ExprLike, index: Index):
        from .functional.structure import getitem
        from .math_slice import SliceExpr
        indices: List[Union[int, SliceExpr]] = [
            SliceExpr.from_slice(s) if isinstance(s, slice) else s
            for s in index
        ]
        return getitem(op1=self, indices=indices)
    # end def __getitem__

# end class ExprOperatorsMixin


class SimplifyRuleMixin:
    """
    Mixin for applying simplification rules to expressions.
    """

    # List of simplification rules
    _rules: tuple[Callable, ...] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Subclass gets the rules of the parents
        inherited = []
        for base in cls.__mro__[1:]:
            if not hasattr(base, "_rules"):
                inherited.extend(getattr(base, "_rules", ()))
            # end if
        # end for

        # 2. collect rules declared in the current class
        local = []
        for _, obj in cls.__dict__.items():
            spec = getattr(obj, "__rule_spec__", None)
            if spec is not None:
                local.append((spec.priority, obj))
            # end if
        # end for

        # 3. sort by priority + stable order
        local.sort(key=lambda x: x[0])

        # 4. concatenate
        cls._rules = tuple(inherited + [fn for _, fn in local])
    # end def __init_subclass__

    def _apply_rule(
            self,
            rule: RuleSpec,
            options: SimplifyOptions | None = None,
            rule_type: SimplifyRuleType | None = None,
    ) -> bool:
        """Return ```True``` if the rule should be applied."""
        if not options:
            return True
        # end if

        if rule_type and rule_type != rule.rule_type and rule.rule_type != SimplifyRuleType.ALL:
            return False
        # end if

        # Rule is not disabled
        if rule.flag not in options.disabled:
            # No enabled list => always apply rule
            if not options.enabled:
                return True
            # Not in the enabled list => don't apply rule
            elif rule.flag not in options.enabled:
                return False
            # In the enabled list => apply rule
            else:
                return True
            # end if
        # end if

        return False
    # end def _apply_rule

    def _run_rules(
            self,
            operands,
            rule_type: SimplifyRuleType,
            options: Optional[SimplifyOptions] = None,
    ) -> OpSimplifyResult | None:
        """
        Apply simplification rules to the expression.

        Parameters
        ----------
        operands: tuple[Op, ...]
            The operands of the expression.
        options: SimplifyOptions
            The simplification options.

        Returns
        -------
        OpSimplifyResult | None
            The result of applying the simplification rules, or None if no rule was applied.
        """
        for fn in self._rules:
            spec = fn.__rule_spec__
            if not self._apply_rule(spec, options, rule_type):
                continue
            # end if
            out = fn(self, operands)   # out: OpSimplifyResult | None
            # out is None if rule did not apply
            if out is not None and out.replacement is not None:
                return out
            elif out is not None:
                operands = out.operands
            # end if
        # end for
        return OpSimplifyResult(operands=operands, replacement=None)
    # end def _run_rules

# end class SimplifyRuleMixin
