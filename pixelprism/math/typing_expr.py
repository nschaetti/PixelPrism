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


from __future__ import annotations

from enum import Enum, auto
from typing import Protocol, runtime_checkable, NamedTuple, List, Optional, FrozenSet, Dict, Any
from typing import TYPE_CHECKING, Tuple, Union, Sequence, TypeAlias
from typing import ClassVar, Mapping
from dataclasses import dataclass

from .typing_data import ScalarLike, Index
from .typing_rules import SimplifyOptions


if TYPE_CHECKING:
    from .tensor import Tensor
    from .math_node import MathNode
    from .math_leaves import Variable, Constant
    from .shape import Shape
    from .dtype import DType
# end if


__all__ = [
    "ExprPattern",
    "NodePattern",
    "VariablePattern",
    "ConstantPattern",
    "AnyPattern",
    "EllipsisPattern",
    "p_node",
    "p_var",
    "p_const",
    "p_any",
    "p_ellipsis",
    "node_match",
    "MatchResult",
    "MathExpr",
    "Operand",
    "Operands",
    "Operator",
    "LeafKind",
    "OpAssociativity",
    "OpSimplifyResult",
    "OpConstruct",
    "AlgebraicExpr",
    "ExprLike",
    "ExprKind",
    "ExprDomain",
    "FoldPolicy",
    "ScalarLike",
    "AritySpec",
    "OperatorSpec",
]


# Expression pattern
@dataclass(frozen=True, slots=True)
class ExprPattern:
    """Pattern for matching symbolic expressions.

    This class represents a pattern that can be used to match symbolic expressions
    against a given expression tree. It provides methods for pattern matching and
    pattern construction.
    """
    name: str | None = None
    return_expr: bool = False

    def __post_init__(self):
        if self.return_expr and self.name is None:
            raise ValueError("An identifier (name) must be provided when return_expr is True")
        # end if
    # end __post_init__

# end class ExprPattern


# Pattern matching a node
@dataclass(frozen=True, slots=True)
class NodePattern(ExprPattern):
    """
    Pattern for matching a specific node type in the expression tree.
    """

    op: str | None = None
    operands: Sequence[ExprPattern] | None = None
    commutative: bool = False
    arity: int | None = None
    variable: bool | None = None
    constant: bool | None = False
    shape: Shape | None = None
    dtype: DType | None = None

    def __str__(self):
        operands = self.operands or ()
        return f"node({self.op}, ({', '.join([str(x) for x in operands])}))"
    # end def __str__

# end class NodePattern

def node_match(
        op: str | None = None,
        children: Sequence[ExprPattern] | None = None,
        comm: bool = False,
        var: bool = False,
        const: bool = False,
        shape: Shape | None = None,
        dtype: DType | None = None,
) -> ExprPattern:
    """
    Factory function to create a NodePattern instance.
    """
    return p_node(
        op,
        *(tuple(children) if children is not None else ()),
        comm=comm,
        var=var,
        const=const,
        shape=shape,
        dtype=dtype,
    )
# end def node_match


def _pattern_capture_kwargs(as_: str | None) -> dict[str, Any]:
    if as_ is None:
        return {}
    # end if
    return {
        "name": as_,
        "return_expr": True,
    }
# end def _pattern_capture_kwargs


def p_node(
        op: str | None = None,
        *operands: ExprPattern,
        comm: bool = False,
        arity: int | None = None,
        var: bool | None = None,
        const: bool = False,
        shape: Shape | None = None,
        dtype: DType | None = None,
        as_: str | None = None,
) -> NodePattern:
    """Compact builder for :class:`NodePattern`."""
    return NodePattern(
        op=op,
        operands=operands or None,
        commutative=comm,
        arity=arity,
        variable=var,
        constant=const,
        shape=shape,
        dtype=dtype,
        **_pattern_capture_kwargs(as_),
    )
# end def p_node


def p_var(
        var_name: str | None = None,
        *,
        shape: Shape | None = None,
        dtype: DType | None = None,
        as_: str | None = None,
) -> VariablePattern:
    """Compact builder for :class:`VariablePattern`."""
    return VariablePattern(
        var_name=var_name,
        shape=shape,
        dtype=dtype,
        **_pattern_capture_kwargs(as_),
    )
# end def p_var


def p_const(
        value: ScalarLike | None = None,
        const_name: str | None = None,
        *,
        shape: Shape | None = None,
        dtype: DType | None = None,
        as_: str | None = None,
) -> ConstantPattern:
    """Compact builder for :class:`ConstantPattern`."""
    return ConstantPattern(
        const_name=const_name,
        value=value,
        shape=shape,
        dtype=dtype,
        **_pattern_capture_kwargs(as_),
    )
# end def p_const


def p_any(
        *,
        shape: Shape | None = None,
        dtype: DType | None = None,
        as_: str | None = None,
) -> AnyPattern:
    """Compact builder for :class:`AnyPattern`."""
    return AnyPattern(
        shape=shape,
        dtype=dtype,
        **_pattern_capture_kwargs(as_),
    )
# end def p_any


def p_ellipsis(
        *,
        as_: str | None = None,
) -> EllipsisPattern:
    """Compact builder for :class:`EllipsisPattern`."""
    return EllipsisPattern(
        **_pattern_capture_kwargs(as_),
    )
# end def p_ellipsis


# Pattern matching a variable
@dataclass(frozen=True, slots=True)
class VariablePattern(ExprPattern):
    """
    Pattern for matching a variable node in the expression tree.
    """
    var_name: str | None = None
    shape: Shape | None = None
    dtype: DType | None = None

    def __str__(self):
        return f"var({self.var_name})"
    # end def __str__

# end class VariablePattern


# Pattern matching a constante
@dataclass(frozen=True, slots=True)
class ConstantPattern(ExprPattern):
    """
    Pattern for matching a constant node in the expression tree.
    """
    const_name: str | None = None
    value: ScalarLike | None = None
    shape: Shape | None = None
    dtype: DType | None = None

    def __str__(self):
        return f"const({self.const_name}={self.value})"
    # end def __str__

# end class ConstantPattern


# Pattern matching any expression
@dataclass(frozen=True, slots=True)
class AnyPattern(ExprPattern):
    """
    Pattern for matching any node in the expression tree.
    """
    shape: Shape | None = None
    dtype: DType | None = None

    def __str__(self):
        return "any()"
    # end def __str__

# end class AnyPattern


# Pattern matching a variable number of operands
@dataclass(frozen=True, slots=True)
class EllipsisPattern(ExprPattern):
    """
    Pattern for matching a variable number of operands.
    """

    def __str__(self):
        return "..."
    # end def __str__

# end class EllipsisPattern


@dataclass(slots=True)
class MatchResult:
    """
    Result of a pattern matching operation.
    """
    matched: bool
    bindings: dict[str, MathExpr]

    @classmethod
    def failed(cls):
        return cls(matched=False, bindings={})
    # end def failed

    @classmethod
    def success(cls, bindings: dict[str, MathExpr]):
        return cls(matched=True, bindings=bindings)
    # end def success

    @classmethod
    def merge(cls, results: Sequence[MatchResult], all_success: bool = False):
        bindings = {}
        for result in results:
            if result.matched:
                bindings.update(result.bindings)
            elif all_success:
                return cls.failed()
            # end if
        # end for
        return cls.success(bindings)
    # end def merge

# end class MatchResult


#
# Expression tree node kinds
#
class ExprKind(Enum):
    NODE = auto()
    LEAF = auto()
    CONSTANT = auto()
    VARIABLE = auto()
    SHAPE = auto()
    SLICE = auto()
# end class ExprKind


#
# Domain of the expression tree.
#
class ExprDomain(Enum):
    ALGEBRAIC = auto()
    SHAPE = auto()
    SLICE = auto()
# end class ExprDomain


#
# Leaf filtering for structural queries (`contains*`).
#
class LeafKind(Enum):
    # No filtering: variables and constants are both considered.
    ANY = auto()
    # Restrict search to variable leaves only.
    VARIABLE = auto()
    # Restrict search to constant leaves only.
    CONSTANT = auto()
# end class LeafKind


#
# Folding policy for constant-only subexpressions.
#
class FoldPolicy(Enum):
    FOLDABLE = auto()
    SYMBOLIC_LOCKED = auto()
    NON_FOLDABLE = auto()
# end class FoldPolicy


# Protocol for mathematical expressions
# Abstract class is ExpressionMixin
@runtime_checkable
class MathExpr(Protocol):

    #
    # Properties and values
    #

    # Symbolic output shape of the expression.
    # Must stay consistent with operator/operand shape inference.
    @property
    def shape(self) -> "Shape": ...

    # Element dtype produced by the expression evaluation.
    # This is the advertised/static dtype contract.
    @property
    def dtype(self) -> "DType": ...

    # Stable expression identifier.
    # Required for tracing, debugging, and deterministic rewrites.
    @property
    def name(self) -> str: ...

    # Tensor rank derived from `shape`.
    # Convenience property used by operators and validators.
    @property
    def rank(self) -> int: ...

    # Domain of the expression tree.
    @property
    def domain(self) -> ExprDomain: ...

    # Expression type.
    @property
    def kind(self) -> ExprKind: ...

    # Operator-based tree nodes have a reference to their operator specification.
    @property
    def spec(self) -> Optional[OperatorSpec]: ...

    # Operator-based tree nodes have a reference to their operator instance.
    @property
    def op(self) -> Optional[Operator]: ...

    # Operator name. Only applicable to nodes (else None).
    @property
    def op_name(self) -> Optional[str]: ...

    # Number of operands for the node. Only applicable to nodes (else 0).
    @property
    def arity(self) -> int: ...

    # Number of children for the node. Only applicable to nodes (else []).
    @property
    def children(self) -> Sequence["MathExpr"]: ...

    # Parent nodes
    @property
    def parents(self) -> FrozenSet[MathExpr]: ...

    #
    # Evaluate and differentiate
    #

    # Evaluate the expression in the active runtime context and return a tensor.
    # The returned value must match the expression contract (`dtype`, `shape`).
    def eval(self) -> "Tensor": ...

    # Symbolic derivative of the expression with respect to one variable.
    # Returns a new expression tree (not a numeric value), suitable for further rewrites.
    def diff(self, wrt: "Variable") -> "MathExpr": ...

    #
    # Structure
    #

    # Return all variable leaves reachable from this expression.
    # Order may follow traversal order; duplicates are not allowed.
    # In ExpressionMixin
    def variables(self) -> Sequence["Variable"]: ...

    # Return all constant leaves reachable from this expression.
    # Order may follow traversal order; duplicates not are allowed.
    # In ExpressionMixin
    def constants(self) -> Sequence["Constant"]: ...

    # Generic membership query in the expression tree.
    # - `leaf` can be a name or an expression instance.
    # - `by_ref=True` enforces identity-based matching.
    # - `check_operator=True` also inspects operator-owned symbolic parameters.
    # - `look_for` narrows the search domain (e.g. "var", "const"); None means no filter.
    # In ExpressionMixin
    def contains(
            self,
            leaf: Union[str, "MathExpr"],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool: ...

    # Convenience wrapper around `contains(..., look_for=LeafKind.VARIABLE)`.
    # In ExpressionMixin
    def contains_variable(
            self,
            variable: Union[str, "Variable"],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool: ...

    # Convenience wrapper around `contains(..., look_for=LeafKind.CONSTANT)`.
    # In ExpressionMixin
    def contains_constant(
            self,
            constant: Union[str, "Constant"],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool: ...

    #
    # Transforms
    #

    # Apply symbolic rewrite rules and return a simplified expression.
    # This operation is pure: it never mutates the current tree.
    # `options` controls which rules are enabled/disabled for this pass.
    def simplify(self, options: SimplifyOptions | None = None) -> "MathExpr": ...

    # Normalize an expression form without changing semantics.
    # Typical effects: associative flattening, deterministic operand ordering, etc.
    def canonicalize(self) -> "MathExpr": ...

    # Fold constant-only subexpressions into constant leaves.
    # This is a focused transform and may be used independently of full simplifying.
    def fold_constants(self) -> "MathExpr": ...

    # Replace matching subexpressions using `mapping` and return a new tree.
    # - `by_re  f=True`: match by object identity.
    # - `by_ref=False`: match by symbolic/tree equality policy.
    # In ExpressionMixin
    def substitute(
            self,
            mapping: Mapping["MathExpr", "MathExpr"],
            *,
            by_ref: bool = True
    ) -> "MathExpr": ...

    # Rename occurrences of `old_name` to `new_name`.
    # The change is in-place and does not return a new tree.
    # In ExpressionMixin
    def renamed(self, old_name: str, new_name: str) -> List[str]: ...

    #
    # Comparison
    #

    # Strict symbolic tree equality.
    # Returns True only if both expressions have the same structure
    # (same node kinds/operators, same operand order, same leaf content).
    def eq_tree(self, other: "MathExpr") -> bool: ...

    # Mathematical symbolic equivalence.
    # Returns True when both expressions represent the same symbolic meaning,
    # even if their trees differ (e.g., after canonicalization/simplification).
    # This check is symbolic only (no numeric/probabilistic fallback).
    def equivalent(self, other: "MathExpr") -> bool: ...

    # Return `true` if the two expressions return the same value when evaluated.
    # In ExpressionMixin
    def equals(
            self: "MathExpr",
            other: "MathExpr | Tuple | List",
            *,
            rtol: float = 1e-6,
            atol: float = 1e-9,
            equal_nan: bool = False,
            require_same_shape: bool = True,
            require_same_dtype: bool = False
    ) -> bool: ...

    # Match a symbolic expression to a pattern.
    # - `pattern` is a string or a regular expression.
    # - `by_ref=True`: match by object identity.
    # - `by_ref=False`: match by symbolic/tree equality policy.
    # In ExpressionMixin
    def match(
            self,
            pattern: ExprPattern
    ) -> MatchResult: ...

    #
    # Boolean checks
    #

    # True if the expression contains no variable dependency
    # (i.e., it evaluates from constants only).
    # In ExpressionMixin
    def is_constant(self) -> bool: ...

    # True if the expression contains at least one variable dependency.
    # In ExpressionMixin
    def is_variable(self) -> bool: ...

    # True for operator-based internal tree nodes.
    def is_node(self) -> bool: ...

    # True for terminal tree elements (variables/constants) with no children.
    def is_leaf(self) -> bool: ...

    # True if leaf and constant
    def is_constant_leaf(self) -> bool: ...

    # True if leaf and variable
    def is_variable_leaf(self) -> bool: ...

    # True if the expression is deterministically evaluable (no randomness or context side effects).
    def is_pure(self) -> bool: ...

    # True if behave like a scalar (rank = 1)
    def is_scalar(self) -> bool: ...
    def is_vector(self) -> bool: ...
    def is_matrix(self) -> bool: ...
    def is_tensor(self) -> bool: ...

    # Check if the expression as an operator with the given name.
    def has_operator(self, name: str) -> bool: ...

    # Check if the expression has children.
    def has_children(self) -> bool: ...

    #
    # Rules and policy
    #

    # True if the expression is foldable
    def is_foldable(self) -> bool: ...

    # Fold policy
    def fold_policy(self) -> FoldPolicy: ...

    #
    # Structure
    #

    # Check the number of children of the expression.
    def num_children(self) -> int: ...

    # Return the maximum depth of the expression tree.
    # Convention: leaves have depth 1.
    def depth(self) -> int: ...

    # Return a copy of the expression.
    # - deep=False: may reuse existing children/subtrees when safe.
    # - deep=True: recursively duplicates the full expression tree.
    def copy(self, deep: bool = False) -> "MathExpr": ...

    #
    # Comparaison
    #
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

    #
    # Representation
    #

    # Human-readable representation intended for users/logs.
    def __str__(self) -> str: ...

    # Developer-oriented representation intended for debugging.
    # Should be unambiguous and include key structural identifiers.
    def __repr__(self) -> str: ...

# end class MathExpr


# Operands are the operands of an operator.
# Can be a single operand or a tuple of operands.
# Can be a MathNode, Variable, or Constant, then a MathExpr.
Operand: TypeAlias = "MathExpr"
Operands = Sequence[Operand]

# Expression type
ExprLike = Union[MathExpr, ScalarLike]


class OpSimplifyResult(NamedTuple):
    operands: Operands | None                      # Normalized/simplified operands
    replacement: MathExpr | None = None     # If not None, the operator was replaced by this expression
# end def OpSimplifyResult


class OpConstruct(NamedTuple):
    expr: Operator | MathExpr           # Constructed operator or simplified expression
    operands: Operands | None           # Operator operands or None if replaced by replacement
# end class OpConstruct


class OpAssociativity(Enum):
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()
# end class Associativity


@dataclass(frozen=True)
class AritySpec:
    # Exact arity for fixed-arity operators.
    exact: int | None = None

    # Declare the minimum operands needed
    min_operands: int = 0

    # True when the operator accepts a variable number of operands.
    # When True, arity validation is handled dynamically.
    variadic: bool = False

    @property
    def n_ops(self) -> int:
        if self.variadic:
            return self.min_operands
        # end if
        if self.exact is None:
            raise ValueError("AritySpec.exact must be set for non-variadic operators")
        # end if
        return self.exact
    # end def arity

# end class AritySpec


@dataclass(frozen=True)
class OperatorSpec:
    # Stable registry/operator identifier (e.g., "add", "mul", "log").
    name: str

    # Symbol of the operator in the expression tree.
    symbol: str

    # Declared operator arity.
    # For fixed-arity operators, this is the exact number of required operands.
    arity: AritySpec

    # True if operand order does not change semantic result.
    # Example: add(a, b) == add(b, a)
    commutative: bool = False

    # Set if the operator is left or right associative.
    # Operators with the same precedence are evaluated from left to right.
    associative: bool = False

    # Set the precedent of the operator in the expression tree.
    # Operators with higher precedence are evaluated first.
    precedence: int = 0

    # Set if the operator is left or right associative.
    # Operators with the same precedence are evaluated from left to right.
    associativity: OpAssociativity = OpAssociativity.NONE

    # True when the operator is intended for differentiation-time semantics
    # (e.g., special derivative helper operators).
    is_diff: bool = False
# end class OperatorSpec


# Protocol for mathematical operators
@runtime_checkable
class Operator(Protocol):

    #
    # Class variables
    #

    # Operator specification.
    SPEC: ClassVar[OperatorSpec]

    #
    # Properties
    #

    # Runtime operator name.
    # Must match the class-level stable identifier (`NAME`).
    @property
    def name(self) -> str: ...

    # Effective arity for this operator instance.
    # For fixed-arity operators, this typically equals `ARITY`.
    # For variadic operators, this may reflect runtime operand count.
    @property
    def arity(self) -> AritySpec: ...

    # The minimum number of operands required by the operator.
    # This is the same as `MIN_OPERANDS` for fixed-arity operators.
    # For variadic operators, this may reflect runtime operand count.
    @property
    def min_operands(self) -> int: ...

    # Operator specification.
    @property
    def spec(self) -> OperatorSpec: ...

    # Operator symbol in the expression tree.
    @property
    def symbol(self) -> str: ...

    # Operator precedence in the expression tree.
    @property
    def precedence(self) -> int: ...

    # Operator associativity in the expression tree.
    @property
    def associativity(self) -> OpAssociativity: ...

    # Commutativity flag.
    @property
    def commutative(self) -> bool: ...

    # True if the operator is intended for differentiation-time semantics.
    @property
    def is_diff(self) -> bool: ...

    # True if the operator accepts a variable number of operands.
    @property
    def is_variadic(self) -> bool: ...

    # Parent operator instance.
    @property
    def parent(self) -> Operator | None: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    #
    # Parent
    #
    def set_parent(self, parent: MathExpr) -> None: ...

    #
    # Evaluate and differentiate
    #

    # Evaluate operator output for already-built symbolic operands.
    # Executes numeric computation in the active runtime context.
    # `kwargs` carries operator-specific runtime parameters when needed.
    def eval(self, operands: Operands, **kwargs) -> "Tensor": ...

    # Return a symbolic derivative of this operator application with respect to `wrt`.
    # The result is an expression tree (not a numeric tensor).
    # `operands` are the original operator inputs used to apply differentiation rules.
    def diff(self, wrt: "Variable", operands: Operands) -> "MathExpr": ...

    #
    # Simplification
    #

    # Fold constant-only subexpressions into constant leaves.
    # This is a focused transform and may be used independently of full simplifying.
    def fold_constants(self, operands: Operands) -> OpSimplifyResult: ...

    # Apply operator-local simplification rules to already simplified operands.
    # Returns:
    # - `operands`: possibly rewritten operands for the same operator node.
    # - `replacement`: optional full-node replacement (e.g., add(x, 0) -> x).
    # If `replacement` is not None, the caller should replace the whole node.
    def simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult: ...

    # Return canonicalized operands for this operator without changing semantics.
    # Intended for a deterministic tree form (e.g., flatten associative, sort commutative).
    # This does not directly replace the node; the caller rebuilds with returned operands.
    def canonicalize(
            self,
            operands: Sequence[MathExpr]
    ) -> OpSimplifyResult: ...

    #
    # Structure
    #

    # Return True if `expr` is referenced by operator-owned symbolic parameters.
    # - `by_ref=True`: identity-based match.
    # - `by_ref=False`: symbolic/name-based match depending on expression policy.
    # - `look_for` restricts lookup scope (any / variables / constants).
    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool: ...

    # Copy the operator instance with new operands.
    def copy(self, deep: bool = False) -> Operator: ...

    #
    # Infer shapes and dtypes
    #

    # Infer the result dtype from operand dtypes and operator semantics.
    # Must be deterministic and consistent with `eval`.
    def infer_dtype(self, operands: Operands) -> "DType": ...

    # Infer the result shape from operand shapes and operator semantics.
    # Must be deterministic and consistent with `eval`.
    def infer_shape(self, operands: Operands) -> "Shape": ...

    #
    # Checks
    #

    # Validate operand-level semantic constraints specific to the operator.
    # Example: required operand kinds, valid axis expression type, etc.
    def check_operands(self, operands: Operands) -> bool: ...

    # Validate operator initialization/runtime parameters.
    # Example: axis range, non-zero step, valid mode flags, etc.
    def check_parameters(self, **kwargs) -> bool: ...

    # Validate shape compatibility constraints for this operator.
    # Example: elementwise compatibility, matmul inner dims, concat axis agreement.
    def check_shapes(self, operands: Operands) -> bool: ...

    @classmethod
    # Validate operand count against operator arity contract.
    # For fixed-arity operators, this checks the exact length.
    # For variadic operators, this validates the minimal / allowed count policy.
    def check_arity(cls, operands: Operands) -> bool: ...

    #
    # Factory methods
    #

    @classmethod
    # Construct the operator. This is the official and only way to create the operator instance.
    # The method will check rule of simplification.
    def construct(cls, operands: Operands, **kwargs) -> OpConstruct: ...

    @classmethod
    # Create a new operator node from the given operands.
    def create_node(cls, operands: Operands, **kwargs) -> MathExpr: ...

    #
    # Representation
    #

    # Print operator as a mathematical expression.
    def print(self, operands: Operands, **kwargs) -> str: ...

    # Two operators are equals if they do the same operation.
    def __eq__(self, other: object) -> bool: ...
    def __neq__(self, other: object) -> bool: ...

    # Human-readable operator representation for logs and user-facing displays.
    def __str__(self) -> str: ...

    # Developer/debug representation including key operator metadata.
    # Should be unambiguous and useful when inspecting expression trees.
    def __repr__(self) -> str: ...

# end class Operator


# Algebraic expression
#  It abstracts class is AlgebraicMixin
@runtime_checkable
class AlgebraicExpr(MathExpr, Protocol):

    # Statics
    @staticmethod
    def add(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def sub(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def mul(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def div(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def neg(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def pow(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def exp(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def log(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def sqrt(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def log2(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def log10(operand: ExprLike) -> MathNode: ...

    @staticmethod
    def matmul(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...

    @staticmethod
    def getitem(operand: ExprLike, index: Index) -> MathNode: ...
    @staticmethod
    def eq(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def ne(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def lt(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def le(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def gt(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def ge(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...

    @staticmethod
    def logical_not(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def logical_any(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def logical_all(operand: ExprLike) -> MathNode: ...
    @staticmethod
    def logical_and(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def logical_or(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...
    @staticmethod
    def logical_xor(operand1: ExprLike, operand2: ExprLike) -> MathNode: ...

    # Override
    def __add__(self, other: ExprLike) -> MathNode: ...
    def __radd__(self, other: ExprLike) -> MathNode: ...
    def __sub__(self, other: ExprLike) -> MathNode: ...
    def __rsub__(self, other: ExprLike) -> MathNode: ...
    def __mul__(self, other: ExprLike) -> MathNode: ...
    def __rmul__(self, other: ExprLike) -> MathNode: ...
    def __truediv__(self, other: ExprLike) -> MathNode: ...
    def __rtruediv__(self, other: ExprLike) -> MathNode: ...
    def __pow__(self, other: ExprLike) -> MathNode: ...
    def __rpow__(self, other: ExprLike) -> MathNode: ...
    def __neg__(self) -> MathNode: ...

    def __matmul__(self, other: ExprLike) -> MathNode: ...
    def __rmatmul__(self, other: ExprLike) -> MathNode: ...

    def __invert__(self) -> MathNode: ...
    def __and__(self, other: ExprLike) -> MathNode: ...
    def __rand__(self, other: ExprLike) -> MathNode: ...
    def __or__(self, other: ExprLike) -> MathNode: ...
    def __ror__(self, other: ExprLike) -> MathNode: ...
    def __xor__(self, other: ExprLike) -> MathNode: ...
    def __rxor__(self, other: ExprLike) -> MathNode: ...
    def __getitem__(self, item: Index) -> MathNode: ...

# end class AlgebraicExpr
