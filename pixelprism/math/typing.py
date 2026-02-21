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

from dataclasses import dataclass
from enum import Enum, auto
from types import EllipsisType
from typing import Protocol, runtime_checkable, NamedTuple, FrozenSet, List
from typing import TYPE_CHECKING, Tuple, Union, Sequence, TypeAlias
from typing import ClassVar, Mapping
import numpy as np


if TYPE_CHECKING:
    from .tensor import Tensor
    from .math_node import MathNode
    from .math_leaves import Variable, Constant
    from .shape import Shape
    from .dtype import DType
# end if


__all__ = [
    "ScalarLike",
    "ScalarListLike",
    "Index",
    "MathExpr",
    "TensorLike",
    "Operand",
    "Operands",
    "Operator",
    "SimplifyRule",
    "SimplifyOptions",
    "LeafKind",
    "OpAssociativity",
    "OpSimplifyResult",
    "AlgebraicExpr",
    "ExprLike"
]


# Type numeric
# Scalar Python/NumPy accepted as atomic numeric values in symbolic expressions.
# Includes bool and complex to match Tensor/domain operator coverage.
ScalarLike = float | int | np.number | bool | complex

# Recursive Python list representation of scalar tensor-like data.
# Example: [1, 2, [3, 4.5]]
ScalarListLike: TypeAlias = list[Union[ScalarLike, "ScalarListLike"]]

# Tensor data
# Any accepted tensor payload: scalar, nested Python lists, or NumPy array.
TensorLike: TypeAlias = Union[ScalarLike, ScalarListLike, np.ndarray]


# Single indexing atom accepted in one axis position.
# - int: pick one element on an axis
# - slice: ranged selection
# - None: insert a new axis (numpy.newaxis)
# - Ellipsis: expand to remaining axes
# - Sequence[int]: integer fancy indexing on one axis
# - Sequence[bool]: boolean mask indexing on one axis
IndexAtom: TypeAlias = Union[
    int,
    slice,
    None,
    EllipsisType,
    Sequence[int],
    Sequence[bool],
]

# Full tensor index:
# - a single atom (e.g. x[3], x[1:5], x[..., 0])
# - or a tuple of atoms for multi-axis indexing (e.g. x[:, 1, ..., None])
Index: TypeAlias = Union[
    IndexAtom,
    Tuple[IndexAtom, ...],
]


# Symbolic rewrite rules available to `simplify`.
# - Algebraic identities reduce expression size (e.g. x + 0 -> x).
# - Canonicalization rules normalize a tree form for stable comparisons.
class SimplifyRule(Enum):
    # Algebraic identities
    ADD_ZERO = auto()           # x + 0 -> x ; 0 + x -> x
    ADD_NEG = auto()            # x + -y -> x - y
    ADD_ITSELF = auto()         # x + x -> 2x
    ADD_AX_BX = auto()          # a*x + b*x -> (a+b)*x

    SUB_ZERO = auto()           # x - 0 -> x
    SUB_ITSELF = auto()         # x - x -> 0

    MUL_ONE = auto()            # x * 1 -> x ; 1 * x -> x
    MUL_ZERO = auto()           # x * 0 -> 0 ; 0 * x -> 0
    MUL_BY_CONSTANT = auto()    # x * c -> c * x
    MUL_ITSELF = auto()         # x * x -> x^2
    MUL_NEG = auto()            # x * -y -> -x * y
    MUL_BY_NEG_ONE = auto()     # x * -1 -> -x
    MUL_BY_INV = auto()         # x * 1/y -> x/y
    MUL_BY_INV_NEG = auto()     # x * -1/y -> -x/y
    MUL_BY_NEG_ITSELF = auto()  # x * -x -> -x^2

    DIV_ONE = auto()           # x / 1 -> x
    ZERO_DIV = auto()          # 0 / x -> 0

    MERGE_CONSTANTS = auto()   # a + b -> c

    # Remove double negation
    NEGATE_NEGATE = auto()

    # Constant folding
    CONST_FOLD = auto()        # combine constant-only subexpressions

    # Canonicalization (normal form)
    FLATTEN_ASSOC = auto()     # (a+b)+c -> a+b+c ; (a*b)*c -> a*b*c
    SORT_COMMUTATIVE = auto()  # stable operand ordering for + and *
# end class SimplifyRule


# Rule selection for one simplify pass.
# - enabled=None means "all rules enabled by default".
# - disabled always takes precedence over enabled.
@dataclass(frozen=True)
class SimplifyOptions:
    enabled: FrozenSet[SimplifyRule] | None = None
    disabled: FrozenSet[SimplifyRule] = frozenset()
# end class SimplifyOptions


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


# Protocol for mathematical expressions
@runtime_checkable
class MathExpr(Protocol):

    #
    # Properties
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

    #
    # Evaluate and differentiate
    #

    # Evaluate the expression in the active runtime context and return a tensor.
    # The returned value must match the expression contract (`dtype`, `shape`).
    def eval(self) -> "Tensor": ...

    # Symbolic derivative of the expression with respect to one variable.
    # Returns a new expression tree (not a numeric value), suitable for further rewrites.
    def diff(self, wrt: "Variable") -> "MathExpr": ...

    # Return `true` if the two expressions return the same value when evaluated.
    def equals(
            self: MathExpr,
            other: MathExpr | Tuple | List,
            *,
            rtol: float = 1e-6,
            atol: float = 1e-9,
            equal_nan: bool = False,
            require_same_shape: bool = True,
            require_same_dtype: bool = False
    ) -> bool: ...

    #
    # Structure
    #

    # Return all variable leaves reachable from this expression.
    # Order may follow traversal order; duplicates are not allowed.
    def variables(self) -> Sequence["Variable"]: ...

    # Return all constant leaves reachable from this expression.
    # Order may follow traversal order; duplicates not are allowed.
    def constants(self) -> Sequence["Constant"]: ...

    # Generic membership query in the expression tree.
    # - `leaf` can be a name or an expression instance.
    # - `by_ref=True` enforces identity-based matching.
    # - `check_operator=True` also inspects operator-owned symbolic parameters.
    # - `look_for` narrows the search domain (e.g. "var", "const"); None means no filter.
    def contains(
            self,
            leaf: Union[str, "MathExpr"],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool: ...

    # Convenience wrapper around `contains(..., look_for=LeafKind.VARIABLE)`.
    def contains_variable(
            self,
            variable: Union[str, "Variable"],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool: ...

    # Convenience wrapper around `contains(..., look_for=LeafKind.CONSTANT)`.
    def contains_constant(
            self,
            constant: Union[str, "Constant"],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool: ...

    #
    # Immutable transforms
    #

    # Apply symbolic rewrite rules and return a simplified expression.
    # This operation is pure: it never mutates the current tree.
    # `options` controls which rules are enabled/disabled for this pass.
    def simplify(self, options: SimplifyOptions | None = None) -> "MathExpr": ...

    # Normalize expression form without changing semantics.
    # Typical effects: associative flattening, deterministic operand ordering, etc.
    def canonicalize(self) -> "MathExpr": ...

    # Fold constant-only subexpressions into constant leaves.
    # This is a focused transform and may be used independently of full simplifying.
    def fold_constants(self) -> "MathExpr": ...

    # Replace matching subexpressions using `mapping` and return a new tree.
    # - `by_ref=True`: match by object identity.
    # - `by_ref=False`: match by symbolic/tree equality policy.
    def substitute(
            self,
            mapping: Mapping["MathExpr", "MathExpr"],
            *,
            by_ref: bool = True
    ) -> "MathExpr": ...

    # Return a new expression where occurrences of `old_name` are renamed to `new_name`.
    # This transform is immutable and does not alter the current instance.
    def renamed(self, old_name: str, new_name: str) -> "MathExpr": ...

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

    #
    # Boolean checks
    #

    # True if the expression contains no variable dependency
    # (i.e., it evaluates from constants only).
    def is_constant(self) -> bool: ...

    # True if the expression contains at least one variable dependency.
    def is_variable(self) -> bool: ...

    # True for operator-based internal tree nodes.
    def is_node(self) -> bool: ...

    # True for terminal tree elements (variables/constants) with no children.
    def is_leaf(self) -> bool: ...

    # Check if the expression as an operator with the given name.
    def has_operator(self, name: str) -> bool: ...

    #
    # Structure
    #

    # Return the maximum depth of the expression tree.
    # Convention: leaves have depth 1.
    def depth(self) -> int: ...

    # Return a copy of the expression.
    # - deep=False: may reuse existing children/subtrees when safe.
    # - deep=True: recursively duplicates the full expression tree.
    def copy(self, deep: bool = False) -> "MathExpr": ...

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
Operand: TypeAlias = "AlgebraicExpr"
Operands = Sequence[Operand]

# Expression type
ExprLike = Union[MathExpr, ScalarLike]


class OpSimplifyResult(NamedTuple):
    operands: Operands                      # Normalized/simplified operands
    replacement: MathExpr | None = None     # If not None, the operator was replaced by this expression
# end def OpSimplifyResult


class OpAssociativity(Enum):
    LEFT = auto()
    RIGHT = auto()
    NONE = auto()
# end class Associativity


# Protocol for mathematical operators
@runtime_checkable
class Operator(Protocol):

    #
    # Class variables
    #

    # Declared operator arity.
    # For fixed-arity operators, this is the exact number of required operands.
    ARITY: ClassVar[int]

    # True when the operator accepts a variable number of operands.
    # When True, arity validation is handled dynamically.
    IS_VARIADIC: ClassVar[bool]

    # True when the operator is intended for differentiation-time semantics
    # (e.g., special derivative helper operators).
    IS_DIFF: ClassVar[bool]

    # Stable registry/operator identifier (e.g., "add", "mul", "log").
    NAME: ClassVar[str]

    # True if operand order does not change semantic result.
    # Example: add(a, b) == add(b, a)
    COMMUTATIVE: ClassVar[bool]

    # True if regrouping does not change semantic result.
    # Example: add(add(a, b), c) == add(a, add(b, c))
    ASSOCIATIVE: ClassVar[bool]

    # Set the precedence of the operator in the expression tree.
    # Operators with higher precedence are evaluated first.
    PRECEDENCE: ClassVar[int]

    # Set if the operator is left or right associative.
    # Operators with the same precedence are evaluated from left to right.
    ASSOCIATIVITY: ClassVar[OpAssociativity]

    # Symbol of the operator in the expression tree.
    SYMBOL: ClassVar[str]

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
    def arity(self) -> int: ...

    # Set effective arity for this operator instance.
    # Intended mainly for variadic operators after operand validation.
    @arity.setter
    def arity(self, value: int) -> None: ...

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
    def canonicalize(self, operands: Sequence[MathExpr]) -> Sequence[MathExpr]: ...

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

    #
    # Factory methods
    #

    @classmethod
    # Validate operand count against operator arity contract.
    # For fixed-arity operators, this checks the exact length.
    # For variadic operators, this validates the minimal / allowed count policy.
    def check_arity(cls, operands: Operands) -> bool: ...

    @classmethod
    # Construct a MathNode bound to this operator from validated operands.
    # Implementations should infer dtype/shape and attach operator parameters from `kwargs`.
    # Returns a symbolic node ready for graph use.
    def create_node(cls, operands: Operands, **kwargs) -> "MathNode": ...

    #
    # Representation
    #

    # Print operator as a mathematical expression.
    def print(self, operands: Operands, **kwargs) -> str: ...

    # Human-readable operator representation for logs and user-facing displays.
    def __str__(self) -> str: ...

    # Developer/debug representation including key operator metadata.
    # Should be unambiguous and useful when inspecting expression trees.
    def __repr__(self) -> str: ...

# end class Operator


# Algebraic expression
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

    # def __eq__(self, other: ExprLike) -> MathNode: ...
    # def __ne__(self, other: ExprLike) -> MathNode: ...
    # def __lt__(self, other: ExprLike) -> MathNode: ...
    # def __le__(self, other: ExprLike) -> MathNode: ...
    # def __gt__(self, other: ExprLike) -> MathNode: ...
    # def __ge__(self, other: ExprLike) -> MathNode: ...

    def __invert__(self) -> MathNode: ...
    def __and__(self, other: ExprLike) -> MathNode: ...
    def __rand__(self, other: ExprLike) -> MathNode: ...
    def __or__(self, other: ExprLike) -> MathNode: ...
    def __ror__(self, other: ExprLike) -> MathNode: ...
    def __xor__(self, other: ExprLike) -> MathNode: ...
    def __rxor__(self, other: ExprLike) -> MathNode: ...
    def __getitem__(self, item: Index) -> MathNode: ...

# end class AlgebraicExpr
