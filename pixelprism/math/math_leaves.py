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
from typing import List, Optional, Tuple, Union, Mapping, Sequence, FrozenSet, Dict
import weakref

from .math_base import MathBase

from .math_exceptions import (
    SymbolicMathNotImplementedError,
    SymbolicMathValidationError,
    SymbolicMathLookupError,
    SymbolicMathRuntimeError
)

from .mixins import ExpressionMixin, AlgebraicMixin
from .dtype import DType, TypeLike, create
from .shape import Shape, ShapeLike, DimLike
from .tensor import Tensor, TensorLike, tensor
from .context import get_value
from .random import rand_name
from .typing import MathExpr, LeafKind, SimplifyOptions, AlgebraicExpr, ExprKind, ExprDomain, Operator, ExprPattern, AnyPattern, VariablePattern, ConstantPattern


__all__ = [
    "MathLeaf",
    "Variable",
    "Constant",
    "var",
    "const"
]

from .typing_expr import FoldPolicy, MatchResult


def var(name: str, dtype: TypeLike, shape: ShapeLike) -> Variable:
    """Create a new variable with the given name and dtype.

    Parameters
    ----------
    name: str
        Name of the variable.
    dtype: AnyDType
        Data type of the variable.
    shape: AnyShape
        Shape of the variable.
    """
    return Variable.create(name=name, dtype=create(dtype), shape=Shape.create(shape))
# end def var


def const(name: str, data: TensorLike, dtype: Optional[TypeLike] = None) -> Constant:
    """Create a new constant with the given value and dtype.

    Parameters
    ----------
    name: str
        Name of the constant.
    data: NumericType
        Data value of the constant.
    dtype: AnyDType, optional
        Data type of the constant. To the dtype of ``data`` if None.
    """
    data = tensor(data=data, dtype=dtype, mutable=False)
    return Constant.create(name=name, data=data)
# end def const


# An expression which does not contain sub-expressions
class MathLeaf(
    MathBase,
    ExpressionMixin,
    AlgebraicMixin,
    AlgebraicExpr,
    ABC
):
    """
    Abstract base for terminal expressions.

    Leaf nodes do not own operators or children; instead they store metadata
    needed to look up runtime tensors (variables) or to hold literal values
    (constants).  Subclasses must implement ``_eval`` and the traversal helpers
    that expose any nested variables/constants they reference.
    """

    def __init__(
            self,
            name: str,
            kind: ExprKind,
            domain: ExprDomain,
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
            kind=kind,
            domain=domain,
            dtype=dtype,
            shape=shape
        )
        self._parents_weak = weakref.WeakSet()
    # end __init__

    # region MATH_EXPR

    @property
    def rank(self) -> int:
        """Return tensor rank derived from shape."""
        return self._shape.rank
    # end def rank

    @property
    def spec(self) -> Optional[Operator]:
        """Leaves do not """
        return None
        # raise SymbolicMathRuntimeError("Leaf nodes do not have a spec.")
    # end def spec

    @property
    def op(self) -> Optional[Operator]:
        """Leaves do not """
        return None
        # raise SymbolicMathRuntimeError("Leaf nodes do not have an op.")
    # end def op

    @property
    def op_name(self) -> Optional[str]:
        """Leaves do not """
        return None
        # raise SymbolicMathRuntimeError("Leaf nodes do not have an op_name.")
    # end def op_name

    @property
    def arity(self) -> int:
        """Leaves do not """
        return 0
    # end def arity

    @property
    def children(self) -> list:
        """Leaves do not """
        return []
    # end def children

    @property
    def parents(self) -> FrozenSet[MathExpr]:
        """Leaves do not """
        return frozenset(self._parents_weak)
    # end def parents

    #
    # Evaluate and differentiate
    #

    @abstractmethod
    def eval(self) -> "Tensor":
        """Evaluate the expression in the active runtime context."""
    # end def eval

    @abstractmethod
    def diff(self, wrt: "Variable") -> "MathExpr":
        """Symbolic derivative of the expression with respect to one variable."""
    # end def diff

    #
    # Structure
    #

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

    def contains(
            self,
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
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
                raise SymbolicMathValidationError(f"Cannot find by reference if string given: var={leaf}.")
            # end if
            if isinstance(leaf, MathLeaf) and leaf is self:
                return True
            # end if
        else:
            if type(leaf) is str and leaf == self.name:
                return True
            elif isinstance(leaf, MathLeaf) and leaf.name == self.name:
                return True
            # end if
        # end if
        return False
    # en def contains

    @abstractmethod
    def contains_variable(
            self,
            variable: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """
        Concrete leaf classes must provide variable-specific lookup logic.
        """
    # end def contains_variable

    @abstractmethod
    def contains_constant(
            self,
            constant: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """
        Concrete leaf classes must provide constant-specific lookup logic.
        """
    # end def contains_constant

    #
    # Transforms
    #

    def simplify(self, options: SimplifyOptions | None = None) -> MathExpr:
        """Return this leaf unchanged."""
        return self
    # end def simplify

    def canonicalize(self) -> MathExpr:
        """Return this leaf unchanged."""
        return self
    # end def canonicalize

    def fold_constants(self) -> MathExpr:
        """Return this leaf unchanged."""
        return self
    # end def fold_constants

    def substitute(
            self,
            mapping: Mapping[MathExpr, MathExpr],
            *,
            by_ref: bool = True
    ) -> MathExpr:
        """Replace this leaf when it matches a mapping key."""
        for old_expr, new_expr in mapping.items():
            if by_ref and old_expr is self:
                return new_expr
            # end if
            if (not by_ref) and old_expr == self:
                return new_expr
            # end if
        # end for
        return self
    # end def substitute

    def renamed(self, old_name: str, new_name: str) -> List[str]:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        if self.name == old_name:
            self._name = new_name
            return [old_name]
        # end if
        return []
    # end rename

    #
    # Comparison
    #

    @abstractmethod
    def eq_tree(self, other: MathExpr) -> bool:
        """Return strict symbolic tree equality for leaves."""
    # end def eq_tree

    def equivalent(self, other: MathExpr) -> bool:
        """Return symbolic equivalence for leaves."""
        return self.eq_tree(other)
    # end def equivalent

    #
    # Boolean predicates
    #

    @abstractmethod
    def is_constant(self):
        """Does the expression contain only constant values?"""
    # end def is_constant

    @abstractmethod
    def is_variable(self):
        """Does the expression contain a variable?"""
    # end def is_variable

    def is_node(self) -> bool:
        """
        Leaves are not nodes.
        """
        return False
    # end def is_node

    def is_leaf(self) -> bool:
        """
        Leaves are leaves.
        """
        return True
    # end def is_leaf

    @abstractmethod
    def is_constant_leaf(self) -> bool:
        """
        Returns
        -------
        'bool'
            ``True`` when the expression is a leaf node and represents a constant value.
        """
    # end def is_constant_leaf

    @abstractmethod
    def is_variable_leaf(self) -> bool:
        """
        Returns
        -------
        'bool'
            ``True`` when the expression is a leaf node and represents a variable.
        """
    # end def is_variable_leaf

    @abstractmethod
    def is_pure(self) -> bool:
        """
        Check if the expression is pure, i.e. has no side effects.
        """
    # end def is_pure

    # True if behave like a scalar (rank = 1)
    def is_scalar(self) -> bool:
        """
        Check if the expression behaves like a scalar (rank = 1).
        """
        return self.rank == 1
    # end def is_scalar

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
        return False
    # end if

    # Check if the expression has children.
    def has_children(self) -> bool:
        """
        Check if the expression has children.
        """
        return False
    # end def has_children

    # Check the number of children of the expression.
    def num_children(self) -> int:
        """
        Check the number of children of the expression.
        """
        return 0
    # end def num_children

    #
    # Rules and policy
    #

    @abstractmethod
    def is_foldable(self) -> bool:
        """
        Check if the expression can be folded into a single node.
        """
    # end def is_foldable

    def fold_policy(self) -> FoldPolicy:
        """
        Get the folding policy for the expression.

        Returns
        -------
        'FoldPolicy'
            The folding policy for the expression.
        """
        return FoldPolicy.FOLDABLE
    # end def fold_policy

    #
    # Structure
    #

    def depth(self) -> int:
        """Return the depth of the node in the tree"""
        return 1
    # end def depth

    @abstractmethod
    def copy(self, deep: bool = False) -> MathExpr:
        """
        Copy the leaf node.
        """
    # end def copy

    #
    # Comparaison
    #

    def __eq__(self, other: object) -> bool:
        """Check if two MathExpr are equal."""
        return self is other
    # end def __eq_

    def __ne__(self, other: object) -> bool:
        """Check if two MathExpr are not equal."""
        return not (self is other)
    # end def __ne_

    __hash__ = MathBase.__hash__

    #
    # Representation
    #

    @abstractmethod
    def __str__(self) -> str:
        """
        Human-readable description of the leaf node.
        """
    # end def __str__

    @abstractmethod
    def __repr__(self) -> str:
        """
        Debug representation identical to :meth:`__str__`.
        """
    # end def __repr__

    # endregion MATH_EXPR

# end MathLeaf


class Variable(MathLeaf):
    """
    Leaf node bound to a runtime tensor via the active context.
    """

    def __init__(
            self,
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
            kind=ExprKind.VARIABLE,
            domain=ExprDomain.ALGEBRAIC,
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
        'Tensor'
            Tensor currently stored in the runtime context.
        """
        return get_value(self._name)
    # end def value

    @property
    def dims(self) -> Sequence[DimLike]:
        """
        Returns
        -------
        Sequence['MathExpr']
            Tuple of shape dimensions.
        """
        return self._shape.dims
    # end def dims

    @property
    def ndim(self) -> int:
        """
        Returns
        -------
        'int'
            Rank (number of axes) of the variable.
        """
        return self._shape.rank
    # end def dim

    @property
    def size(self) -> int:
        """
        Returns
        -------
        'int'
            Total number of elements (when fully defined).
        """
        return self._shape.size
    # end def size

    @property
    def n_elements(self) -> int:
        """
        Returns
        -------
        'int'
            Total number of elements (alias for ``size``).
        """
        return self._shape.size
    # end def n_elements

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
            Rank (number of axes) of the variable.
        """
        return self._shape.rank
    # end def rank

    #
    # Evaluate and differentiate
    #

    def eval(self) -> Tensor:
        """
        Evaluate this leaf node in the current context.

        Returns
        -------
        'Tensor'
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
            raise SymbolicMathLookupError(f"Variable {self._name} not found in context.")
        # end if
        if var_val.dtype != self._dtype:
            var_val = var_val.astype(self._dtype)
        # end if
        if not self._shape.equals(list(var_val.shape.dims)):
            raise SymbolicMathValidationError(
                f"Variable {self._name} shape mismatch: {var_val.shape} != {self._shape}"
            )
        # end if
        return var_val
    # end def eval

    def diff(self, wrt: MathExpr) -> MathExpr:
        """
        Compute the derivative of this leaf with respect to a variable.
        """
        if isinstance(wrt, Variable) and wrt.name == self.name:
            return Constant(
                name=rand_name(f"{self.name}_autodiff_"),
                data=Tensor.full(1, shape=self.shape.eval().tolist(), dtype=self.dtype)
            )
        else:
            return Constant(
                name=rand_name(f"{self.name}_autodiff_"),
                data=Tensor.full(0, shape=self.shape.eval().tolist(), dtype=self.dtype)
            )
        # end if
    # end def diff

    #
    # Structure
    #

    def variables(self) -> list:
        """
        Returns
        -------
        'list'
            List containing ``self``.
        """
        return [self]
    # end def variable

    def constants(self) -> List:
        """
        Returns
        -------
        'list'
            Empty list because variables do not reference constants.
        """
        return []
    # end def constants

    def contains(
            self,
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
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
        if look_for in {LeafKind.ANY, LeafKind.VARIABLE}:
            return super(Variable, self).contains(
                leaf=leaf,
                by_ref=by_ref,
                check_operator=check_operator,
                look_for=LeafKind.ANY
            )
        # end if
        return False
    # end def contains

    def contains_variable(
            self,
            variable: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return self.contains(variable, by_ref=by_ref, check_operator=check_operator, look_for=LeafKind.VARIABLE)
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        """
        Check if ``constant`` is contained in this expression.
        """
        return False
    # end def contains_constant

    #
    # Comparison
    #

    # def replace(self, old_m: MathExpr, new_m: MathExpr) -> MathExpr:
    #     """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.
    #
    #     Parameters
    #     ----------
    #     old_m: ExpressionMixin
    #         MathExpr to replace.
    #     new_m: ExpressionMixin
    #         New MathExpr replacing the old one.
    #     """
    #     # TODO: decide whether leaf-level replace should mutate in-place
    #     # (rename/value swap) or be handled exclusively by parent nodes.
    #     raise SymbolicMathNotImplementedError(
    #         "TODO: implement Variable.replace(...) semantics for leaf substitution."
    #     )
    # # end def replace

    def renamed(self, old_name: str, new_name: str) -> List[str]:
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
            return [old_name]
        # end if
        return []
    # end rename

    def eq_tree(self, other: MathExpr) -> bool:
        """
        Check if two MathExpr trees are equal in structure and value.
        """
        return (
            isinstance(other, Variable)
            and self._name == other._name
            and self._dtype == other._dtype
            and self._shape == other._shape
        )
    # end def eq_tree

    # Match a symbolic expression to a pattern.
    # - `pattern` is a string or a regular expression.
    def match(
            self,
            pattern: ExprPattern
    ) -> MatchResult:
        """
        Match a symbolic expression to a pattern.
        """
        # Any, but check shape and dtype
        any_match = self._match_any_pattern(pattern)
        if any_match:
            return any_match
        # end if

        if isinstance(pattern, VariablePattern):
            if pattern.var_name and pattern.var_name != self.name:
                return MatchResult.failed()
            # end if
            if not self._match_dtype(pattern.dtype) or not self._match_shape(pattern.shape):
                return MatchResult.failed()
            # end if
            return MatchResult.success(self._match_return(pattern))
        # end if

        return MatchResult.failed()
    # end def match

    #
    # Boolean checks
    #

    def is_constant(self) -> bool:
        """Does the expression contain only constant values?"""
        return False
    # end def is_constant

    def is_variable(self) -> bool:
        """Does the expression contain a variable?"""
        return True
    # end def is_variable

    def is_constant_leaf(self) -> bool:
        """Does the expression contain only constant values?"""
        return False
    # end def is_constant_leaf

    def is_variable_leaf(self) -> bool:
        """Does the expression contain a variable?"""
        return True
    # end def is_variable_leaf

    def is_pure(self) -> bool:
        """Is the expression pure?"""
        return False
    # end def is_pure

    #
    # Rules and policy
    #

    def is_foldable(self) -> bool:
        """Is the expression foldable?"""
        return False
    # end def is_foldable

    def fold_policy(self) -> FoldPolicy:
        """Get the folding policy for the expression."""
        return FoldPolicy.NON_FOLDABLE
    # end def fold_policy

    #
    # Structure
    #

    def copy(
            self,
            deep: bool = False
    ) -> 'Variable':
        """
        Create a shallow copy with a new name.

        Parameters
        ----------
        deep : bool, optional

        Returns
        -------
        Variable
            New variable sharing dtype/shape metadata.
        """
        return Variable(
            name=self._name,
            dtype=self._dtype,
            shape=self._shape.copy(),
        )
    # end def copy

    def __str__(self):
        """
        Returns
        -------
        'str'
            Human-readable description of the variable.
        """
        return self.name
    # end __str__

    def __repr__(self):
        """
        Returns
        -------
        'str'
            Debug representation identical to :meth:`__str__`.
        """
        return f"variable({self.name}, dtype={self._dtype}, shape={self._shape})"
    # end __repr__

    # endregion MATH_EXPR

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
            kind=ExprKind.CONSTANT,
            domain=ExprDomain.ALGEBRAIC,
            dtype=data.dtype,
            shape=Shape(dims=data.shape.dims)
        )
        if data.shape != self._shape:
            raise SymbolicMathValidationError(f"Constant shape mismatch: {data.shape} != {self._shape}")
        # end if
        if data.dtype != self._dtype:
            raise SymbolicMathValidationError(f"Constant dtype mismatch: {data.dtype} != {self._dtype}")
        # end if
        self._data = data
    # end __init__

    # region PROPERTIES

    @property
    def value(self) -> Tensor:
        """
        Returns
        -------
        'Tensor'
            Stored tensor value.
        """
        return self._data
    # end def value

    @property
    def dims(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        tuple[int, ...]
            Tuple of dimensions.
        """
        return tuple(self._shape.dims)
    # end def dims

    @property
    def ndim(self) -> int:
        """
        Returns
        -------
        'int'
            Rank (number of axes).
        """
        return self._shape.rank
    # end def dim

    @property
    def size(self) -> int:
        """
        Returns
        -------
        'int'
            Total number of elements.
        """
        return self._shape.size
    # end def size

    @property
    def n_elements(self) -> int:
        """
        Returns
        -------
        'int'
            Total number of elements.
        """
        return self._shape.size
    # end def n_elements

    # endregion PROPERTIES

    # region MATH_EXPR

    @property
    def rank(self) -> int:
        """
        Returns
        -------
        'int'
            Rank (number of axes).
        """
        return self._shape.rank
    # end def rank

    #
    # Evaluate and differentiate
    #

    def eval(self) -> Tensor:
        """
        Returns
        -------
        'Tensor'
            Copy of the stored tensor.
        """
        return self._data.copy()
    # end def eval

    def diff(self, wrt: MathExpr) -> MathExpr:
        """
        Compute the derivative of this leaf with respect to a variable.
        """
        return Constant(
            name=rand_name(f"{self.name}_autodiff"),
            data=Tensor.zeros_like(self.value, self.dtype)
        )
    # end def diff

    #
    # Structure
    #

    def variables(self) -> list:
        """
        Returns
        -------
        'list'
            Empty list because constants do not reference variables.
        """
        return []
    # end def variable

    def constants(self) -> List:
        """
        Returns
        -------
        'list'
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
            leaf: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: LeafKind = LeafKind.ANY
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
        if look_for in {LeafKind.ANY, LeafKind.CONSTANT}:
            return super(Constant, self).contains(
                leaf=leaf,
                by_ref=by_ref,
                check_operator=check_operator,
                look_for=LeafKind.ANY
            )
        # end if
        return False
    # en def contains

    def contains_variable(
            self,
            variable: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return False
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        if constant is self:
            return True
        # end if
        if isinstance(constant, Constant) and constant.name == self.name and constant.value == self.value:
            return True
        # end if
        return False
    # end def contains_constant

    #
    # Transforms
    #

    def renamed(self, old_name: str, new_name: str) -> List[str]:
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
            return [old_name]
        # end if
        return []
    # end rename

    #
    # Comparison
    #

    def eq_tree(self, other: MathExpr) -> bool:
        """Check if two constants are equal in terms of name, dtype, shape, and data.

        Returns
        -------
        'bool'
            True if the constants are equal, False otherwise.
        """
        return (
            isinstance(other, Constant)
            and self._name == other._name
            and self._dtype == other._dtype
            and self._shape == other._shape
            and self._data == other._data
        )
    # end def eq_tree

    # Match a symbolic expression to a pattern.
    # - `pattern` is a string or a regular expression.
    def match(
            self,
            pattern: ExprPattern
    ) -> MatchResult:
        """
        Match a symbolic expression to a pattern.
        """
        # Any, but check shape and dtype
        any_match = self._match_any_pattern(pattern)
        if any_match:
            return any_match
        # end if

        if isinstance(pattern, ConstantPattern):
            if pattern.const_name and pattern.const_name != self.name:
                return MatchResult.failed()
            # end if
            if pattern.value is not None and pattern.value != self.value:
                return MatchResult.failed()
            # end if
            if not self._match_dtype(pattern.dtype) or not self._match_shape(pattern.shape):
                return MatchResult.failed()
            # end if
            return MatchResult.success(self._match_return(pattern))
        # end if

        return MatchResult.failed()
    # end def match

    #
    # Boolean checks
    #

    def is_constant_leaf(self) -> bool:
        """Check if the node is a constant leaf.

        Returns
        -------
        'bool'
            True if the node is a constant leaf, False otherwise.
        """
        return True
    # end def is_constant_leaf

    def is_variable_leaf(self) -> bool:
        """
        Determines if a variable is a leaf in a computational graph.

        A leaf variable is typically considered as one that is not a result of
        any operation in a computational graph, often serving as input data
        or a parameter. This function checks whether the current variable meets
        these criteria and returns the result.

        Returns
        -------
        'bool'
            Indicates whether the variable is a leaf node.
        """
        return False
    # end def is_variable_leaf

    def is_pure(self) -> bool:
        """
        Determines if the current instance meets the purity criteria.

        This method evaluates whether the instance satisfies a predefined set
        of conditions to be considered "pure." The specifics of these purity
        criteria are determined by the implementation details of the class.

        Returns
        -------
        'bool'
            True if the instance is pure, according to the defined criteria, otherwise False.
        """
        return True
    # end def is_pure

    #
    # Rules and policy
    #

    def is_foldable(self) -> bool:
        """
        Determines if the current instance can be folded into a single operation.

        Returns
        -------
        'bool'
            True if the instance can be folded, otherwise False.
        """
        return True
    # end def is_foldable

    def fold_policy(self) -> FoldPolicy:
        """
        Returns the folding policy for the current instance.
        """
        return FoldPolicy.FOLDABLE
    # end def fold_policy

    #
    # Structure
    #

    def copy(
            self,
            deep: bool = False
    ) -> 'Constant':
        """
        Create a copy of the constant with a new name.

        Parameters
        ----------
        deep : bool, optional

        Returns
        -------
        Constant
            New constant carrying a copy of the tensor.
        """
        return Constant(
            name=self._name,
            data=self._data.copy()
        )
    # end def copy

    # Check if the expression has children.
    def has_children(self) -> bool:
        """
        Check if the expression has children.
        """
        return False
    # end def has_children

    # Check the number of children of the expression.
    def num_children(self) -> int:
        """
        Check the number of children of the expression.
        """
        return 0
    # end def num_children

    #
    # Representation
    #

    def __str__(self):
        """
        Returns
        -------
        'str'
            Human-readable description of the constant.
        """
        return self.value.__str__()
    # end __str__

    def __repr__(self):
        """
        Returns
        -------
        'str'
            Debug representation identical to :meth:`__str__`.
        """
        return f"constant({self.name}, dtype={self._dtype}, shape={self._shape})"
    # end __repr__

    # endregion MATH_EXPR

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
            raise SymbolicMathValidationError(f"Constant shape mismatch: {data.shape} != {self._shape}")
        # end if
        if data.dtype != self._dtype:
            raise SymbolicMathValidationError(f"Constant dtype mismatch: {data.dtype} != {self._dtype}")
        # end if
        self._data = data
    # end def set

    # endregion PUBLIC

    # region STATIC

    @staticmethod
    def create(name: str, data: Tensor):
        """Create a new constant node."""
        return Constant(
            name=name,
            data=data
        )
    # end def create

    @staticmethod
    def new(data, dtype: DType = DType.R):
        """Create a new constant node with a random name."""
        return Constant(
            name=rand_name("const_"),
            data=Tensor(data=data, dtype=dtype)
        )
    # end def new

    # endregion STATIC

# end class Constant
