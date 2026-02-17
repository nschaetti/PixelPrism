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
from typing import List, Optional, Tuple, Union, Dict, Mapping, Sequence
import weakref

from .math_base import MathBase

from .math_exceptions import (
    SymbolicMathNotImplementedError,
    SymbolicMathValidationError,
    SymbolicMathLookupError
)
# from .math_node import MathNode
from .mixins import PredicateMixin, ExprOperatorsMixin
from .dtype import DType, TypeLike, create
from .shape import Shape, ShapeLike
from .tensor import Tensor, TensorLike, tensor
from .context import get_value
from .random import rand_name
from .typing import MathExpr, LeafKind, SimplifyOptions


__all__ = [
    "MathLeaf",
    "Variable",
    "Constant",
]


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
    PredicateMixin,
    ExprOperatorsMixin,
    MathExpr,
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
        super(MathLeaf, self).__init__(name=name, dtype=dtype, shape=shape)
        self._parents_weak = weakref.WeakSet()
    # end __init__

    # region MATH_EXPR

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

    def depth(self) -> int:
        """Return the depth of the node in the tree"""
        return 1
    # end def depth

    @property
    def rank(self) -> int:
        """Return tensor rank derived from shape."""
        return self._shape.rank
    # end def rank

    def is_constant(self):
        """Does the expression contain only constant values?"""
        raise SymbolicMathNotImplementedError("Class must override is_constant.")
    # end def is_constant

    def is_variable(self):
        """Does the expression contain a variable?"""
        raise SymbolicMathNotImplementedError("Class must override is_variable.")
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
        TODO: concrete leaf classes must provide variable-specific lookup logic.
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
        TODO: concrete leaf classes must provide constant-specific lookup logic.
        """
    # end def contains_constant

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

    def replace(self, old_m: MathExpr, new_m: MathExpr) -> MathExpr:
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
        raise SymbolicMathNotImplementedError("Leaf nodes do not support replace.")
    # end def replace

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
        """Rename all variables/constants named ``old_name`` with ``new_name`` in the tree. The replacement is in-place.

        Parameters
        ----------
        old_name : str
            Name of the variable/constant to rename.
        new_name: str
            New name for the variable/constant.
        """
        raise SymbolicMathNotImplementedError("Leaf nodes do not support rename.")
    # end rename

    def eq_tree(self, other: MathExpr) -> bool:
        """Return strict symbolic tree equality for leaves."""
        return self is other
    # end def eq_tree

    def equivalent(self, other: MathExpr) -> bool:
        """Return symbolic equivalence for leaves."""
        return self.eq_tree(other)
    # end def equivalent

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

    def copy(self, deep: bool = False) -> "MathLeaf":
        """
        TODO: unify leaf copy contract (`deep`) with Variable/Constant copy APIs.
        """
        # TODO: return type should likely be covariant and preserve leaf subtype.
        return self
    # end def copy

    def __str__(self) -> str:
        """
        TODO: subclasses should provide domain-specific string representations.
        """
        return self.__repr__()
    # end def __str__

    def __repr__(self) -> str:
        """
        TODO: subclasses should override for richer debug output.
        """
        return f"{self.__class__.__name__}({self.name})"
    # end def __repr__

    # endregion MATH_EXPR

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
    def dims(self) -> Sequence[MathExpr]:
        """
        Returns
        -------
        Sequence[MathExpr]
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

    # region MATH_EXPR

    # region EVALUABLE_MIXIN

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
        if var_val.shape != self._shape:
            raise SymbolicMathValidationError(
                f"Variable {self._name} shape mismatch: {var_val.shape} != {self._shape}"
            )
        # end if
        return var_val
    # end def eval

    # endregion EVALUABLE_MIXIN

    # region PREDICATE_MIXIN

    def contains_variable(
            self,
            variable: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        # TODO: keep dedicated leaf-level specialization if variable lookup
        # semantics diverge from generic `contains(..., look_for="var")`.
        return self.contains(variable, by_ref=by_ref, check_operator=check_operator, look_for="var")
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, MathExpr],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        # TODO: if variables can capture constants via metadata later, update.
        return False
    # end def contains_constant

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

    def replace(self, old_m: MathExpr, new_m: MathExpr) -> MathExpr:
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: PredicateMixin
            MathExpr to replace.
        new_m: PredicateMixin
            New MathExpr replacing the old one.
        """
        # TODO: decide whether leaf-level replace should mutate in-place
        # (rename/value swap) or be handled exclusively by parent nodes.
        raise SymbolicMathNotImplementedError(
            "TODO: implement Variable.replace(...) semantics for leaf substitution."
        )
    # end def replace

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
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
        return self
    # end rename

    # endregion PREDICATE_MIXIN

    # region DIFFERENTIABLE_MIXIN

    def diff(self, wrt: MathExpr) -> MathExpr:
        """
        Compute the derivative of this leaf with respect to a variable.
        """
        if wrt is self:
            return Constant(
                name=rand_name(f"{self.name}_autodiff_"),
                data=Tensor(data=1, dtype=self._dtype)
            )
        else:
            return Constant(
                name=rand_name(f"{self.name}_autodiff_"),
                data=Tensor(data=0, dtype=self._dtype)
            )
        # end if
    # end def diff

    # endregion DIFFERENTIABLE_MIXIN

    # region MATH_EXPR_MISC

    def copy(
            self,
            deep: bool = False,
            name: Optional[str] = None,
    ) -> 'Variable':
        """
        Create a shallow copy with a new name.

        Parameters
        ----------
        name : str, optional
            Name assigned to the clone. Uses current name when ``None``.
        deep : bool, optional
            TODO: deep copy semantics are currently unused for leaves.

        Returns
        -------
        Variable
            New variable sharing dtype/shape metadata.
        """
        return Variable(
            name=name or self._name,
            dtype=self._dtype,
            shape=self._shape.copy(),
        )
    # end def copy

    # endregion MATH_EXPR_MISC

    # region MATH_EXPR_MISC

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

    # endregion MATH_EXPR_MISC

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
        Tensor
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

    # region MATH_EXPR

    # region EVALUABLE_MIXIN

    def eval(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            Copy of the stored tensor.
        """
        return self._data.copy()
    # end def eval

    # endregion EVALUABLE_MIXIN

    # region PREDICATE_MIXIN

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
        if isinstance(constant, Constant):
            return constant is self
        elif isinstance(constant, Variable):
            return False
        elif isinstance(constant, str):
            return constant == self._name
        # end if
        return False
    # end def contains_constant

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

    def replace(self, old_m: MathExpr, new_m: MathExpr) -> MathExpr:
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: MathNode
            MathExpr to replace.
        new_m: MathNode
            New MathExpr replacing the old one.
        """
        # TODO: decide whether replacing a constant leaf should mutate this
        # instance or be performed at parent-level by child substitution.
        raise SymbolicMathNotImplementedError(
            "TODO: implement Constant.replace(...) semantics for leaf substitution."
        )
    # end def replace

    def renamed(self, old_name: str, new_name: str) -> MathExpr:
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
        return self
    # end rename

    # endregion PREDICATE_MIXIN

    # region DIFFERENTIABLE_MIXIN

    def diff(self, wrt: MathExpr) -> MathExpr:
        """
        Compute the derivative of this leaf with respect to a variable.
        """
        return Constant(
            name=rand_name(f"{self.name}_autodiff_"),
            data=Tensor(data=0, dtype=self._dtype)
        )
    # end def diff

    # endregion DIFFERENTIABLE_MIXIN

    # region MATH_EXPR_MISC

    def copy(
            self,
            deep: bool = False,
            name: Optional[str] = None,
    ) -> 'Constant':
        """
        Create a copy of the constant with a new name.

        Parameters
        ----------
        name : str, optional
            Identifier assigned to the clone. Uses current name when ``None``.
        deep : bool, optional
            TODO: deep copy semantics are currently unused for leaves.

        Returns
        -------
        Constant
            New constant carrying a copy of the tensor.
        """
        return Constant(
            name=name or self._name,
            data=self._data.copy()
        )
    # end def copy

    # endregion MATH_EXPR_MISC

    # region MATH_EXPR_MISC

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

    # endregion MATH_EXPR_MISC

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
