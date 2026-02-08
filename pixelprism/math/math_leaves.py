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
from typing import List, Optional, Tuple, Union, Dict
import numpy as np

from .math_base import MathBase

from .math_exceptions import (
    SymbolicMathNotImplementedError,
    SymbolicMathValidationError,
    SymbolicMathLookupError
)
# from .math_node import MathNode
from .mixins import DifferentiableMixin, PredicateMixin, EvaluableMixin
from .dtype import DType
from .shape import Shape
from .tensor import Tensor
from .context import get_value
from .random import rand_name


__all__ = [
    "MathLeaf",
    "Variable",
    "Constant",
]


# An expression which does not contain sub-expressions
class MathLeaf(MathBase, DifferentiableMixin, EvaluableMixin, PredicateMixin, ABC):
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
        super(MathLeaf, self).__init__(name=name, dtype=dtype, shape=shape)
    # end __init__

    # region PREDICATE_MIXIN

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
            leaf: Union[str, PredicateMixin],
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

    def replace(self, old_m: PredicateMixin, new_m: PredicateMixin):
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

    def rename(self, old_name: str, new_name: str) -> Dict[str, str]:
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

    # endregion PREDICATE_MIXIN

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

    # region EVALUABLE_MIXIN

    def eval(self) -> Tensor:
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
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        pass
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        pass
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
            leaf: Union[str, PredicateMixin],
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
    # end def contains

    def replace(self, old_m: PredicateMixin, new_m: PredicateMixin):
        """Replace all occurrences of ``old`` with ``new`` in the tree. The replacement is in-place and by occurrence.

        Parameters
        ----------
        old_m: PredicateMixin
            MathExpr to replace.
        new_m: PredicateMixin
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

    # endregion PREDICATE_MIXIN

    # region DIFFERENTIABLE_MIXIN

    def diff(self, wrt: Variable) -> MathBase:
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

    # region PUBLIC

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
            dtype=self._dtype,
            shape=self._shape.copy(),
        )
    # end def copy

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
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool:
        return False
    # end def contains_variable

    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
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
            leaf: Union[str, PredicateMixin],
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

    def replace(self, old_m: PredicateMixin, new_m: PredicateMixin):
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

    # endregion PREDICATE_MIXIN

    # region DIFFERENTIABLE_MIXIN

    def diff(self, wrt: Variable) -> MathBase:
        """
        Compute the derivative of this leaf with respect to a variable.
        """
        return Constant(
            name=rand_name(f"{self.name}_autodiff_"),
            data=Tensor(data=0, dtype=self._dtype)
        )
    # end def diff

    # endregion DIFFERENTIABLE_MIXIN

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

