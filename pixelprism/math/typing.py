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
from typing import Protocol, runtime_checkable
from typing import TYPE_CHECKING, Tuple, Union, Sequence, List, Optional, Dict, TypeAlias
from typing import ClassVar
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
    "DimExpr",
    "MathExpr",
    "TensorLike",
    "DimLike",
    "DimInt",
    "DimsInt",
    "Operand",
    "Operands",
    "Operator"
]


# Type numeric
ScalarLike = float | int | np.number | bool | complex
ScalarListLike: TypeAlias = list[Union[ScalarLike, "ScalarListLike"]]


# Tensor data
TensorLike: TypeAlias = Union[ScalarLike, ScalarListLike, np.ndarray]


# Represents a valid index
Index = Union[
    int,
    slice,
    Sequence[int],
    Tuple[Union[int, slice], ...],
]


# Protocol for mathematical expressions
@runtime_checkable
class MathExpr(Protocol):
    # Output shape
    @property
    def shape(self) -> "Shape": ...
    @property
    def dtype(self) -> "DType": ...
    @property
    def name(self) -> str: ...

    # Evaluate and differentiate
    def eval(self) -> "Tensor": ...
    def diff(self, wrt: 'Variable') -> 'MathExpr': ...

    # Structure
    def variables(self) -> Sequence['Variable']: ...
    def constants(self) -> Sequence['Constant']: ...
    def contains(
            self,
            leaf: Union[str, 'MathExpr'],
            by_ref: bool = False,
            check_operator: bool = True,
            look_for: Optional[str] = None
    ) -> bool: ...
    def contains_variable(
            self,
            variable: Union[str, 'Variable'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool: ...
    def contains_constant(
            self,
            constant: Union[str, 'Constant'],
            by_ref: bool = False,
            check_operator: bool = True
    ) -> bool: ...
    def rename(self, old_name: str, new_name: str) -> Dict[str, str]: ...
    def is_constant(self) -> bool: ...
    def is_variable(self) -> bool: ...
    def is_node(self) -> bool: ...
    def is_leaf(self) -> bool: ...
    def depth(self) -> int: ...
    def copy(self, deep: bool = False) -> 'MathExpr': ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
# end class MathExpr


# Operands are the operands of an operator.
# Can be a single operand or a tuple of operands.
# Can be a MathNode, Variable, or Constant, then a MathExpr.
Operand: TypeAlias = "MathExpr"
Operands = Sequence[Operand]


# Protocol for mathematical operators
@runtime_checkable
class Operator(Protocol):
    # Class variables
    ARITY: ClassVar[int]
    IS_VARIADIC: ClassVar[bool]
    IS_DIFF: ClassVar[bool]
    NAME: ClassVar[str]

    # Properties
    @property
    def name(self) -> str: ...
    @property
    def arity(self) -> int: ...
    @arity.setter
    def arity(self, value: int) -> None: ...
    @property
    def is_variadic(self) -> bool: ...
    @property
    def is_diff(self) -> bool: ...

    # Evaluate and differentiate
    def eval(self, operands: Operands, **kwargs) -> "Tensor": ...
    def diff(self, wrt: "Variable", operands: Operands) -> "MathExpr": ...

    # Structure
    def contains(self, expr: "MathExpr", by_ref: bool = False, look_for: Optional[str] = None) -> bool: ...

    # Infer shapes and dtypes
    def infer_dtype(self, operands: Operands) -> "DType": ...
    def infer_shape(self, operands: Operands) -> "Shape": ...

    # Checks
    def check_operands(self, operands: Operands) -> bool: ...
    def check_parameters(self, **kwargs) -> bool: ...
    def check_shapes(self, operands: Operands) -> bool: ...

    @classmethod
    def check_arity(cls, operands: Operands): ...

    @classmethod
    def create_node( cls, operands: Operands, **kwargs) -> "MathNode": ...

    # Representation
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
# end class Operator


# Dimensions
DimExpr: TypeAlias = "MathExpr"
DimInt = int
DimsInt = Tuple[int, ...]
DimLike: TypeAlias = Union[int, "MathExpr"]
