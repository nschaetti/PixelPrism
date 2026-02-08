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


from typing import Protocol
from typing import Tuple, Union, Sequence, List, Optional, Dict, TypeAlias
import numpy as np


__all__ = [
    "NumberLike",
    "ScalarLike",
    "NumberListLike",
    "Index",
    "DimExpr",
    "MathExpr",
    "TensorLike",
    "TensorDim",
    "TensorDims",
]


# Type numeric
NumberLike = float | int | bool | complex | np.number
ScalarLike = int | float | np.number | bool | complex
NumberListLike: TypeAlias = list[Union[ScalarLike, "NumberListLike"]]

# Tensor data
TensorLike: TypeAlias = Union[NumberLike, NumberListLike, np.ndarray]

# Tensor dimension
TensorDim = int
TensorDims = Union[Tuple[TensorDim, ...], List[TensorDim], Sequence[TensorDim]]


# Represents a valid index
Index = Union[
    int,
    slice,
    Sequence[int],
    Tuple[Union[int, slice], ...],
]


class DimExpr(Protocol):
    """
    TODO: Add documentation.
    """
    def eval(self) -> 'Tensor': ...
    def variables(self) -> List['Variable']: ...
    def constants(self) -> List['Constant']: ...
    def contains(
            self,
            leaf: Union[str, 'PredicateMixin'],
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
    def replace(self, old_m: 'PredicateMixin', new_m: 'PredicateMixin'): ...
    def rename(self, old_name: str, new_name: str) -> Dict[str, str]: ...
    def is_constant(self) -> bool: ...
    def is_variable(self) -> bool: ...
# end class DimExpr


class MathExpr(Protocol):
    def eval(self) -> 'Tensor': ...
    def diff(self, wrt: 'Variable') -> 'MathExpr': ...
    def variables(self) -> List['Variable']: ...
    def constants(self) -> List['Constant']: ...
    def contains(
            self,
            leaf: Union[str, 'PredicateMixin'],
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
    def replace(self, old_m: 'PredicateMixin', new_m: 'PredicateMixin'): ...
    def rename(self, old_name: str, new_name: str) -> Dict[str, str]: ...
    def is_constant(self) -> bool: ...
    def is_variable(self) -> bool: ...
# end class MathExpr

