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
"""
Structure operator implementations.
"""

# Imports
from abc import ABC
from typing import Optional, Sequence, Union, Any, List

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor, einsum
from .base import Operands, Operand, operator_registry, Operator, ParametricOperator

__all__ = [

]


class StructureOperator(Operator, ABC):
    """
    Linear algebra operator.
    """

    def contains(
            self,
            expr: "MathExpr",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric operators must implement contains(..).")
    # end def contains

    def check_parameters(self, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

# end class StructureOperator


class Getitem(StructureOperator):
    """Getitem operator."""

    def __init__(self, indices: Union[int, Sequence[int]], **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
    # end def __init__

