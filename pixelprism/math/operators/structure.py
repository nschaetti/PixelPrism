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
from ..math_expr import SliceExpr, MathNode
from .base import Operands, Operand, operator_registry, Operator, ParametricOperator

__all__ = [

]


class StructureOperator(Operator, ABC):
    """
    Linear algebra operator.
    """

    def contains(
            self,
            expr: "MathNode",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric operators must implement contains(..).")
    # end def contains

    def check_parameters(self, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        raise NotImplementedError("Parametric operators must implement check_parameters(..).")
    # end def check_shapes

# end class StructureOperator


class Getitem(StructureOperator):
    """Getitem operator."""

    def __init__(self, indices: List[Union[SliceExpr, int]], **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
    # end def __init__

    # region PUBLIC

    def check_parameters(self, **kwargs) -> bool:
        def _check_slice(s: SliceExpr) -> bool:
            return s.step is None or s.step.eval() != 0
        # end def _check_slice
        return all([
            _check_slice(o)
            for o in self._indices
        ])
    # end for

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        pass
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        for n_i, i in enumerate(self._indices):
            start = i.start.eval() if isinstance(i, SliceExpr) else i
            if start < -operands[0].shape[n_i] or start >= operands[0].shape[n_i]:
                return False
            # end if
        # end for
        return True
    # end def check_shapes

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return any([s.contains(expr, by_ref=by_ref, look_for=look_for) for s in self._indices])
    # end def contains

    # endregion PUBLIC

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        indices = list()
        for i in self._indices:
            indices.append(i.eval() if isinstance(i, SliceExpr) else i)
        # end for
        return operands[0].eval()[*indices]
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("GetItem does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class GetItem

