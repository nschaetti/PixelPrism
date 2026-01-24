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
Boolean comparison elementwise operators.
"""

from abc import ABC
from typing import Sequence, Any, Optional, Union

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from .base import Operands, Operator, ParametricOperator, operator_registry
from .elementwise import UnaryElementwiseOperator, UnaryElementwiseParametricOperator

__all__ = [
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
]


class ComparisonOperator(Operator, ABC):
    pass
# end class ComparisonOperator


class _BinaryComparisonOperator(ComparisonOperator):
    """Shared binary comparison implementation."""

    ARITY = 2
    IS_BINARY = False
    TENSOR_METHOD: str

    def contains(self, expr: "MathExpr", by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        ref_operand = operands[0]
        for op in operands:
            if ref_operand.shape != op.shape:
                return False
            # end if
        # end for
        return True
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (a, b) = operands
        left = a.eval()
        right = b.eval()
        method = getattr(left, self.TENSOR_METHOD)
        return method(right)
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.BOOL
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].shape.copy()
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].shape == operands[1].shape
    # end def check_shapes

    def __str__(self) -> str:
        return f"{self.NAME}(a, b)"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a, b)"
    # end def __repr__

class Eq(_BinaryComparisonOperator):
    """Element-wise equality operator."""

    NAME = "eq"
    TENSOR_METHOD = "equal"
# end class Eq


class Ne(_BinaryComparisonOperator):
    """Element-wise inequality operator."""

    NAME = "ne"
    TENSOR_METHOD = "not_equal"
# end class Ne


class Lt(_BinaryComparisonOperator):
    """Element-wise less-than operator."""

    NAME = "lt"
    TENSOR_METHOD = "less"
# end class Lt


class Le(_BinaryComparisonOperator):
    """Element-wise less-than-or-equal operator."""

    NAME = "le"
    TENSOR_METHOD = "less_equal"
# end class Le


class Gt(_BinaryComparisonOperator):
    """Element-wise greater-than operator."""

    NAME = "gt"
    TENSOR_METHOD = "greater"
# end class Gt


class Ge(_BinaryComparisonOperator):
    """Element-wise greater-than-or-equal operator."""

    NAME = "ge"
    TENSOR_METHOD = "greater_equal"
# end class Ge


# Register operators
operator_registry.register(Eq)
operator_registry.register(Ne)
operator_registry.register(Lt)
operator_registry.register(Le)
operator_registry.register(Gt)
operator_registry.register(Ge)
