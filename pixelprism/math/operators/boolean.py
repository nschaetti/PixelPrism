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
from typing import Sequence, Any, Optional

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from ..math_expr import MathNode
from .base import Operands, Operator, operator_registry

__all__ = [
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "Not",
    "Any",
    "All",
    "And",
    "Or",
    "Xor",
]


class ComparisonOperator(Operator, ABC):
    pass
# end class ComparisonOperator


class _BinaryComparisonOperator(ComparisonOperator):
    """Shared binary comparison implementation."""

    ARITY = 2
    IS_BINARY = False
    TENSOR_METHOD: str

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        ref_operand = operands[0]
        for op in operands:
            if ref_operand.input_shape != op.input_shape:
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

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.B
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].input_shape.copy()
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].input_shape == operands[1].input_shape
    # end def check_shapes

    def __str__(self) -> str:
        return f"{self.NAME}(a, b)"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a, b)"
    # end def __repr__


class _BinaryBooleanComparisonOperator(_BinaryComparisonOperator, ABC):
    """Binary operators that only accept boolean tensors."""

    def _ensure_boolean_operands(self, operands: Operands) -> None:
        for operand in operands:
            if operand.dtype != DType.B:
                raise TypeError(f"{self.NAME} expects boolean tensors, got {operand.dtype}.")
            # end if
        # end for
    # end def _ensure_boolean_operands

    def check_operands(self, operands: Operands) -> bool:
        return super().check_operands(operands) and all(op.dtype == DType.B for op in operands)
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        self._ensure_boolean_operands(operands)
        return super()._eval(operands, **kwargs)
    # end def _eval
# end class _BinaryBooleanComparisonOperator


class _UnaryBooleanComparisonOperator(ComparisonOperator, ABC):
    """Shared unary boolean comparison implementation."""

    ARITY = 1
    TENSOR_METHOD: str

    def check_operands(self, operands: Operands) -> bool:
        operand, = operands
        return operand.dtype == DType.B
    # end def check_operands

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    def _ensure_boolean_operand(self, operand: MathNode) -> None:
        if operand.dtype != DType.B:
            raise TypeError(f"{self.NAME} expects boolean tensors, got {operand.dtype}.")
        # end if
    # end def _ensure_boolean_operand

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (expr,) = operands
        self._ensure_boolean_operand(expr)
        tensor = expr.eval()
        method = getattr(tensor, self.TENSOR_METHOD)
        return method()
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support backward.")
    # end def _backward

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.B
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        operand, = operands
        return self._result_shape(operand)
    # end def infer_shape

    def _result_shape(self, operand: MathNode) -> Shape:
        return operand.input_shape
    # end def _result_shape

    def check_shapes(self, operands: Operands) -> bool:
        return True
    # end def check_shapes

    def __str__(self) -> str:
        return f"{self.NAME}(a)"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a)"
    # end def __repr__


class _ScalarBooleanComparisonOperator(_UnaryBooleanComparisonOperator, ABC):
    """Boolean reductions that always return scalars."""

    def _result_shape(self, operand: MathNode) -> Shape:
        return Shape(dims=())
    # end def _result_shape

# end class _ScalarBooleanComparisonOperator

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


class Not(_UnaryBooleanComparisonOperator):
    """Element-wise logical negation for boolean tensors."""

    NAME = "not"
    TENSOR_METHOD = "logical_not"
# end class Not


class Any(_ScalarBooleanComparisonOperator):
    """Check whether any element in a boolean tensor is True."""

    NAME = "any"
    TENSOR_METHOD = "any"
# end class Any


class All(_ScalarBooleanComparisonOperator):
    """Check whether every element in a boolean tensor is True."""

    NAME = "all"
    TENSOR_METHOD = "all"
# end class All


class And(_BinaryBooleanComparisonOperator):
    """Element-wise logical AND."""

    NAME = "and"
    TENSOR_METHOD = "logical_and"
# end class And


class Or(_BinaryBooleanComparisonOperator):
    """Element-wise logical OR."""

    NAME = "or"
    TENSOR_METHOD = "logical_or"
# end class Or


class Xor(_BinaryBooleanComparisonOperator):
    """Element-wise logical XOR."""

    NAME = "xor"
    TENSOR_METHOD = "logical_xor"
# end class Xor


# Register operators
operator_registry.register(Eq)
operator_registry.register(Ne)
operator_registry.register(Lt)
operator_registry.register(Le)
operator_registry.register(Gt)
operator_registry.register(Ge)
operator_registry.register(Not)
operator_registry.register(Any)
operator_registry.register(All)
operator_registry.register(And)
operator_registry.register(Or)
operator_registry.register(Xor)
