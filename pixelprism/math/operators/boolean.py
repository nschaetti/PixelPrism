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
from typing import Sequence, Optional

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from ..math_node import MathNode
from ..typing import MathExpr, LeafKind, OperatorSpec, AritySpec, OpAssociativity
from .base import Operands, OperatorBase, operator_registry

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


class ComparisonOperator(OperatorBase, ABC):
    def check_parameters(self, **kwargs) -> bool:
        return True
    # end def check_parameters
# end class ComparisonOperator


class _BinaryComparisonOperator(ComparisonOperator):
    """Shared binary comparison implementation."""

    ARITY = 2
    IS_BINARY = False
    NAME = "binary_comparison"
    TENSOR_METHOD: str

    def contains(self, expr: MathExpr, by_ref: bool = False, look_for: LeafKind = LeafKind.ANY) -> bool:
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

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support backward.")
    # end def _backward

    def _needs_parentheses(self, child: MathExpr, is_right_child: bool) -> None:
        return None
    # end def _needs_parentheses

    def _needs_parentheses_bool(self, child: MathExpr, is_right_child: bool) -> bool:
        if hasattr(child, "op") and child.op is not None:
            child_prec = child.op.SPEC.precedence
            self_prec = self.SPEC.precedence
            if child_prec < self_prec:
                return True
            if child_prec == self_prec:
                if self.SPEC.associativity == OpAssociativity.LEFT and is_right_child:
                    return self.name != child.op.SPEC.name
                if self.SPEC.associativity == OpAssociativity.RIGHT and not is_right_child:
                    return self.name != child.op.SPEC.name
            # end if
        # end if
        return False
    # end def _needs_parentheses_bool

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.B
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].shape.copy()
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].shape == operands[1].shape
    # end def check_shapes

    def print(self, operands: Operands, **kwargs) -> str:
        left_need = self._needs_parentheses_bool(child=operands[0], is_right_child=False)
        right_need = self._needs_parentheses_bool(child=operands[1], is_right_child=True)
        left_str = f"({operands[0]})" if left_need else str(operands[0])
        right_str = f"({operands[1]})" if right_need else str(operands[1])
        return f"{left_str} {self.SPEC.symbol} {right_str}"
    # end def print

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
    NAME = "unary_boolean_comparison"
    TENSOR_METHOD: str

    def check_operands(self, operands: Operands) -> bool:
        operand, = operands
        return operand.dtype == DType.B
    # end def check_operands

    def contains(self, expr: MathExpr, by_ref: bool = False, look_for: LeafKind = LeafKind.ANY) -> bool:
        return False
    # end def contains

    def _ensure_boolean_operand(self, operand: MathExpr) -> None:
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

    def _needs_parentheses(self, child: MathExpr, is_right_child: bool = False) -> None:
        return None
    # end def _needs_parentheses

    def _needs_parentheses_bool(self, child: MathExpr, is_right_child: bool = False) -> bool:
        if hasattr(child, "op") and child.op is not None:
            return child.op.SPEC.precedence <= self.SPEC.precedence
        # end if
        return False
    # end def _needs_parentheses_bool

    def infer_dtype(self, operands: Operands) -> DType:
        return DType.B
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        operand, = operands
        return self._result_shape(operand)
    # end def infer_shape

    def _result_shape(self, operand: MathExpr) -> Shape:
        return operand.shape
    # end def _result_shape

    def check_shapes(self, operands: Operands) -> bool:
        return True
    # end def check_shapes

    def print(self, operands: Operands, **kwargs) -> str:
        operand = operands[0]
        need_paren = self._needs_parentheses_bool(operand)
        operand_str = f"({operand})" if need_paren else str(operand)
        return f"{self.SPEC.symbol} {operand_str}"
    # end def print

    def __str__(self) -> str:
        return f"{self.NAME}(a)"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a)"
    # end def __repr__


class _ScalarBooleanComparisonOperator(_UnaryBooleanComparisonOperator, ABC):
    """Boolean reductions that always return scalars."""

    def _result_shape(self, operand: MathExpr) -> Shape:
        return Shape.scalar()
    # end def _result_shape

    def print(self, operands: Operands, **kwargs) -> str:
        return f"{self.SPEC.symbol}({operands[0]})"
    # end def print

# end class _ScalarBooleanComparisonOperator

class Eq(_BinaryComparisonOperator):
    """Element-wise equality operator."""

    SPEC = OperatorSpec(
        name="eq",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="==",
        precedence=7,
        associativity=OpAssociativity.NONE,
        commutative=True,
        associative=False,
        is_diff=False,
    )

    NAME = "eq"
    TENSOR_METHOD = "equal"
# end class Eq


class Ne(_BinaryComparisonOperator):
    """Element-wise inequality operator."""

    SPEC = OperatorSpec(
        name="ne",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="!=",
        precedence=7,
        associativity=OpAssociativity.NONE,
        commutative=True,
        associative=False,
        is_diff=False,
    )

    NAME = "ne"
    TENSOR_METHOD = "not_equal"
# end class Ne


class Lt(_BinaryComparisonOperator):
    """Element-wise less-than operator."""

    SPEC = OperatorSpec(
        name="lt",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="<",
        precedence=7,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "lt"
    TENSOR_METHOD = "less"
# end class Lt


class Le(_BinaryComparisonOperator):
    """Element-wise less-than-or-equal operator."""

    SPEC = OperatorSpec(
        name="le",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="<=",
        precedence=7,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "le"
    TENSOR_METHOD = "less_equal"
# end class Le


class Gt(_BinaryComparisonOperator):
    """Element-wise greater-than operator."""

    SPEC = OperatorSpec(
        name="gt",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol=">",
        precedence=7,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "gt"
    TENSOR_METHOD = "greater"
# end class Gt


class Ge(_BinaryComparisonOperator):
    """Element-wise greater-than-or-equal operator."""

    SPEC = OperatorSpec(
        name="ge",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol=">=",
        precedence=7,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "ge"
    TENSOR_METHOD = "greater_equal"
# end class Ge


class Not(_UnaryBooleanComparisonOperator):
    """Element-wise logical negation for boolean tensors."""

    SPEC = OperatorSpec(
        name="not",
        arity=AritySpec(exact=1, min_operands=1, variadic=False),
        symbol="not",
        precedence=6,
        associativity=OpAssociativity.RIGHT,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "not"
    TENSOR_METHOD = "logical_not"
# end class Not


class Any(_ScalarBooleanComparisonOperator):
    """Check whether any element in a boolean tensor is True."""

    SPEC = OperatorSpec(
        name="any",
        arity=AritySpec(exact=1, min_operands=1, variadic=False),
        symbol="any",
        precedence=6,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "any"
    TENSOR_METHOD = "any"
# end class Any


class All(_ScalarBooleanComparisonOperator):
    """Check whether every element in a boolean tensor is True."""

    SPEC = OperatorSpec(
        name="all",
        arity=AritySpec(exact=1, min_operands=1, variadic=False),
        symbol="all",
        precedence=6,
        associativity=OpAssociativity.NONE,
        commutative=False,
        associative=False,
        is_diff=False,
    )

    NAME = "all"
    TENSOR_METHOD = "all"
# end class All


class And(_BinaryBooleanComparisonOperator):
    """Element-wise logical AND."""

    SPEC = OperatorSpec(
        name="and",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="and",
        precedence=5,
        associativity=OpAssociativity.LEFT,
        commutative=True,
        associative=True,
        is_diff=False,
    )

    NAME = "and"
    TENSOR_METHOD = "logical_and"
# end class And


class Or(_BinaryBooleanComparisonOperator):
    """Element-wise logical OR."""

    SPEC = OperatorSpec(
        name="or",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="or",
        precedence=3,
        associativity=OpAssociativity.LEFT,
        commutative=True,
        associative=True,
        is_diff=False,
    )

    NAME = "or"
    TENSOR_METHOD = "logical_or"
# end class Or


class Xor(_BinaryBooleanComparisonOperator):
    """Element-wise logical XOR."""

    SPEC = OperatorSpec(
        name="xor",
        arity=AritySpec(exact=2, min_operands=2, variadic=False),
        symbol="xor",
        precedence=4,
        associativity=OpAssociativity.LEFT,
        commutative=True,
        associative=True,
        is_diff=False,
    )

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
