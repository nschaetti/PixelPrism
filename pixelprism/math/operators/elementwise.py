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
Elementwise operator implementations.
"""

from typing import Sequence

import numpy as np

from ..dtype import DType
from ..shape import Shape
from .base import Operands, Operator, operator_registry

__all__ = [
    "ElementwiseOperator",
    "UnaryElementwiseOperator",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Exp",
    "Log",
    "Log2",
    "Log10",
    "Sqrt",
    "Neg",
]


class ElementwiseOperator(Operator):
    """
    Element-wise operator.
    """

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """
        Check that both operands have identical shapes or support scalar broadcasting.
        """
        a, b = operands
        if a.rank != b.rank:
            if a.rank != 0 and b.rank != 0:
                raise ValueError(
                    f"{cls.NAME} requires operands with identical ranks if not scalar, "
                    f"got {a.rank} and {b.rank}."
                )
            # end if
        else:
            # Same rank, but require same shape
            if a.shape != b.shape:
                raise ValueError(
                    f"{cls.NAME} requires operands with identical shapes, "
                    f"got {a.shape} and {b.shape}."
                )
            # end if
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """
        Output shape is identical to operand shapes.
        """
        a, b = operands
        if a.rank == b.rank:
            return operands[0].shape
        elif a.rank > b.rank:
            return operands[0].shape
        else:
            return operands[1].shape
        # end if
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

    # endregion STATIC

# end class ElementwiseOperator


class Add(ElementwiseOperator):
    """
    Element-wise addition operator.
    """

    ARITY = 2
    NAME = "add"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        """
        Evaluate element-wise addition.
        """
        a, b = values
        return a + b
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Add does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Add


class Sub(ElementwiseOperator):
    """
    Element-wise subtraction operator.
    """

    ARITY = 2
    NAME = "sub"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        """Evaluate element-wise subtraction."""
        a, b = values
        return a - b
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Sub does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Sub


class Mul(ElementwiseOperator):
    """
    Element-wise multiplication operator.
    """

    ARITY = 2
    NAME = "mul"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        """Evaluate element-wise multiplication."""
        a, b = values
        return a * b
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Mul does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Mul


class Div(ElementwiseOperator):
    """
    Element-wise division operator.
    """

    ARITY = 2
    NAME = "div"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        """Evaluate element-wise division."""
        a, b = values
        return a / b
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Div does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Div


class Pow(ElementwiseOperator):
    """
    Element-wise power operator.
    """

    ARITY = 2
    NAME = "pow"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        """Evaluate element-wise exponentiation."""
        base, exponent = values
        return np.power(base, exponent)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Pow does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Pow


class UnaryElementwiseOperator(Operator):
    """
    Base class for unary element-wise operators.
    """

    ARITY = 1

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """Unary operators always match operand shape."""
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """Shape matches operand."""
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Return floating dtype for results."""
        operand_dtype = operands[0].dtype
        if operand_dtype.is_float:
            return operand_dtype
        return DType.FLOAT32
    # end def infer_dtype

    # endregion STATIC

# end class UnaryElementwiseOperator


class Exp(UnaryElementwiseOperator):
    """
    Element-wise exponential operator.
    """

    NAME = "exp"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.exp(value)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Exp does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Exp


class Log(UnaryElementwiseOperator):
    """
    Element-wise natural logarithm operator.
    """

    NAME = "log"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.log(value)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Log does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Log


class Sqrt(UnaryElementwiseOperator):
    """
    Element-wise square root operator.
    """

    NAME = "sqrt"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.sqrt(value)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Sqrt does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Sqrt


class Log2(UnaryElementwiseOperator):
    """
    Element-wise base-2 logarithm operator.
    """

    NAME = "log2"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.log2(value)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Log2 does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Log2


class Log10(UnaryElementwiseOperator):
    """
    Element-wise base-10 logarithm operator.
    """

    NAME = "log10"

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.log10(value)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Log10 does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Log10


class Neg(Operator):
    """
    Element-wise negation operator.
    """

    ARITY = 1
    NAME = "neg"

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """Neg accepts any operand shape."""
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """Shape matches operand."""
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Return operand dtype."""
        return operands[0].dtype
    # end def infer_dtype

    # endregion STATIC

    # region PRIVATE

    def _eval(self, values: np.ndarray) -> np.ndarray:
        """Evaluate element-wise negation."""
        (value,) = values
        return -value
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Neg does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Neg


operator_registry.register(Add())
operator_registry.register(Sub())
operator_registry.register(Mul())
operator_registry.register(Div())
operator_registry.register(Pow())
operator_registry.register(Exp())
operator_registry.register(Log())
operator_registry.register(Sqrt())
operator_registry.register(Log2())
operator_registry.register(Log10())
operator_registry.register(Neg())
