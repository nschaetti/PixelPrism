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
from abc import ABC
from typing import Sequence, Optional

from . import ParametricOperator
from ..tensor import Tensor
from ..dtype import DType
from ..shape import Shape
from .base import Operands, Operator, operator_registry

__all__ = [
    "ElementwiseOperator",
    "UnaryElementwiseOperator",
    "UnaryElementwiseParametricOperator",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Exp",
    "Exp2",
    "Expm1",
    "Log",
    "Log1p",
    "Log2",
    "Log10",
    "Sqrt",
    "Square",
    "Cbrt",
    "Reciprocal",
    "Deg2rad",
    "Rad2deg",
    "Absolute",
    "Abs",
    "Neg",
]


class ElementwiseOperator(Operator, ABC):
    """
    Element-wise operator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # end def __init__

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return True
    # end def check_operands

    def contains(
            self,
            expr: "MathExpr",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return False
    # end def contains

    def __str__(self) -> str:
        """Return a concise human-readable identifier."""
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return f"{self.__class__.__name__}(arity={self.ARITY})"
    # end def __repr__

    # region STATIC

    def check_shapes(self, operands: Operands) -> bool:
        """
        Check that both operands have identical shapes or support scalar broadcasting.
        """
        a, b = operands
        if a.rank != b.rank:
            if a.rank != 0 and b.rank != 0:
                raise ValueError(
                    f"{self.NAME} requires operands with identical ranks if not scalar, "
                    f"got {a.rank} and {b.rank}."
                )
            # end if
        else:
            # Same rank, but require same shape
            if a.input_shape != b.input_shape:
                raise ValueError(
                    f"{self.NAME} requires operands with identical shapes, "
                    f"got {a.input_shape} and {b.input_shape}."
                )
            # end if
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        """
        Output shape is identical to operand shapes.
        """
        a, b = operands
        if a.rank == b.rank:
            return operands[0].input_shape
        elif a.rank > b.rank:
            return operands[0].input_shape
        else:
            return operands[1].input_shape
        # end if
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

    # endregion STATIC

# end class ElementwiseOperator


class Add(ElementwiseOperator):
    """
    Element-wise addition operator.
    """

    ARITY = 2
    NAME = "add"

    # region PRIVATE

    def _eval(self, operands: Operands) -> Tensor:
        """
        Evaluate element-wise addition.
        """
        a, b = operands
        return a.eval() + b.eval()
    # end def _eval

    def _diff(self, wrt: "Variable", operands: Operands) -> "MathExpr":
        a, b = operands
        return a.diff(wrt) + b.diff(wrt)
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

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise subtraction."""
        a, b = operands
        return a.eval() - b.eval()
    # end def _eval

    def _diff(self, wrt: "Variable", operands: Operands) -> "MathExpr":
        a, b = operands
        return a.diff(wrt) - b.diff(wrt)
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

    def _eval(self, operands: Operands) -> Tensor:
        """Evaluate element-wise multiplication."""
        a, b = operands
        return a.eval() * b.eval()
    # end def _eval

    def _diff(self, wrt: "Variable", operands: Operands) -> "MathExpr":
        a, b = operands
        return a.diff(wrt) * b + b.diff(wrt) * a
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

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise division."""
        a, b = operands
        return a.eval() / b.eval()
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

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise exponentiation."""
        base, exponent = operands
        return Tensor.pow(base.eval(), exponent.eval())
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


class UnaryElementwiseOperator(Operator, ABC):
    """
    Base class for unary element-wise operators.
    """

    ARITY = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    # end def __init__

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return True
    # end def check_operands

    def contains(
            self,
            expr: "MathExpr",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return False
    # end def contains

    def __str__(self) -> str:
        """Return a concise human-readable identifier."""
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return f"{self.__class__.__name__}(arity={self.ARITY})"
    # end def __repr__

    # region STATIC

    def check_shapes(self, operands: Operands) -> bool:
        """Unary operators always match operand shape."""
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        """Shape matches operand."""
        return operands[0].input_shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """Return floating dtype for results."""
        operand_dtype = operands[0].dtype
        if operand_dtype.is_float:
            return operand_dtype
        return DType.FLOAT32
    # end def infer_dtype

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

    # endregion STATIC

# end class UnaryElementwiseOperator


class UnaryElementwiseParametricOperator(UnaryElementwiseOperator, ParametricOperator, ABC):
    pass
# end class UnaryElementwiseParametricOperator


class Exp(UnaryElementwiseOperator):
    """
    Element-wise exponential operator.
    """

    NAME = "exp"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.exp(value.eval())
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


class Exp2(UnaryElementwiseOperator):
    """
    Element-wise base-2 exponential operator.
    """

    NAME = "exp2"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.exp2(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Exp2 does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Exp2


class Expm1(UnaryElementwiseOperator):
    """
    Element-wise exp(x) - 1 operator.
    """

    NAME = "expm1"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.expm1(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Expm1 does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Expm1


class Log(UnaryElementwiseOperator):
    """
    Element-wise natural logarithm operator.
    """

    NAME = "log"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log(value.eval())
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


class Log1p(UnaryElementwiseOperator):
    """
    Element-wise log(1 + x) operator.
    """

    NAME = "log1p"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log1p(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Log1p does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Log1p


class Sqrt(UnaryElementwiseOperator):
    """
    Element-wise square root operator.
    """

    NAME = "sqrt"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.sqrt(value.eval())
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


class Square(UnaryElementwiseOperator):
    """
    Element-wise square operator.
    """

    NAME = "square"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.square(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Square does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Square


class Cbrt(UnaryElementwiseOperator):
    """
    Element-wise cubic root operator.
    """

    NAME = "cbrt"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.cbrt(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Cbrt does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Cbrt


class Reciprocal(UnaryElementwiseOperator):
    """
    Element-wise reciprocal operator.
    """

    NAME = "reciprocal"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.reciprocal(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Reciprocal does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Reciprocal


class Log2(UnaryElementwiseOperator):
    """
    Element-wise base-2 logarithm operator.
    """

    NAME = "log2"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log2(value.eval())
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

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.log10(value.eval())
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


class Deg2rad(UnaryElementwiseOperator):
    """
    Convert degrees to radians.
    """

    NAME = "deg2rad"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.deg2rad(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Deg2rad does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Deg2rad


class Rad2deg(UnaryElementwiseOperator):
    """
    Convert radians to degrees.
    """

    NAME = "rad2deg"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.rad2deg(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Rad2deg does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Rad2deg


class Absolute(UnaryElementwiseOperator):
    """
    Element-wise absolute value operator.
    """

    NAME = "absolute"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.absolute(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Absolute does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Absolute


class Abs(UnaryElementwiseOperator):
    """
    Element-wise absolute value alias operator.
    """

    NAME = "abs"

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.abs(value.eval())
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Abs does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class Abs


class Neg(UnaryElementwiseOperator):
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
        return operands[0].input_shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Return operand dtype."""
        return operands[0].dtype
    # end def infer_dtype

    # endregion STATIC

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate element-wise negation."""
        (value,) = operands
        return -value.eval()
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


operator_registry.register(Add)
operator_registry.register(Sub)
operator_registry.register(Mul)
operator_registry.register(Div)
operator_registry.register(Pow)
operator_registry.register(Exp)
operator_registry.register(Exp2)
operator_registry.register(Expm1)
operator_registry.register(Log)
operator_registry.register(Log1p)
operator_registry.register(Sqrt)
operator_registry.register(Square)
operator_registry.register(Cbrt)
operator_registry.register(Reciprocal)
operator_registry.register(Log2)
operator_registry.register(Log10)
operator_registry.register(Deg2rad)
operator_registry.register(Rad2deg)
operator_registry.register(Absolute)
operator_registry.register(Abs)
operator_registry.register(Neg)
