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
from abc import ABC, abstractmethod
from typing import List, Any, Sequence
import numpy as np
from .dtype import DType
from .shape import Shape


__all__ = [
    "Operator",
    "OperatorRegistry",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "operator_registry"
]


Operand = "MathExpr"
Operands = List["MathExpr"]


class Operator(ABC):
    """
    Represents an operator that can be applied to a value.
    """

    # How many operands
    ARITY: int

    # Operator name
    NAME: str

    def __init__(self):
        """
        Constructor.

        Parameters
        ----------
        operands : Operands
            Operands of the operator.
        """
        if not hasattr(self, "ARITY"):
            raise TypeError("Operator subclasses must define ARITY")
        # end if
        if not hasattr(self, "NAME"):
            raise TypeError("Operator subclasses must define NAME")
        # end if
    # end def __init__

    # region PROPERTIES

    @property
    def name(self) -> str:
        """Return the name of the operator."""
        return self.NAME
    # end def name

    @property
    def arity(self) -> int:
        """Return the arity of the operator."""
        return self.ARITY
    # end def arity

    # endregion PROPERTIES

    # region PUBLIC

    def eval(self, values) -> np.ndarray:
        """Evaluate the operator."""
        return self._eval(values=values)
    # end def eval

    def backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        """
        Local backward rule for this operator.

        Given the gradient with respect to the output of this operator
        (out_grad), return the gradients with respect to each input
        operand, in the same order as `node.children`.

        Parameters
        ----------
        out_grad : 'MathExpr'
            Gradient of the loss with respect to the output of this node
            (∂L / ∂out).
        node : 'MathExpr'
            The operator node being differentiated. Provides access to
            the input expressions (children).

        Returns
        -------
        Sequence['MathExpr']
            Gradients with respect to each input operand
            (∂L / ∂input_i), in the same order as node.children.

        Notes
        -----
        - This method defines a *local* differentiation rule.
        - It does not choose the variable with respect to which the
          derivative is taken.
        - It must not perform any global graph traversal or accumulation.
        """
        return self._backward(out_grad, node)
    # end def backward

    # endregion PUBLIC

    # region PRIVATE

    @abstractmethod
    def _eval(self, values) -> np.ndarray:
        """Evaluate the operator."""
    # end def _eval

    @abstractmethod
    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        """
        Local backward rule for this operator.
        """
    # end _backward

    # endregion PRIVATE

    # region STATIC

    @classmethod
    @abstractmethod
    def infer_dtype(
            cls,
            operands: Operands
    ) -> DType:
        """Return the output data type of the operator."""
    # end def output_dtype

    @classmethod
    @abstractmethod
    def infer_shape(
            cls,
            operands: Operands
    ) -> Shape:
        """Return the output shape of the operator."""
    # end def output_shape

    @classmethod
    @abstractmethod
    def check_shapes(
            cls,
            operands: Operands
    ) -> bool:
        """Check that the operands have compatible shapes."""

    # end def check_operands_shapes

    @classmethod
    def check_arity(
            cls,
            operands: Operands
    ):
        """Check that the operands have the correct arity."""
        return len(operands) == cls.ARITY
    # end def check_operands_arity

    # endregion STATIC

# end class Operator


class Add(Operator):
    """
    Element-wise addition operator.
    """

    ARITY = 2
    NAME = "add"

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """
        Check that both operands have identical shapes.
        """
        a, b = operands
        if a.shape != b.shape:
            raise ValueError(
                f"Add requires operands with identical shapes, "
                f"got {a.shape} and {b.shape}."
            )
        # end if
        return True
    # end def check_operands_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """
        Output shape is identical to operand shapes.
        """
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def output_dtype

    # endregion STATIC

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


class Sub(Operator):
    """
    Element-wise subtraction operator.
    """

    ARITY = 2
    NAME = "sub"

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """Require identical operand shapes."""
        a, b = operands
        if a.shape != b.shape:
            raise ValueError(
                f"Sub requires operands with identical shapes, "
                f"got {a.shape} and {b.shape}."
            )
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """Shape matches operands."""
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Promote operand dtypes."""
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

    # endregion STATIC

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


class Mul(Operator):
    """
    Element-wise multiplication operator.
    """

    ARITY = 2
    NAME = "mul"

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """Require identical operand shapes."""
        a, b = operands
        if a.shape != b.shape:
            raise ValueError(
                f"Mul requires operands with identical shapes, "
                f"got {a.shape} and {b.shape}."
            )
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """Shape matches operands."""
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Promote operand dtypes."""
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

    # endregion STATIC

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


class Div(Operator):
    """
    Element-wise division operator.
    """

    ARITY = 2
    NAME = "div"

    def __init__(self):
        super().__init__()
    # end def __init__

    # region STATIC

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        """Require identical operand shapes."""
        a, b = operands
        if a.shape != b.shape:
            raise ValueError(
                f"Div requires operands with identical shapes, "
                f"got {a.shape} and {b.shape}."
            )
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        """Shape matches operands."""
        return operands[0].shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """Promote operand dtypes."""
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

    # endregion STATIC

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


class OperatorRegistry:
    """
    Global registry for Operator instances.
    """

    _by_name: dict[str, Operator] = {}

    @classmethod
    def register(cls, op: Operator) -> None:
        """
        Register an operator.

        Raises
        ------
        ValueError
            If an operator with the same name is already registered.
        """
        if op.name in cls._by_name:
            raise ValueError(f"Operator '{op.name}' is already registered.")
        # end if
        cls._by_name[op.name] = op
    # end def register

    @classmethod
    def get(cls, name: str) -> Operator:
        """
        Retrieve a registered operator by name.

        Raises
        ------
        KeyError
            If the operator is not registered.
        """
        return cls._by_name[name]
    # end def get

    @classmethod
    def has(cls, name: str) -> bool:
        """Check whether an operator is registered."""
        return name in cls._by_name
    # end def has

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for tests)."""
        cls._by_name.clear()
    # end def clear

# end class OperatorRegistry


# The operator registry
operator_registry = OperatorRegistry()
operator_registry.register(Add())
operator_registry.register(Sub())
operator_registry.register(Mul())
operator_registry.register(Div())


