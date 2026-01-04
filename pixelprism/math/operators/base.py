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
Operator base classes and registry.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence, Type

import numpy as np

from ..dtype import DType
from ..shape import Shape

__all__ = [
    "Operands",
    "Operator",
    "BinderOperator",
    "OperatorRegistry",
    "operator_registry",
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

    def __init__(self, **kwargs):
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

    @abstractmethod
    def contains(self, expr: "MathExpr") -> bool:
        """Does the operator contain the given expression (in parameters)?"""
    # end def contains

    @abstractmethod
    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
    # end def check_operands

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
    # end def infer_dtype

    @classmethod
    @abstractmethod
    def infer_shape(
            cls,
            operands: Operands
    ) -> Shape:
        """Return the output shape of the operator."""
    # end def infer_shape

    @classmethod
    @abstractmethod
    def check_shapes(
            cls,
            operands: Operands
    ) -> bool:
        """Check that the operands have compatible shapes."""
    # end def check_shapes

    @classmethod
    def check_arity(
            cls,
            operands: Operands
    ):
        """Check that the operands have the correct arity."""
        return len(operands) == cls.ARITY
    # end def check_arity

    # endregion STATIC

# end class Operator


class BinderOperator(Operator):
    """
    Base class for operators that bind operands to a variable.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_parameters(**kwargs)
    # end def __init__

    def eval_node(self, operands: Operands, **kwargs) -> np.ndarray:
        """Evaluate the binder operator."""
        return self._eval_node(operands=operands, **kwargs)
    # end def eval_node

    # region PRIVATE

    @classmethod
    @abstractmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
    # end def check_shapes

    def _check_parameters(self, **kwargs):
        """Check parameters."""
        self.__class__.check_parameters(**kwargs)
    # end def _check_parameters

    @abstractmethod
    def _eval_node(self, operands: Operands, **kwargs) -> np.ndarray:
        """Evaluate the operator."""
    # end def _eval_node

# end class BinderOperator


class OperatorRegistry:
    """
    Global registry for Operator instances.
    """

    _by_name: dict[str, Type[Operator]] = {}

    @classmethod
    def register(cls, op_cls: Type[Operator]) -> None:
        """
        Register an operator.

        Raises
        ------
        ValueError
            If an operator with the same name is already registered.
            The operator class must be a subclass of Operator.
        """
        if not issubclass(op_cls, Operator):
            raise TypeError("Only Operator subclasses can be registered")
        # end if
        if op_cls.NAME in cls._by_name:
            raise ValueError(f"Operator '{op_cls.NAME}' is already registered.")
        # end if
        cls._by_name[op_cls.NAME] = op_cls
    # end def register

    @classmethod
    def get(cls, name: str) -> Type[Operator]:
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

