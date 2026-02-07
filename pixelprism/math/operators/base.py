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
from typing import List, Type, Tuple, Any, Optional

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from ..math_expr import MathNode, Variable


__all__ = [
    "Operands",
    "Operator",
    "ParametricOperator",
    "OperatorRegistry",
    "operator_registry",
]


Operand = MathNode
Operands = List[MathNode] | Tuple[MathNode, ...]


class Operator(ABC):
    """
    Represents an operator that can be applied to a value.
    """

    # How many operands
    ARITY: int

    # Is an operator with variable number of operands ?
    IS_VARIADIC: bool = False

    # Is the operator differentiable ?
    IS_DIFF: bool = False

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
        self._parameters: dict[str, Any] = kwargs
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

    @arity.setter
    def arity(self, value: int) -> None:
        if not self.IS_VARIADIC:
            raise ValueError("Cannot set arity of non-variable operator")
        # end if
        self.ARITY = value
    # end def arity

    @property
    def is_variadic(self) -> bool:
        """Return whether the operator accepts a variable number of operands."""
        return self.IS_VARIADIC
    # end def is_variadic

    @property
    def is_diff(self) -> bool:
        """Return whether the operator is differentiable."""
        return self.IS_DIFF
    # end def is_diff

    # endregion PROPERTIES

    # region PUBLIC

    @abstractmethod
    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
    # end def contains

    @abstractmethod
    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
    # end def check_operands

    def get_parameters(self) -> dict[str, Any]:
        """Return the operator parameters."""
        return self._parameters.copy()
    # end def get_parameters

    def get_parameter(self, name: str) -> Any:
        """Return the value of a parameter."""
        return self._parameters.get(name)
    # end def get_parameter

    def eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate the operator."""
        if len(operands) != self.arity:
            raise ValueError(f"Expected {self.arity} operands, got {len(operands)}")
        # end if
        return self._eval(operands=operands, **kwargs)
    # end def eval

    def diff(
            self,
            wrt: Variable,
            operands: Operands
    ) -> MathNode:
        """
        Local derivative of the operator wrt the given expression.
        """
        if not self.IS_DIFF:
            raise ValueError(f"Operator '{self.name}' is not differentiable")
        else:
            return self._diff(wrt=wrt, operands=operands)
        # end if
    # end def backward

    # endregion PUBLIC

    # region PRIVATE

    @abstractmethod
    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate the operator."""
    # end def _eval

    def _diff(
            self,
            wrt: Variable,
            operands: Operands,
    ) -> MathNode:
        """
        Local backward rule for this operator.
        """
        raise NotImplementedError(f"{self.NAME} is not differentiable.")
    # end _backward

    # endregion PRIVATE

    # region STATIC

    @abstractmethod
    def infer_dtype(
            self,
            operands: Operands
    ) -> DType:
        """Return the output data type of the operator."""
    # end def infer_dtype

    @abstractmethod
    def infer_shape(
            self,
            operands: Operands
    ) -> Shape:
        """Return the output shape of the operator."""
    # end def infer_shape

    @abstractmethod
    def check_shapes(
            self,
            operands: Operands
    ) -> bool:
        """Check that the operands have compatible shapes."""
    # end def check_shapes

    @abstractmethod
    def __str__(self) -> str:
        """Return a concise human-readable name for the operator."""
    # end def __str__

    @abstractmethod
    def __repr__(self) -> str:
        """Return a debug-friendly description for the operator."""
    # end def __repr__

    @staticmethod
    def _resolve_parameter(v: Any):
        if v is None:
            return None
        # end if
        if MathNode is not None and isinstance(v, MathNode):
            return v.eval().item()
        # end if
        return v
    # end def _resolve_parameter

    @classmethod
    def check_arity(
            cls,
            operands: Operands
    ):
        """Check that the operands have the correct arity."""
        if not cls.IS_VARIADIC:
            return len(operands) == cls.ARITY
        # end if
        return True
    # end def check_arity

    @classmethod
    def create_node(
            cls,
            operands: Operands,
            **kwargs
    ) -> MathNode:
        """
        Build a MathNode by applying a registered operator to operands.
        """
        # Get operator class
        op_cls = cls

        # We check that operator arity is respected
        if not op_cls.check_arity(operands):
            raise TypeError(
                f"Operator {op_cls.NAME}({op_cls.ARITY}) expected {op_cls.ARITY} operands, "
                f"got {len(operands)}"
            )
        # end if

        # Instantiate operator
        op = op_cls(**kwargs)

        # We check that shapes of the operands are compatible
        if not op.check_shapes(operands):
            shapes = ", ".join(str(o.shape) for o in operands)
            raise TypeError(
                f"Incompatible shapes for operator {op_cls.NAME}: {shapes}"
            )
        # end if

        # We check that the operator approves the operand(s)
        if not op.check_operands(operands):
            raise ValueError(f"Invalid parameters for operator {op.name}: {kwargs}")
        # end if

        return MathNode(
            name=op_cls.NAME,
            op=op,
            children=operands,
            dtype=op.infer_dtype(operands),
            shape=op.infer_shape(operands),
        )
    # end def create_node

    # endregion STATIC

# end class Operator


class ParametricOperator(ABC):
    """
    Operator with parameters.
    """

    def __init__(self, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        operands : Operands
            Operands of the operator.
        """
        self._parameters: dict[str, Any] = kwargs
        self._check_parameters(**kwargs)
    # end def __init__

    def _check_parameters(self, **kwargs) -> bool:
        return self.check_parameters(**kwargs)
    # end def _check_parameters

    @classmethod
    @abstractmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
    # end def check_shapes

# end class ParametricOperator


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
