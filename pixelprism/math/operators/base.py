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
from typing import List, Type, Tuple, Any, Optional, Sequence, Union

from ..random import rand_name
from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from ..math_node import MathNode
from ..math_leaves import Variable
from ..typing_expr import (
    Operands,
    Operator,
    MathExpr,
    LeafKind,
    OpSimplifyResult,
    OpAssociativity,
    OpConstruct
)
from ..typing_rules import SimplifyOptions, SimplifyRule, SimplifyRuleType
from ..mixins import SimplifyRuleMixin

__all__ = [
    "OperatorBase",
    "ParametricOperator",
    "OperatorRegistry",
    "operator_registry",
]


class OperatorBase(
    Operator,
    SimplifyRuleMixin,
    ABC
):
    """
    Represents an operator that can be applied to a value.
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
        self.__class__.ARITY = value
    # end def arity

    @property
    def min_operands(self) -> int:
        """Return the minimum operand count."""
        return self.MIN_OPERANDS
    # end def min_operands

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
            expr: "MathExpr",
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
    # end def contains

    def simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        """Return operator-local simplification result."""
        return self._simplify(operands=operands, options=options)
    # end def simplify

    def canonicalize(
            self,
            operands: Sequence[MathExpr]
    ) -> OpSimplifyResult:
        """Return canonicalized operands for this operator."""
        return self._canonicalize(operands=operands)
    # end def canonicalize

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
        if self.is_variadic and len(operands) < self.min_operands:
            raise ValueError(
                f"Expected at least {self.min_operands} operands for operator '{self.name}', "
                f"got {len(operands)}: {operands}"
            )
        # end if
        if not self.is_variadic and len(operands) != self.arity:
            raise ValueError(f"Expected {self.arity} operands, got {len(operands)}")
        # end if
        return self._eval(operands=operands, **kwargs)
    # end def eval

    def diff(
            self,
            wrt: Variable,
            operands: Operands
    ) -> MathExpr:
        """
        Local derivative of the operator wrt the given expression.
        """
        if not self.IS_DIFF:
            raise ValueError(f"Operator '{self.name}' is not differentiable")
        else:
            return self._diff(wrt=wrt, operands=operands)
        # end if
    # end def diff

    # endregion PUBLIC

    # region PRIVATE

    @abstractmethod
    def _needs_parentheses(self, *args, **kwargs):
        """Return ```True``` if the child operator needs parentheses.

        Parameters
        ----------
        child : MathExpr
            The child operator to check.
        is_right_child : bool
            Whether the child is the right child of the current operator.

        Returns
        -------
        bool
            Whether the child operator needs parentheses.
        """
    # end def _needs_parentheses

    @abstractmethod
    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate the operator."""
    # end def _eval

    def _diff(
            self,
            wrt: Variable,
            operands: Operands,
    ) -> MathExpr:
        """
        Local backward rule for this operator.
        """
        raise NotImplementedError(f"{self.NAME} is not differentiable.")
    # end _diff

    def _simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        """Return operator-local simplification result."""
        hit = self._run_rules(
            operands=operands,
            options=options,
            rule_type=SimplifyRuleType.SIMPLIFICATION
        )
        return hit or OpSimplifyResult(operands=operands, replacement=None)
    # end def simplify

    def _canonicalize(self, operands: Sequence[MathExpr]) -> OpSimplifyResult:
        """Return canonicalized operands for this operator."""
        hit = self._run_rules(
            operands=operands,
            options=None,
            rule_type=SimplifyRuleType.CANONICALIZATION
        )
        return hit or OpSimplifyResult(operands=operands, replacement=None)
    # end def canonicalize

    # endregion PRIVATE

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
    def print(self, operands: Operands, **kwargs) -> str:
        """Return a human-readable representation of the operator."""
    # end def print

    @abstractmethod
    def __str__(self) -> str:
        """Return a concise human-readable name for the operator."""
    # end def __str__

    @abstractmethod
    def __repr__(self) -> str:
        """Return a debug-friendly description for the operator."""
    # end def __repr__

    # region STATIC

    @staticmethod
    def _resolve_parameter(v: Any):
        if v is None:
            return None
        # end if
        if MathExpr is not None and isinstance(v, MathExpr):
            return v.eval().item()
        # end if
        return v
    # end def _resolve_parameter

    # endregion STATIC

    # region CLASS METHODS

    @classmethod
    def check_arity(
            cls,
            operands: Operands
    ):
        """Check that the operands have the correct arity."""
        if not cls.IS_VARIADIC:
            return len(operands) == cls.ARITY
        else:
            return cls.MIN_OPERANDS <= len(operands)
        # end if
        return True
    # end def check_arity

    @classmethod
    def create_node(
            cls,
            operands: Operands,
            **kwargs
    ) -> MathExpr:
        """
        Build a MathNode by applying a registered operator to operands.
        """
        # Instantiate operator
        op_result = cls.construct(operands=operands, **kwargs)

        if isinstance(op_result.expr, MathExpr):
            return op_result.expr
        elif isinstance(op_result.expr, Operator):
            op = op_result.expr
            operands = op_result.operands
            return MathNode(
                name=rand_name(f"{cls.NAME}"),
                op=op,
                children=operands,
                dtype=op.infer_dtype(operands),
                shape=op.infer_shape(operands),
            )
        else:
            raise TypeError(f"Unexpected operator type: {type(op_result.expr)}")
        # end if
    # end def create_node

    @classmethod
    def _check_arity(
            cls,
            operands: Operands
    ):
        if not cls.check_arity(operands):
            if not cls.IS_VARIADIC:
                raise TypeError(
                    f"Operator {cls.NAME}({cls.ARITY}) expected {cls.ARITY} operands, "
                    f"got {len(operands)}"
                )
            else:
                raise TypeError(
                    f"Operator {cls.NAME} expected minimum {cls.MIN_OPERANDS} operands, "
                    f"got {len(operands)}"
                )
            # end if
        # end if
    # end def _check_arity

    @classmethod
    def _check_shapes(cls, op: Operator, operands: Operands):
        # We check that the shapes of the operands are compatible
        if not op.check_shapes(operands):
            shapes = ", ".join(str(o.shape) for o in operands)
            raise TypeError(
                f"Incompatible shapes for operator {op.name}: {shapes}"
            )
        # end if
    # end def _check_shapes

    @classmethod
    def _check_operands(cls, op: Operator, operands: Operands):
        # We check that the operator approves the operand(s)
        if not op.check_operands(operands):
            raise ValueError(f"Invalid parameters for operator {op.name}: {kwargs}")
        # end if
    # end def _check_operands

    @classmethod
    # Construct the operator. This is the official and only way to create the operator instance.
    # The method will check rule of simplification.
    def construct(cls, operands: Operands, **kwargs) -> OpConstruct:
        """
        Construct the operator. This is the official and only way to create the operator instance.
        The method will check the rule of simplification.

        Parameters
        ----------
        operands : Operands
            The operands for the operator.
        kwargs : dict
            Additional keyword arguments for the operator.

        Returns
        -------
        OpConstruct
            The constructed operator or simplified expression.
        """
        # Instantiate operator
        op = cls(**kwargs)

        # Canonicalize operands
        canon_result: OpSimplifyResult = op.canonicalize(operands)

        if canon_result.replacement is None:
            cls._check_arity(canon_result.operands)
            cls._check_shapes(op, canon_result.operands)
            cls._check_operands(op, canon_result.operands)
            return OpConstruct(expr=op, operands=canon_result.operands)
        else:
            return OpConstruct(expr=canon_result.replacement, operands=canon_result.operands)
        # end if
    # endregion CLASS METHODS

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

    _by_name: dict[str, Type[OperatorBase]] = {}

    @classmethod
    def register(cls, op_cls: Type[OperatorBase]) -> None:
        """
        Register an operator.

        Raises
        ------
        ValueError
            If an operator with the same name is already registered.
            The operator class must be a subclass of Operator.
        """
        if not issubclass(op_cls, OperatorBase):
            raise TypeError("Only Operator subclasses can be registered")
        # end if
        if op_cls.NAME in cls._by_name:
            raise ValueError(f"Operator '{op_cls.NAME}' is already registered.")
        # end if
        cls._by_name[op_cls.NAME] = op_cls
    # end def register

    @classmethod
    def get(cls, name: str) -> Type[OperatorBase]:
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
