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
from typing import Type, Any, Optional, Sequence

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
    OpConstruct,
    AritySpec,
    OperatorSpec
)
from ..typing_rules import SimplifyOptions, SimplifyRuleType
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
        self._parent: Optional[MathExpr] = None
        self.check_parameters(**kwargs)
    # end def __init__

    # region OPERATOR

    @property
    def name(self) -> str:
        """Return the name of the operator."""
        return self.spec.name
    # end def name

    @property
    def arity(self) -> AritySpec:
        """Return the arity of the operator."""
        return self.spec.arity
    # end def arity

    @property
    def min_operands(self) -> int:
        """Return the minimum operand count."""
        return self.spec.arity.min_operands
    # end def min_operands

    @property
    def spec(self) -> OperatorSpec:
        """Return the operator specification."""
        return self.SPEC
    # end def spec

    @property
    def symbol(self) -> str:
        """Return the operator symbol."""
        return self.spec.symbol
    # end def symbol

    @property
    def precedence(self) -> int:
        """Return the operator precedence."""
        return self.spec.precedence
    # end def precedence

    @property
    def associativity(self) -> OpAssociativity:
        """Return the operator associativity."""
        return self.spec.associativity
    # end def associativity

    @property
    def commutative(self) -> bool:
        """Return whether the operator is commutative."""
        return self.spec.commutative
    # end def commutative

    @property
    def is_variadic(self) -> bool:
        """Return whether the operator accepts a variable number of operands."""
        return self.spec.arity.variadic
    # end def is_variadic

    @property
    def is_diff(self) -> bool:
        """Return whether the operator is differentiable."""
        return self.spec.is_diff
    # end def is_diff

    @property
    def parent(self) -> MathExpr | None:
        """Return the parent expression of the operator."""
        return self._parent
    # end def parent

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the operator parameters."""
        return self._parameters
    # end def parameters

    # endregion PROPERTIES

    # region PUBLIC

    def set_parent(self, parent: MathExpr) -> None:
        """Set the parent expression of the operator."""
        self._parent = parent
    # end def set_parent

    def eval(self, operands: Operands, **kwargs) -> "Tensor":
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
        if not self.is_diff:
            raise ValueError(f"Operator '{self.name}' is not differentiable")
        else:
            return self._diff(wrt=wrt, operands=operands)
        # end if
    # end def diff

    def fold_constants(self, operands: Operands) -> OpSimplifyResult:
        """Return operator-local simplification result for constant folding."""
        new_operands = list()
        for operand in operands:
            child_result = operand.fold_constants()
            new_operands.append(child_result)
        # end for
        return self._run_rules(
            operands=new_operands,
            rule_type=SimplifyRuleType.FOLD_CONSTANTS
        )
    # end def fold_constants

    def simplify(
            self,
            operands: Sequence[MathExpr],
            options: SimplifyOptions | None = None
    ) -> OpSimplifyResult:
        """Return operator-local simplification result."""
        return self._simplify(operands=operands, options=options)
    # end def check_operands

    def canonicalize(
            self,
            operands: Sequence[MathExpr]
    ) -> OpSimplifyResult:
        """Return canonicalized operands for this operator."""
        return self._canonicalize(operands=operands)
    # end def canonicalize

    @abstractmethod
    def contains(
            self,
            expr: "MathExpr",
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
    # end def contains

    def copy(self, deep: bool = False) -> Operator:
        """Return a copy of the operator."""
        new_op = self.__class__(**self._parameters.copy())
        new_op.set_parent(self._parent)
        return new_op
    # end def copy

    #
    # Infer shape and dtype
    #

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

    #
    # Checks
    #

    @abstractmethod
    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
    # end def check_operands

    @abstractmethod
    def check_parameters(self, **kwargs) -> bool:
        """Check that the operator parameters are valid."""
    # end def check_parameters

    @abstractmethod
    def check_shapes(
            self,
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
        if not cls.SPEC.arity.variadic:
            return len(operands) == cls.SPEC.arity.exact
        else:
            return cls.SPEC.arity.min_operands <= len(operands)
        # end if
        return True
    # end def check_arity

    #
    # Factory methods
    #

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
                name=rand_name(f"{cls.SPEC.name}"),
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
    # end def construct

    # endregion OPERATOR

    # region PUBLIC

    def get_parameters(self) -> dict[str, Any]:
        """Return the operator parameters."""
        return self._parameters.copy()
    # end def get_parameters

    def get_parameter(self, name: str) -> Any:
        """Return the value of a parameter."""
        return self._parameters.get(name)
    # end def get_parameter

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
        raise NotImplementedError(f"{self.name} is not differentiable.")
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
    def print(self, operands: Operands, **kwargs) -> str:
        """Return a human-readable representation of the operator."""
    # end def print

    def __eq__(self, other: Any) -> bool:
        """Check that two operators are equal."""
        return (
            isinstance(other, OperatorBase)
            and self.__class__ == other.__class__
            and self.spec == other.spec
            and self._parameters == other._parameters
        )
    # end def __eq__

    def __neq__(self, other: Any) -> bool:
        return not self.__eq__(other)
    # end def __neq__

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
    def _check_arity(
            cls,
            operands: Operands
    ):
        if not cls.check_arity(operands):
            if not cls.SPEC.arity.variadic:
                raise TypeError(
                    f"Operator {cls.SPEC.name} expected {cls.SPEC.arity.exact} operands, "
                    f"got {len(operands)}"
                )
            else:
                raise TypeError(
                    f"Operator {cls.SPEC.name} expected minimum {cls.SPEC.arity.min_operands} operands, "
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
            raise ValueError(f"Invalid operands for operator {op.name}: {operands}")
        # end if
    # end def _check_operands

# end class Operator


class ParametricOperator:
    """Marker mixin for operators carrying explicit parameters."""

    pass

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
        if op_cls.SPEC.name in cls._by_name:
            raise ValueError(f"Operator '{op_cls.SPEC.name}' is already registered.")
        # end if
        cls._by_name[op_cls.SPEC.name] = op_cls
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
