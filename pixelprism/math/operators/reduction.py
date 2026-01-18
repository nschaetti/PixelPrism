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
from abc import ABC
from typing import Sequence, List, Any, Optional
import numpy as np

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor
from .base import Operands, Operand, operator_registry, Operator

__all__ = [
    "ReductionOperator",
    "Mean",
    "Sum",
    "Std",
    "Summation"
]


class ReductionOperator(Operator, ABC):
    """
    Reduction operators.
    """

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return True
    # end def check_operands

    def contains(self, expr: "MathExpr") -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return False
    # end def contains

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, = operands
        return a.dtype
    # end def infer_dtype

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

# end class ReductionOperator


class Sum(ReductionOperator):
    """
    Sum operator.
    """

    NAME = "sum"
    ARITY = 1

    def __init__(self, *, axis: Optional["MathExpr"] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._axis = axis
    # end def __init__

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError(f"Sum requires exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        return Shape(())
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        if self._axis is not None:
            return a.eval().sum(axis=self._axis.eval().item())
        else:
            return a.eval().sum()
        # end if
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Sum


class Mean(ReductionOperator):
    """
    Mean operator.
    """

    NAME = "mean"
    ARITY = 1

    def __init__(self, *, axis: Optional["MathExpr"] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._axis = axis
    # end def __init__

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError(f"Mean requires exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        return Shape(())
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate mean.
        """
        a, = operands
        if self._axis is not None:
            return a.eval().mean(axis=self._axis.eval().item())
        else:
            return a.eval().mean(a)
        # end if
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Mean


class Std(ReductionOperator):
    """
    Standard deviation operator.
    """

    NAME = "std"
    ARITY = 1

    def __init__(self, *, axis: Optional["MathExpr"] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._axis = axis
    # end def __init__

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError(f"Std requires exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        return Shape(())
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate standard deviation."""
        a, = operands
        if self._axis is not None:
            return a.eval().std(axis=self._axis.eval().item())
        else:
            return a.eval().std()
        # end if
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Std


class ReductionParametricOperator(Operator, ABC):
    """
    Reduction operators with parameters.
    """

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        raise NotImplementedError("Parametric reduction operators must check their parameters (not implemented).")
    # end def check_operands

    def contains(self, expr: "MathExpr") -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric reduction operators must implement contains(..).")
    # end def contains

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        raise NotImplementedError("Parametric reduction operators must check their parameters (not implemented).")
    # end def check_shapes

# end class ReductionParametricOperator


class Summation(ReductionParametricOperator):
    """
    Summation operator.
    """

    NAME = "summation"
    ARITY = 1

    def __init__(self, lower: "MathExpr", upper: "MathExpr", bounded_variable: "Tensor"):
        super().__init__(
            lower=lower,
            upper=upper,
            bounded_variable=bounded_variable
        )
        self._lower = lower
        self._upper = upper
        self._bounded_variable = bounded_variable
    # end def __init__

    @property
    def lower(self):
        """Lower bound."""
        return self._lower
    # end def lower

    @property
    def upper(self):
        """Upper bound."""
        return self._upper
    # end def upper

    @property
    def bounded_variable(self):
        """Bounded variable."""
        return self._bounded_variable
    # end def bounded_variable

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        body, = operands
        index_found = body.contains(self.bounded_variable, by_ref=True)
        if not index_found:
            raise ValueError(f"Bounded variable {self.bounded_variable} not found in body : {body}.")
        # end if
        return True
    # end def check_operands

    def contains(self, expr: "MathExpr") -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return expr in {self.lower, self.upper, self.bounded_variable}
    # end def contains

    @classmethod
    def check_parameters(cls, lower: "MathExpr", upper: "MathExpr", bounded_variable: Operand) -> bool:
        """Check that the operands have compatible shapes."""
        # Lower must be int32 or int64
        if lower.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError(f"Lower bound must be int32 or int64, got {lower.dtype}")
        # end if

        # Upper must be int32 or int64
        if upper.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError(f"Upper bound must be int32 or int64, got {upper.dtype}")
        # end if

        # Bounded variable must a leaf
        if not bounded_variable.is_leaf:
            raise ValueError(f"Bounded variable must be a leaf, got {bounded_variable}")
        # end if

        # Bounded variable must be int32 or int64
        if bounded_variable.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError(f"Bounded variable must be int32 or int64, got {bounded_variable.dtype}")
        # end if

        return True
    # end def check_shapes

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        if len(operands) != cls.ARITY:
            raise ValueError(f"Sum requires exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        body, = operands
        return body.shape
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, = operands
        return a.dtype
    # end def infer_dtype

    def _eval_node(self, operands: Operands) -> np.ndarray:
        # Get body
        body, = operands

        # Evaluate lower and upper bound
        lower = self.lower.eval()
        upper = self.upper.eval()

        # Iterate
        ret_value = np.zeros(body.shape.dims, dtype=body.dtype.to_numpy())
        for i in range(lower, upper + 1):
            self._bounded_variable.set(np.array(i, dtype=self._bounded_variable.dtype.to_numpy()))
            ret_value += body.eval()
        # end for

        return ret_value
    # end def _eval

    def _eval(self, values: List[np.ndarray]) -> np.ndarray:
        """Evaluate summation.
        """
        raise RuntimeError("Summation must be evaluated using _eval_node(..).")
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Summation


operator_registry.register(Sum)
operator_registry.register(Mean)
operator_registry.register(Std)
operator_registry.register(Summation)
