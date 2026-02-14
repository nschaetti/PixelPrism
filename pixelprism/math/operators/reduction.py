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
from typing import Any, Optional, Union
import numpy as np

from ..dtype import DType, to_numpy
from ..shape import Shape
from ..tensor import Tensor
from ..context import new_context, set_value
from ..math_node import MathNode
from .base import Operands, operator_registry, OperatorBase, ParametricOperator


__all__ = [
    "ReductionOperator",
    "Mean",
    "Sum",
    "Std",
    "Median",
    "Max",
    "Min",
    "Q1",
    "Q3",
    "Summation",
    "Product",
]


class ReductionOperator(OperatorBase, ParametricOperator, ABC):
    """
    Reduction operators.
    """

    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, = operands
        return a.dtype
    # end def infer_dtype

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        raise NotImplementedError(f"{self.__class__.__name__} operator must check their parameters (not implemented).")
    # end def check_operands

    def contains(
            self,
            expr: "MathNode",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric reduction operators must implement contains(..).")
    # end def contains

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        raise NotImplementedError("Parametric reduction operators must check their parameters (not implemented).")
    # end def check_shapes

    def __str__(self) -> str:
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    # end def __repr__

# end class ReductionOperator


class AxisReductionOperator(ReductionOperator, ABC):
    """
    Reduction operators.
    """

    def __init__(self, *, axis: Optional[Union[MathNode, int]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        from ..math_leaves import Variable
        if axis and isinstance(axis, int):
            from ..random import random_const_name
            from ..math_leaves import const
            axis = const(random_const_name(f"{self.__class__.__name__.lower()}-axis-"), axis)
        elif axis and isinstance(axis, Variable):
            # Only constant (otherwise we have dynamic shapes)
            raise ValueError(f"{self.__class__.__name__} does not support dynamic axis.")
        # end if
        if axis is not None and axis.value.item() < 0:
            raise ValueError(f"Axis {axis.value} is negative.")
        # end if
        self._axis = axis
    # end def __init__

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return len(operands) == 1
    # end def check_operands

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return self._axis is not None and self._axis.contains(expr, by_ref=by_ref)
    # end def contains

    def _axis_value(self) -> Optional[int]:
        """Return the resolved axis integer when available."""
        if self._axis is None:
            return None
        if hasattr(self._axis, "value"):
            return int(self._axis.value.item())
        if hasattr(self._axis, "eval"):
            return int(self._axis.eval().item())
        return int(self._axis)
    # end def _axis_value

    def __str__(self) -> str:
        return f"{self.NAME}(axis={self._axis_value()})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self._axis_value()})"
    # end def __repr__

    def infer_shape(self, operands: Operands) -> Shape:
        axis_value = self._axis_value()
        if axis_value is None:
            return Shape(dims=())
        # end if
        return operands[0].input_shape.drop_axis(axis_value)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError(f"{self.__name__} requires exactly 1 operand, got {len(operands)}")
        # end if
        axis_value = self._axis_value()
        if axis_value is not None and axis_value < 0:
            raise ValueError(f"Axis {axis_value} is negative.")
        elif axis_value is not None and axis_value >= operands[0].input_shape.rank:
            raise ValueError(f"Axis {axis_value} is out of bounds for rank {operands[0].input_shape.rank}.")
        # end if
        return True
    # end def check_shapes

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        return True
    # end def check_shapes

# end class AxisReductionOperator


class Sum(AxisReductionOperator):
    """
    Sum operator.
    """

    NAME = "sum"
    ARITY = 1

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        axis_value = self._axis_value()
        if axis_value is not None:
            return a.eval().sum(axis=axis_value)
        # end if
        return a.eval().sum()
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

    @classmethod
    def check_parameters(cls, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        return True
    # end def check_shapes

# end class Sum


class Mean(AxisReductionOperator):
    """
    Mean operator.
    """

    NAME = "mean"
    ARITY = 1

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return len(operands) == 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate mean.
        """
        a, = operands
        axis_value = self._axis_value()
        if axis_value is not None:
            return a.eval().mean(axis=axis_value)
        return a.eval().mean()
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Mean


class Std(AxisReductionOperator):
    """
    Standard deviation operator.
    """

    NAME = "std"
    ARITY = 1

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return len(operands) == 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate standard deviation."""
        a, = operands
        axis_value = self._axis_value()
        if axis_value is not None:
            return a.eval().std(axis=axis_value)
        # end if
        return a.eval().std()
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Std


class Median(AxisReductionOperator):
    r"""
    Median reduction following :func:`numpy.median`.

    Parameters
    ----------
    axis : MathNode or int or None, optional
        Axis along which the median is computed. When ``None``, the tensor is
        flattened prior to reduction.

    Returns
    -------
    Tensor
        Tensor containing the element-wise medians.

    Notes
    -----
    - Axis arguments must be non-negative and within range of the operand rank.
    - The dtype matches the operand dtype to mirror NumPy promotion rules for
      homogeneous inputs.

    Examples
    --------
    >>> x = pm.const(\"med_a\", data=[[1., 3.], [2., 4.]], dtype=pm.DType.R)
    >>> R.median(x, axis=0).eval()
    tensor([1.5, 3.5], dtype=float32)
    """

    NAME = "median"
    ARITY = 1
    _PERCENTILE = 50.0

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        axis_val = self._axis_value()
        if axis_val is not None:
            if self._PERCENTILE == 25.0:
                reduced = tensor.q1(axis=axis_val)
            elif self._PERCENTILE == 50.0:
                reduced = tensor.median(axis=axis_val)
            elif self._PERCENTILE == 75.0:
                reduced = tensor.q3(axis=axis_val)
            else:
                raise ValueError(f"Invalid percentile {self._PERCENTILE}.")
            # end if
        else:
            if self._PERCENTILE == 25.0:
                reduced = tensor.q1()
            elif self._PERCENTILE == 50.0:
                reduced = tensor.median()
            elif self._PERCENTILE == 75.0:
                reduced = tensor.q3()
            else:
                raise ValueError(f"Invalid percentile {self._PERCENTILE}.")
            # end if
        # end if
        return reduced
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Median


class Max(AxisReductionOperator):
    r"""
    Maximum reduction following :func:`numpy.max`.

    Parameters
    ----------
    axis : MathNode or int or None, optional
        Axis along which the maximum is computed. When ``None``, the tensor is
        flattened prior to reduction.

    Returns
    -------
    Tensor
        Tensor containing the maximum values.

    Examples
    --------
    >>> x = pm.const(\"max_a\", data=[[1, 2], [3, 4]], dtype=pm.DType.R)
    >>> R.max(x, axis=0).eval()
    tensor([3., 4.], dtype=float32)
    """

    NAME = "max"
    ARITY = 1

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        axis_val = self._axis_value()
        if axis_val is not None:
            reduced = tensor.max(axis=axis_val)
        else:
            reduced = tensor.max()
        # end if
        return reduced
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Max


class Min(AxisReductionOperator):
    r"""
    Minimum reduction following :func:`numpy.min`.

    Parameters
    ----------
    axis : MathNode or int or None, optional
        Axis along which the minimum is computed.

    Returns
    -------
    Tensor
        Tensor containing the minimum values.
    """

    NAME = "min"
    ARITY = 1

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        tensor = operands[0].eval()
        axis_val = self._axis_value()
        if axis_val is not None:
            reduced = tensor.min(axis=axis_val)
        else:
            reduced = tensor.min()
        # end if
        return reduced
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Min


class Q1(Median):
    r"""
    First quartile (25th percentile) reduction via :func:`numpy.percentile`.

    Parameters
    ----------
    axis : MathNode or int or None, optional
        Axis along which the percentile is computed.

    Returns
    -------
    Tensor
        Tensor containing the 25th percentile values.
    """

    NAME = "q1"
    ARITY = 1
    _PERCENTILE = 25.0

# end class Q1


class Q3(Q1):
    r"""
    Third quartile (75th percentile) reduction via :func:`numpy.percentile`.

    Parameters
    ----------
    axis : MathNode or int or None, optional
        Axis along which the percentile is computed.

    Returns
    -------
    Tensor
        Tensor containing the 75th percentile values.
    """

    NAME = "q3"
    _PERCENTILE = 75.0

# end class Q3


class Summation(ReductionOperator):
    """
    Summation operator.
    """

    NAME = "summation"
    ARITY = 1

    def __init__(
            self,
            lower: Union[MathNode, int],
            upper: Union[MathNode, int],
            i: str
    ):
        super().__init__(
            lower=lower,
            upper=upper,
            bounded_variable=i
        )
        if isinstance(lower, int):
            from ..math_leaves import const
            from ..random import random_const_name
            lower = const(random_const_name("summation-lower-"), lower)
        # end if
        if isinstance(upper, int):
            from ..math_leaves import const
            from ..random import random_const_name
            upper = const(random_const_name("summation-upper-"), upper)
        # end if
        self._lower = lower
        self._upper = upper
        self._i: str = i
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
    def bounded_var(self) -> str:
        """Bounded variable."""
        return self._i
    # end def bounded_variable

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        if len(operands) != 1:
            raise ValueError(f"{self.NAME} requires exactly 1 operand, got {len(operands)}")
        # end if
        body, = operands
        index_found = body.contains_variable(self.bounded_var, check_operator=False, by_ref=False)
        if not index_found:
            raise ValueError(f"Bounded variable {self.bounded_var} not found in body : {body}.")
        # end if
        return True
    # end def check_operands

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        ret = [o.contains(expr, by_ref=by_ref, look_for=look_for) for o in [self.lower, self.upper]]
        return any(ret)
    # end def contains

    @classmethod
    def check_parameters(cls, lower: MathNode, upper: MathNode, i: str) -> bool:
        """Check that the operands have compatible shapes."""
        # Lower must be int
        if lower.dtype not in {DType.Z}:
            raise ValueError(f"Lower bound must be int, got {lower.dtype}")
        # end if

        # Upper must be int
        if upper.dtype not in {DType.Z}:
            raise ValueError(f"Upper bound must be int, got {upper.dtype}")
        # end if

        return True
    # end def check_shapes

    def check_shapes(self, operands: Operands) -> bool:
        if len(operands) != self.ARITY:
            raise ValueError(f"Sum requires exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        body, = operands
        return body.input_shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, = operands
        return a.dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate summation.
        """
        # Get body
        body, = operands

        # Evaluate lower and upper bound
        lower = self.lower.eval().item()
        upper = self.upper.eval().item()

        # Iterate
        ret_value = Tensor.zeros(body.input_shape, dtype=body.dtype)
        with new_context():
            for i in range(lower, upper + 1):
                set_value(self._i, i)
                ret_value += body.eval()
            # end for
        # end with

        return ret_value
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(lower={self._format_bound(self._lower)}, "
            f"upper={self._format_bound(self._upper)}, i='{self._i}')"
        )
    # end def __str__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lower={self._format_bound(self._lower)}, "
            f"upper={self._format_bound(self._upper)}, i='{self._i}')"
        )
    # end def __repr__

    def _format_bound(self, bound):
        if hasattr(bound, "eval"):
            evaluated = bound.eval()
            if hasattr(evaluated, "item"):
                return evaluated.item()
            return evaluated
        return bound
    # end def _format_bound

# end class Summation


class Product(ReductionOperator):
    r"""
    Discrete product operator mirroring :func:`numpy.prod`.

    Product multiplies the operand body evaluated at every integer value of a
    bounded variable.  The bounds are inclusive and may be provided as Python
    integers or scalar math expressions.  During execution, the bounded
    variable is assigned each value between ``lower`` and ``upper`` and the
    body expression is evaluated and multiplied into the accumulator.  This
    mirrors NumPy's ``prod`` semantics while keeping the computation graph
    symbolic for inspection, differentiation, or LaTeX rendering.

    Parameters
    ----------
    lower : MathNode or int
        Inclusive lower bound for the bounded variable.
    upper : MathNode or int
        Inclusive upper bound for the bounded variable.
    i : str
        Name of the bounded variable updated inside the product.

    Notes
    -----
    - Bounds must evaluate to ``INT32`` or ``INT64`` tensors.
    - When ``lower > upper`` the loop body is skipped and the multiplicative
      identity (ones) is returned, matching ``numpy.prod`` on empty slices.

    Examples
    --------
    >>> i = pm.var("i", dtype=pm.DType.Z, shape=())
    >>> expr = R.product(i + 1, lower=1, upper=3, bounded_variable="i")
    >>> expr.eval()
    tensor(24., dtype=float32)
    """

    NAME = "product"
    ARITY = 1

    def __init__(
            self,
            lower: Union[MathNode, int],
            upper: Union[MathNode, int],
            i: str
    ):
        super().__init__(
            lower=lower,
            upper=upper,
            bounded_variable=i
        )
        if isinstance(lower, int):
            from ..math_leaves import const
            from ..random import random_const_name
            lower = const(random_const_name("product-lower-"), lower)
        # end if
        if isinstance(upper, int):
            from ..math_leaves import const
            from ..random import random_const_name
            upper = const(random_const_name("product-upper-"), upper)
        # end if
        self._lower = lower
        self._upper = upper
        self._i: str = i
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
    def bounded_var(self) -> str:
        """Bounded variable."""
        return self._i
    # end def bounded_variable

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        if len(operands) != 1:
            raise ValueError(f"{self.NAME} requires exactly 1 operand, got {len(operands)}")
        body, = operands
        index_found = body.contains_variable(self.bounded_var, check_operator=False, by_ref=False)
        if not index_found:
            raise ValueError(f"Bounded variable {self.bounded_var} not found in body : {body}.")
        return True
    # end def check_operands

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        ret = [o.contains(expr, by_ref=by_ref, look_for=look_for) for o in [self.lower, self.upper]]
        return any(ret)
    # end def contains

    @classmethod
    def check_parameters(cls, lower: MathNode, upper: MathNode, i: str) -> bool:
        """Check that the operands have compatible shapes."""
        if lower.dtype not in {DType.Z}:
            raise ValueError(f"Lower bound must be int, got {lower.dtype}")
        # end if
        if upper.dtype not in {DType.Z}:
            raise ValueError(f"Upper bound must be int, got {upper.dtype}")
        # end if
        return True
    # end def check_parameters

    def check_shapes(self, operands: Operands) -> bool:
        if len(operands) != self.ARITY:
            raise ValueError(f"{self.NAME} requires exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        body, = operands
        return body.input_shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, = operands
        return a.dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        """Evaluate product."""
        body, = operands

        lower = self.lower.eval().item()
        upper = self.upper.eval().item()

        dtype = body.dtype
        init = np.ones(shape=body.input_shape.dims, dtype=to_numpy(dtype))
        ret_value = Tensor(data=init, dtype=dtype)

        with new_context():
            for i in range(lower, upper + 1):
                set_value(self._i, i)
                ret_value = ret_value * body.eval()
            # end for
        # end with

        return ret_value
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(lower={self._format_bound(self._lower)}, "
            f"upper={self._format_bound(self._upper)}, i='{self._i}')"
        )
    # end def __str__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lower={self._format_bound(self._lower)}, "
            f"upper={self._format_bound(self._upper)}, i='{self._i}')"
        )
    # end def __repr__

    def _format_bound(self, bound):
        if hasattr(bound, "eval"):
            evaluated = bound.eval()
            if hasattr(evaluated, "item"):
                return evaluated.item()
            return evaluated
        return bound
    # end def _format_bound

# end class Product


operator_registry.register(Sum)
operator_registry.register(Mean)
operator_registry.register(Std)
operator_registry.register(Median)
operator_registry.register(Max)
operator_registry.register(Min)
operator_registry.register(Q1)
operator_registry.register(Q3)
operator_registry.register(Summation)
operator_registry.register(Product)
