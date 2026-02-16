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
Statistical sampling operators.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np

from ..build import as_expr
from ..dtype import DType, to_numpy
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import Tensor
from ..typing import MathExpr
from .base import Operands, operator_registry
from .builders import ParametricBuilder


ScalarParameter = Union[MathExpr, int, float]


__all__ = [
    "Normal",
    "Uniform",
    "RandInt",
    "Poisson",
    "Bernoulli",
    "Covariance",
    "Correlation",
    "ZScore",
]


def _ensure_scalar_parameter(value: ScalarParameter, name: str) -> MathExpr:
    expr = as_expr(value)
    if expr.rank != 0:
        raise ValueError(f"{name} must be a scalar expression.")
    return expr
# end def _ensure_scalar_parameter


def _eval_scalar_parameter(expr: MathExpr, name: str) -> float:
    value = np.asarray(expr.eval().value)
    if value.shape != ():
        raise ValueError(f"{name} must evaluate to a scalar.")
    return float(value.item())
# end def _eval_scalar_parameter


def _eval_int_parameter(expr: MathExpr, name: str) -> int:
    value = _eval_scalar_parameter(expr, name)
    if not float(value).is_integer():
        raise ValueError(f"{name} must evaluate to an integer.")
    return int(value)
# end def _eval_int_parameter


class Normal(ParametricBuilder):
    """Tensor sampled from a normal distribution."""

    NAME = "normal"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(
            self,
            shape: Shape,
            loc: ScalarParameter = 0.0,
            scale: ScalarParameter = 1.0,
            dtype: DType = DType.R
    ):
        loc_expr = _ensure_scalar_parameter(loc, f"{self.NAME} loc")
        scale_expr = _ensure_scalar_parameter(scale, f"{self.NAME} scale")
        super().__init__(shape=shape, loc=loc_expr, scale=scale_expr, dtype=dtype)
        self._shape = shape
        self._loc = loc_expr
        self._scale = scale_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._loc.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._scale.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            loc: ScalarParameter = 0.0,
            scale: ScalarParameter = 1.0,
            dtype: DType = DType.R
    ) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        loc = _eval_scalar_parameter(self._loc, f"{self.NAME} loc")
        scale = _eval_scalar_parameter(self._scale, f"{self.NAME} scale")
        if scale < 0:
            raise ValueError("Normal requires scale >= 0.")
        # end if
        numpy_dtype = to_numpy(self._dtype)
        data = np.random.normal(loc=loc, scale=scale, size=self._shape.dims)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Normal does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(shape={self._shape.dims}, "
            f"loc={self._loc.name}, "
            f"scale={self._scale.name}, "
            f"dtype={self._dtype.name})"
        )
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Normal


class Uniform(ParametricBuilder):
    """Tensor sampled from a uniform distribution."""

    NAME = "uniform"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(
            self,
            shape: Shape,
            low: ScalarParameter = 0.0,
            high: ScalarParameter = 1.0,
            dtype: DType = DType.R
    ):
        low_expr = _ensure_scalar_parameter(low, f"{self.NAME} low")
        high_expr = _ensure_scalar_parameter(high, f"{self.NAME} high")
        super().__init__(shape=shape, low=low_expr, high=high_expr, dtype=dtype)
        self._shape = shape
        self._low = low_expr
        self._high = high_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._low.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._high.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            low: ScalarParameter = 0.0,
            high: ScalarParameter = 1.0,
            dtype: DType = DType.R
    ) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        low = _eval_scalar_parameter(self._low, f"{self.NAME} low")
        high = _eval_scalar_parameter(self._high, f"{self.NAME} high")
        if high <= low:
            raise ValueError("Uniform requires high > low.")
        # end if
        numpy_dtype = to_numpy(self._dtype)
        data = np.random.uniform(low=low, high=high, size=self._shape.dims)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Uniform does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return (
            f"{self.NAME}(shape={self._shape.dims}, "
            f"low={self._low.name}, "
            f"high={self._high.name}, "
            f"dtype={self._dtype.name})"
        )
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Uniform


class RandInt(ParametricBuilder):
    """Tensor sampled from a discrete uniform integer distribution."""

    NAME = "randint"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(
            self,
            shape: Shape,
            low: ScalarParameter,
            high: Optional[ScalarParameter] = None,
            dtype: DType = DType.Z
    ):
        low_expr = _ensure_scalar_parameter(low, f"{self.NAME} low")
        high_expr = _ensure_scalar_parameter(high, f"{self.NAME} high") if high is not None else None
        super().__init__(shape=shape, low=low_expr, high=high_expr, dtype=dtype)
        self._shape = shape
        self._low = low_expr
        self._high = high_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        if self._high is None:
            return self._low.contains(expr, by_ref=by_ref, look_for=look_for)
        # end if
        return (
            self._low.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._high.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            low: ScalarParameter,
            high: Optional[ScalarParameter] = None,
            dtype: DType = DType.Z
    ) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        low = _eval_int_parameter(self._low, f"{self.NAME} low")
        high = _eval_int_parameter(self._high, f"{self.NAME} high") if self._high is not None else None
        if high is None:
            if low <= 0:
                raise ValueError("RandInt requires low > 0 when high is None.")
            # end if
        elif high <= low:
            raise ValueError("RandInt requires high > low.")
        # end if
        numpy_dtype = to_numpy(self._dtype)
        data = np.random.randint(low=low, high=high, size=self._shape.dims)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("RandInt does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        high = self._high.name if self._high is not None else None
        return f"{self.NAME}(shape={self._shape.dims}, low={self._low.name}, high={high}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class RandInt


class Poisson(ParametricBuilder):
    """Tensor sampled from a Poisson distribution."""

    NAME = "poisson"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(
            self,
            shape: Shape,
            lam: ScalarParameter = 1.0,
            dtype: DType = DType.Z
    ):
        lam_expr = _ensure_scalar_parameter(lam, f"{self.NAME} lam")
        super().__init__(shape=shape, lam=lam_expr, dtype=dtype)
        self._shape = shape
        self._lam = lam_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return self._lam.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            lam: ScalarParameter = 1.0,
            dtype: DType = DType.Z
    ) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        lam = _eval_scalar_parameter(self._lam, f"{self.NAME} lam")
        if lam < 0:
            raise ValueError("Poisson requires lam >= 0.")
        # end if
        numpy_dtype = to_numpy(self._dtype)
        data = np.random.poisson(lam=lam, size=self._shape.dims)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Poisson does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(shape={self._shape.dims}, lam={self._lam.name}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Poisson


class Bernoulli(ParametricBuilder):
    """Tensor sampled from independent Bernoulli trials."""

    NAME = "bernoulli"
    ARITY = 0
    IS_VARIADIC = False

    def __init__(
            self,
            shape: Shape,
            p: ScalarParameter = 0.5,
            dtype: DType = DType.Z
    ):
        p_expr = _ensure_scalar_parameter(p, f"{self.NAME} p")
        super().__init__(shape=shape, p=p_expr, dtype=dtype)
        self._shape = shape
        self._p = p_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return self._p.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            shape: Shape,
            p: ScalarParameter = 0.5,
            dtype: DType = DType.Z
    ) -> bool:
        return shape.n_dims >= 0
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 0
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        p = _eval_scalar_parameter(self._p, f"{self.NAME} p")
        if p < 0.0 or p > 1.0:
            raise ValueError("Bernoulli requires p in [0, 1].")
        # end if
        numpy_dtype = to_numpy(self._dtype)
        data = np.random.binomial(n=1, p=p, size=self._shape.dims)
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Bernoulli does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(shape={self._shape.dims}, p={self._p.name}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Bernoulli


class Covariance(ParametricBuilder):
    """Covariance scalar or matrix operator."""

    NAME = "cov"
    ARITY = 1
    IS_VARIADIC = True

    def __init__(
            self,
            rowvar: bool = False,
            bias: bool = False,
            ddof: Optional[ScalarParameter] = None,
            dtype: DType = DType.R
    ):
        ddof_expr = _ensure_scalar_parameter(ddof, f"{self.NAME} ddof") if ddof is not None else None
        super().__init__(rowvar=rowvar, bias=bias, ddof=ddof_expr, dtype=dtype)
        self._rowvar = bool(rowvar)
        self._bias = bool(bias)
        self._ddof = ddof_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        if self._ddof is None:
            return False
        # end if
        return self._ddof.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            rowvar: bool = False,
            bias: bool = False,
            ddof: Optional[ScalarParameter] = None,
            dtype: DType = DType.R
    ) -> bool:
        return True
    # end def check_parameters

    @classmethod
    def check_arity(cls, operands: Operands) -> bool:
        return 1 <= len(operands) <= 2
    # end def check_arity

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) not in {1, 2}:
            raise ValueError(f"{self.NAME} expects 1 or 2 operands, got {len(operands)}")
        # end if
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x = operands[0]
        if x.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects rank-1 or rank-2 input, got rank={x.rank}")
        # end if
        if len(operands) == 2 and operands[1].rank != 1:
            raise ValueError(f"{self.NAME} second operand must be rank-1, got rank={operands[1].rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        if len(operands) == 2:
            return Shape.matrix(2, 2)
        # end if
        x = operands[0]
        if x.rank == 1:
            return Shape.scalar()
        # end if
        n_features = x.shape[0] if self._rowvar else x.shape[1]
        return Shape.matrix(n_features, n_features)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        y = np.asarray(operands[1].eval().value) if len(operands) == 2 else None
        ddof = _eval_int_parameter(self._ddof, f"{self.NAME} ddof") if self._ddof is not None else None
        out = np.cov(
            m=x,
            y=y,
            rowvar=self._rowvar,
            bias=self._bias,
            ddof=ddof,
        )
        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(out, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Covariance does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        ddof = self._ddof.name if self._ddof is not None else None
        return f"{self.NAME}(rowvar={self._rowvar}, bias={self._bias}, ddof={ddof}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Covariance


class Correlation(ParametricBuilder):
    """Correlation scalar or matrix operator."""

    NAME = "corr"
    ARITY = 1
    IS_VARIADIC = True

    def __init__(
            self,
            rowvar: bool = False,
            dtype: DType = DType.R
    ):
        super().__init__(rowvar=rowvar, dtype=dtype)
        self._rowvar = bool(rowvar)
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return False
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            rowvar: bool = False,
            dtype: DType = DType.R
    ) -> bool:
        return True
    # end def check_parameters

    @classmethod
    def check_arity(cls, operands: Operands) -> bool:
        return 1 <= len(operands) <= 2
    # end def check_arity

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) not in {1, 2}:
            raise ValueError(f"{self.NAME} expects 1 or 2 operands, got {len(operands)}")
        # end if
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        x = operands[0]
        if x.rank not in {1, 2}:
            raise ValueError(f"{self.NAME} expects rank-1 or rank-2 input, got rank={x.rank}")
        # end if
        if len(operands) == 2 and operands[1].rank != 1:
            raise ValueError(f"{self.NAME} second operand must be rank-1, got rank={operands[1].rank}")
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        if len(operands) == 2:
            return Shape.matrix(2, 2)
        # end if
        x = operands[0]
        if x.rank == 1:
            return Shape.scalar()
        # end if
        n_features = x.shape[0] if self._rowvar else x.shape[1]
        return Shape.matrix(n_features, n_features)
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        y = np.asarray(operands[1].eval().value) if len(operands) == 2 else None
        out = np.corrcoef(
            x=x,
            y=y,
            rowvar=self._rowvar,
        )
        numpy_dtype = to_numpy(self._dtype)
        return Tensor(data=np.asarray(out, dtype=numpy_dtype), dtype=self._dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Correlation does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(rowvar={self._rowvar}, dtype={self._dtype.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Correlation


class ZScore(ParametricBuilder):
    """Z-score normalization operator."""

    NAME = "zscore"
    ARITY = 1

    def __init__(
            self,
            axis: Optional[int] = None,
            ddof: ScalarParameter = 0,
            eps: ScalarParameter = 1e-8,
            dtype: Optional[DType] = None
    ):
        ddof_expr = _ensure_scalar_parameter(ddof, f"{self.NAME} ddof")
        eps_expr = _ensure_scalar_parameter(eps, f"{self.NAME} eps")
        super().__init__(axis=axis, ddof=ddof_expr, eps=eps_expr, dtype=dtype)
        self._axis = axis
        self._ddof = ddof_expr
        self._eps = eps_expr
        self._dtype = dtype
    # end def __init__

    def contains(self, expr: MathNode, by_ref: bool = False, look_for: Optional[str] = None) -> bool:
        return (
            self._ddof.contains(expr, by_ref=by_ref, look_for=look_for)
            or self._eps.contains(expr, by_ref=by_ref, look_for=look_for)
        )
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            axis: Optional[int] = None,
            ddof: ScalarParameter = 0,
            eps: ScalarParameter = 1e-8,
            dtype: Optional[DType] = None
    ) -> bool:
        return True
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != 1:
            raise ValueError(f"{self.NAME} expects exactly 1 operand, got {len(operands)}")
        # end if
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        if self._axis is not None:
            rank = operands[0].rank
            if self._axis < -rank or self._axis >= rank:
                raise ValueError(f"axis {self._axis} is out of bounds for rank {rank}")
            # end if
        # end if
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        if self._dtype is not None:
            return self._dtype
        # end if
        return operands[0].dtype if operands[0].dtype in {DType.R, DType.C} else DType.R
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        x = np.asarray(operands[0].eval().value)
        ddof = _eval_int_parameter(self._ddof, f"{self.NAME} ddof")
        eps = _eval_scalar_parameter(self._eps, f"{self.NAME} eps")
        if eps <= 0.0:
            raise ValueError("zscore requires eps > 0.")
        # end if

        axis = self._axis
        keepdims = axis is not None
        mean = np.mean(x, axis=axis, keepdims=keepdims)
        std = np.std(x, axis=axis, ddof=ddof, keepdims=keepdims)
        out = (x - mean) / (std + eps)

        dtype = self.infer_dtype(operands)
        numpy_dtype = to_numpy(dtype)
        return Tensor(data=np.asarray(out, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("ZScore does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        dtype_name = self._dtype.name if self._dtype is not None else "auto"
        return (
            f"{self.NAME}(axis={self._axis}, ddof={self._ddof.name}, "
            f"eps={self._eps.name}, dtype={dtype_name})"
        )
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class ZScore


operator_registry.register(Normal)
operator_registry.register(Uniform)
operator_registry.register(RandInt)
operator_registry.register(Poisson)
operator_registry.register(Bernoulli)
operator_registry.register(Covariance)
operator_registry.register(Correlation)
operator_registry.register(ZScore)
