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
Linear algebra operator implementations.
"""

# Imports
from abc import ABC
from typing import Optional, Sequence, Union, Any, List
import numpy as np

from ..math_leaves import const
from ..random import random_const_name
from ..dtype import DType, to_numpy, promote
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import Tensor, einsum
from ..typing import MathExpr, LeafKind, OperatorSpec, AritySpec, OpAssociativity
from .base import Operands, operator_registry, OperatorBase


__all__ = [
    "LinearAlgebraOperator",
    "LinearAlgebraParametricOperator",
    "MatMul",
    "Dot",
    "Outer",
    "Trace",
    "Transpose",
    "Det",
    "Inverse",
    "Norm",
    "InftyNorm",
    "FrobeniusNorm",
]


def _la_spec(name: str, *, exact: int | None, min_operands: int, variadic: bool,
             commutative: bool = False, associative: bool = False) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        arity=AritySpec(exact=exact, min_operands=min_operands, variadic=variadic),
        symbol=name,
        precedence=30,
        associativity=OpAssociativity.NONE,
        commutative=commutative,
        associative=associative,
        is_diff=False,
    )
# end def _la_spec


class LinearAlgebraOperator(OperatorBase, ABC):
    """
    Linear algebra operator.
    """

    def check_operands(self, operands: Operands) -> bool:
        """Check that the operands have the correct arity."""
        return True
    # end def check_operands

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        return False
    # end def contains

    def __str__(self) -> str:
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arity={self.ARITY})"
    # end def __repr__

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return promote(a.dtype, b.dtype)
    # end def infer_dtype

    def check_parameters(self, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        return True
    # end def check_shapes

    def _needs_parentheses(self, *args, **kwargs):
        return None
    # end def _needs_parentheses

    def print(self, operands: Operands, **kwargs) -> str:
        return str(self)
    # end def print

# end class LinearAlgebraOperator


class LinearAlgebraParametricOperator(LinearAlgebraOperator, ABC):
    """Linear algebra parametric operator."""

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: LeafKind = LeafKind.ANY
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric operators must implement contains(..).")
    # end def contains

# end class LinearAlgebraParametricOperator


class MatMul(LinearAlgebraOperator):
    """
    Matrix Multiplication operator.
    """

    SPEC = _la_spec("matmul", exact=2, min_operands=2, variadic=False, commutative=False, associative=True)

    NAME = "matmul"
    ARITY = 2
    COMMUTATIVE = False
    ASSOCIATIVE = True

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, b = operands

        # Rank >= 1
        if a.rank < 1 or b.rank < 1:
            raise ValueError(
                f"MatMul requires at least 1D operands, got {a.rank} and {b.rank}"
            )
        # end if

        # Vector-vector forbidden
        if a.rank == 1 and b.rank == 1:
            raise ValueError("MatMul does not support vector-vector multiplication.")
        # end if

        # Evaluate shape
        a_shape = a.shape.eval()
        b_shape = b.shape.eval()

        # -------- Matrix @ Matrix (batched) --------
        if a.rank >= 2 and b.rank >= 2:
            if a_shape[-1] != b_shape[-2]:
                raise ValueError(
                    f"Matrix-matrix requires contraction dims to match, "
                    f"got {a_shape[-1]} and {b_shape[-2]}"
                )
            # end if

            if a_shape[:-2] != b_shape[:-2]:
                raise ValueError(
                    f"MatMul requires batch dimensions to match, "
                    f"got {a_shape[:-2]} and {b_shape[:-2]}"
                )
            # end if

            return True
        # end if

        # -------- Matrix @ Vector --------
        if a.rank >= 2 and b.rank == 1:
            if a_shape[-1] != b_shape[-1]:
                raise ValueError(
                    f"Matrix-vector requires contraction dims to match, "
                    f"got {a_shape[-1]} and {b_shape[-1]}"
                )
            # end if

            if a_shape[:-2] != b_shape[:-1]:
                raise ValueError(
                    f"MatMul requires batch dimensions to match, "
                    f"got {a_shape[:-2]} and {b_shape[:-1]}"
                )
            # end if

            return True
        # end if

        # -------- Vector @ Matrix --------
        if a.rank == 1 and b.rank >= 2:
            if a_shape[-1] != b_shape[-2]:
                raise ValueError(
                    f"Vector-matrix requires contraction dims to match, "
                    f"got {a_shape[-1]} and {b_shape[-2]}"
                )
            # end if

            if a_shape[:-1] != b_shape[:-2]:
                raise ValueError(
                    f"MatMul requires batch dimensions to match, "
                    f"got {a_shape[:-1]} and {b_shape[:-2]}"
                )
            # end if

            return True
        # end if

        raise RuntimeError("Invalid matmul operand configuration")
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        A, B = operands
        dims_a = A.shape.dims
        dims_b = B.shape.dims

        rank_a = len(dims_a)
        rank_b = len(dims_b)

        # Matrix @ Matrix (possibly batched)
        if rank_a >= 2 and rank_b >= 2:
            batch = dims_a[:-2]
            return Shape(dims=list(batch) + [dims_a[-2], dims_b[-1]])
        # end if

        # Matrix @ Vector
        if rank_a >= 2 and rank_b == 1:
            batch = dims_a[:-2]
            return Shape(dims=list(batch) + [dims_a[-2]])
        # end if

        # Vector @ Matrix
        if rank_a == 1 and rank_b >= 2:
            batch = dims_b[:-2]
            return Shape(dims=list(batch) + [dims_b[-1]])
        # end if

        # Vector @ Vector is excluded by design
        raise RuntimeError("infer_shape called with invalid matmul operands")
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, b = operands
        return a.eval() @ b.eval()
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class MatMul


class Dot(LinearAlgebraOperator):
    """
    Dot product operator.
    """

    SPEC = _la_spec("dot", exact=2, min_operands=2, variadic=False, commutative=True, associative=False)

    NAME = "dot"
    ARITY = 2
    COMMUTATIVE = True
    ASSOCIATIVE = False

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, b = operands

        a_shape = a.shape.eval()
        b_shape = b.shape.eval()

        # Same rank
        if a.rank != b.rank:
            raise ValueError(f"Dot requires same rank operands, got {a.rank} and {b.rank}")
        # end if

        # Last dim equivalent
        if a_shape[-1] != b_shape[-1]:
            raise ValueError(f"Dot requires last dim to match, got {a_shape[-1]} and {b_shape[-1]}")
        # end if

        # Equivalent batch dims
        if a.rank > 1 and a_shape[:-1] != b_shape[:-1]:
            raise ValueError(f"Dot requires batch dims to match, got {a_shape[:-1]} and {b_shape[:-1]}")
        # end if

        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        A, B = operands
        dims_a = A.shape.dims
        rank_a = len(dims_a)

        # Batch dim
        batch_dim = []
        if rank_a > 1:
            batch_dim = dims_a[:-1]
        # end if

        return Shape(dims=batch_dim)
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, b = operands
        if a.rank == 1:
            return einsum("i,i->", a.eval(), b.eval())
        else:
            return einsum("...i,...i->...", a.eval(), b.eval())
        # end if
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Dot


class Outer(LinearAlgebraOperator):
    """
    Outer product operator.
    """

    SPEC = _la_spec("outer", exact=2, min_operands=2, variadic=False, commutative=False, associative=False)

    NAME = "outer"
    ARITY = 2
    COMMUTATIVE = False
    ASSOCIATIVE = False

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, b = operands

        a_shape = a.shape.eval()
        b_shape = b.shape.eval()

        # Same rank
        if a.rank != b.rank:
            raise ValueError(f"Outer requires same rank operands, got {a.rank} and {b.rank}")
        # end if

        # Equivalent batch dims
        if a.rank > 1 and a_shape[:-1] != b_shape[:-1]:
            raise ValueError(f"Outer requires batch dims to match, got {a_shape[:-1]} and {b_shape[:-1]}")
        # end if

        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        A, B = operands
        dims_a = A.shape.dims
        dims_b = B.shape.dims
        rank_a = len(dims_a)

        # Batch dim
        batch_dim = []
        if rank_a > 1:
            batch_dim = dims_a[:-1]
        # end if

        return Shape(dims=batch_dim + [dims_a[-1], dims_b[-1]])
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, b = operands
        if a.rank == 1:
            return einsum("i,j->ij", a.eval(), b.eval())
        else:
            return einsum("...i,...j->...ij", a.eval(), b.eval())
        # end if
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Outer


class Trace(LinearAlgebraOperator):
    """
    Trace operator.
    """

    SPEC = _la_spec("trace", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "trace"
    ARITY = 1
    COMMUTATIVE = False
    ASSOCIATIVE = False

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, = operands

        a_shape = a.shape.eval()

        # Same rank
        if a.rank < 2:
            raise ValueError(f"Trace requires rank >= 2, got {a.rank}")
        # end if

        # Square matrices
        if a_shape[-1] != a_shape[-2]:
            raise ValueError(f"Trace requires square matrix, got {a_shape[-1]}x{a_shape[-2]}")
        # end if

        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        A, = operands
        dims_a = A.shape.dims
        rank_a = len(dims_a)

        # Batch dim
        batch_dim = []
        if rank_a > 2:
            batch_dim = dims_a[:-2]
        # end if

        return Shape(dims=batch_dim)
    # end def infer_shape

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, = operands
        return a.dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        return Tensor.trace(a.eval(), axis1=-2, axis2=-1)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Trace


class Transpose(LinearAlgebraParametricOperator):
    """
    Transpose operator.
    """

    SPEC = _la_spec("transpose", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "transpose"
    ARITY = 1
    COMMUTATIVE = False
    ASSOCIATIVE = False

    def __init__(self, axes: Optional[Union[MathNode, List[int]]] = None, **kwargs: Any):
        """
        Transpose
        """
        super(Transpose, self).__init__(axes=axes)
        from ..math_leaves import Variable
        if axes and isinstance(axes, list):
            # from ..utils import random_const_name, const
            self._check_axes(axes)
            axes = const(random_const_name(f"{self.__class__.__name__.lower()}-axes-"), axes, dtype=DType.Z)
        elif axes and isinstance(axes, Variable):
            # Only constant (otherwise we have dynamic shapes)
            raise ValueError(f"{self.__class__.__name__} does not support dynamic axes.")
        # end if
        self._axes = axes
    # end def __init__

    def _check_axes(self, axes: Sequence[int]) -> None:
        if len(axes) != len(set(axes)):
            raise ValueError(f"Duplicate axes {axes} in transpose.")
        # end if
        for axis in axes:
            if type(axis) != int:
                raise ValueError(f"Invalid axis type {type(axis)} in transpose.")
            # end if
            if axis < 0:
                raise ValueError(f"Invalid axis {axis} in transpose.")
            # end if
        # end for
    # end def _check_axes

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        axes = self._axes.eval().tolist() if self._axes else None
        return a.eval().transpose(axes=axes)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("Transpose does not support backward.")
    # end def backward

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a,  = operands
        return a.dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        a: 'MathExpr' = operands[0]
        if self._axes and isinstance(self._axes, MathExpr):
            axes = self._axes.eval().tolist()
        else:
            axes = self._axes
        # end if
        return a.shape.transpose(axes=axes)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        return True
    # end def check_shapes

    def __str__(self) -> str:
        axes = self._axes.eval().tolist() if self._axes else None
        return f"{self.NAME}(axes={axes})"
    # end def __str__

    def __repr__(self) -> str:
        axes = self._axes.eval().tolist() if self._axes else None
        return f"{self.__class__.__name__}(axes={axes})"
    # end def __repr__

# end class Transpose


class Det(LinearAlgebraOperator):
    """
    Compute the determinant of a square matrix.
    """

    SPEC = _la_spec("det", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "det"
    ARITY = 1
    COMMUTATIVE = False
    ASSOCIATIVE = False

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        return a.eval().det()
    # end def _eval

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a,  = operands
        return a.dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        a: 'MathExpr' = operands[0]
        new_shape = a.shape.copy()
        new_shape = new_shape[:-2]
        return Shape(dims=new_shape)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        a: 'MathExpr' = operands[0]
        a_shape = a.shape.eval()
        return a.rank >= 2 and a_shape[-1] == a_shape[-2]
    # end def check_shapes

# end class Det


class Inverse(LinearAlgebraOperator):
    """
    Compute the inverse of a square matrix.
    """

    SPEC = _la_spec("inverse", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "inverse"
    ARITY = 1
    COMMUTATIVE = False
    ASSOCIATIVE = False

    def infer_shape(self, operands: Operands) -> Shape:
        a: 'MathExpr' = operands[0]
        return a.shape.copy()
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Infer the dtype from the input dtype.
        """
        return DType.R
    # end def infer_dtype

    def check_shapes(self, operands: Operands) -> bool:
        a: 'MathExpr' = operands[0]
        return a.rank >= 2 and a.shape[-1] == a.shape[-2]
    # end def check_shapes

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        return a.eval().inverse()
    # end def _eval

# end class Inverse


class Norm(LinearAlgebraParametricOperator):
    """
    Compute the norm of a vector.
    """
    
    SPEC = _la_spec("norm", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "norm"
    ARITY = 1
    IS_SCALAR = False
    COMMUTATIVE = False
    ASSOCIATIVE = False

    def __init__(self, order: Union[MathExpr, int, float] = None):
        super(Norm, self).__init__(order=order)
        # from ..utils import random_const_name
        if not order:
            self._order = const(
                name=random_const_name(f"{self.__class__.__name__.lower()}-order-"),
                data=2,
                dtype=DType.Z
            )
        elif isinstance(order, (int, float)):
            self._order = const(
                name=random_const_name(f"{self.__class__.__name__.lower()}-order-"),
                data=order,
                dtype=DType.R
            )
        else:
            self._order = order
        # end if
    # end def __init__

    def infer_shape(
            self,
            operands: Operands
    ) -> Shape:
        """Infer the shape of the output."""
        return Shape.scalar()
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """Infer the dtype of the output."""
        dtype = operands[0].dtype
        if dtype in {DType.R, DType.C}:
            return dtype
        return DType.R
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        vector, = operands
        order_value = self._resolve_parameter(self._order)
        if order_value is None:
            order_value = 2
        # end if
        if not isinstance(order_value, (int, float)):
            raise ValueError(f"Invalid order value {order_value} for norm.")
        # end if
        result = vector.eval().norm(order=order_value)
        dtype = self.infer_dtype(operands)
        if result.dtype != dtype:
            result = result.astype(dtype)
        return result
    # end def _eval

    def check_shapes(self, operands: Operands) -> bool:
        return operands[0].rank == 1
    # end def check_shapes

# end class Norm


class InftyNorm(LinearAlgebraOperator):
    """
    Infinity norm returning the maximum absolute element per vector entry.
    """

    SPEC = _la_spec("infty_norm", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "infty_norm"
    ARITY = 1
    COMMUTATIVE = False
    ASSOCIATIVE = False

    @staticmethod
    def _scalar_or_batch_shape(vector: MathExpr) -> Shape:
        if vector.rank <= 1:
            return Shape.scalar()
        # end if
        return Shape(dims=vector.shape.dims[:-1])
    # end def _scalar_or_batch_shape

    def infer_shape(self, operands: Operands) -> Shape:
        return self._scalar_or_batch_shape(operands[0])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        if dtype in {DType.R, DType.C}:
            return dtype
        # end if
        return DType.R
    # end def infer_dtype

    def check_shapes(self, operands: Operands) -> bool:
        vector = operands[0]
        if vector.rank < 1:
            raise ValueError("InftyNorm expects at least a 1-D vector.")
        # end if
        return True
    # end def check_shapes

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        vector, = operands
        dtype = self.infer_dtype(operands)
        data = vector.eval().value.astype(to_numpy(dtype), copy=False)
        result = np.max(np.abs(data), axis=-1)
        return Tensor(data=np.asarray(result, dtype=to_numpy(dtype)), dtype=dtype)
    # end def _eval

# end class InftyNorm


class FrobeniusNorm(LinearAlgebraOperator):
    """
    Frobenius norm computed over the last two axes of a tensor.
    """

    SPEC = _la_spec("frobenius_norm", exact=1, min_operands=1, variadic=False, commutative=False, associative=False)

    NAME = "frobenius_norm"
    ARITY = 1
    COMMUTATIVE = False
    ASSOCIATIVE = False

    def infer_shape(self, operands: Operands) -> Shape:
        matrix = operands[0]
        if matrix.rank <= 2:
            return Shape.scalar()
        # end if
        return Shape(dims=matrix.shape.dims[:-2])
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        if dtype in {DType.R, DType.C}:
            return dtype
        # end if
        return DType.R
    # end def infer_dtype

    def check_shapes(self, operands: Operands) -> bool:
        matrix = operands[0]
        if matrix.rank < 2:
            raise ValueError("FrobeniusNorm expects at least a 2-D matrix.")
        # end if
        return True
    # end def check_shapes

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        matrix, = operands
        dtype = self.infer_dtype(operands)
        values = matrix.eval().value.astype(to_numpy(dtype), copy=False)
        squared = np.square(values)
        sum_axes = (-2, -1)
        summed = np.sum(squared, axis=sum_axes)
        result = np.sqrt(summed)
        return Tensor(data=np.asarray(result, dtype=to_numpy(dtype)), dtype=dtype)
    # end def _eval

# end class FrobeniusNorm


operator_registry.register(MatMul)
operator_registry.register(Dot)
operator_registry.register(Outer)
operator_registry.register(Trace)
operator_registry.register(Transpose)
operator_registry.register(Det)
operator_registry.register(Inverse)
operator_registry.register(Norm)
operator_registry.register(InftyNorm)
operator_registry.register(FrobeniusNorm)
