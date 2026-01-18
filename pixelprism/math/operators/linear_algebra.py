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

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor, einsum
from .base import Operands, Operand, operator_registry, Operator, ParametricOperator

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
]


class LinearAlgebraOperator(Operator, ABC):
    """
    Linear algebra operator.
    """

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

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

    def check_parameters(self, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        pass
    # end def check_shapes

# end class LinearAlgebraOperator


class LinearAlgebraParametricOperator(LinearAlgebraOperator, ParametricOperator, ABC):
    """Linear algebra parametric operator."""

    def contains(
            self,
            expr: "MathExpr",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric operators must implement contains(..).")
    # end def contains

# end class LinearAlgebraParametricOperator


class MatMul(LinearAlgebraOperator):
    """
    Matrix Multiplication operator.
    """

    NAME = "matmul"
    ARITY = 2

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

        # -------- Matrix @ Matrix (batched) --------
        if a.rank >= 2 and b.rank >= 2:
            if a.shape[-1] != b.shape[-2]:
                raise ValueError(
                    f"Matrix-matrix requires contraction dims to match, "
                    f"got {a.shape[-1]} and {b.shape[-2]}"
                )
            # end if

            if a.shape[:-2] != b.shape[:-2]:
                raise ValueError(
                    f"MatMul requires batch dimensions to match, "
                    f"got {a.shape[:-2]} and {b.shape[:-2]}"
                )
            # end if

            return True
        # end if

        # -------- Matrix @ Vector --------
        if a.rank >= 2 and b.rank == 1:
            if a.shape[-1] != b.shape[-1]:
                raise ValueError(
                    f"Matrix-vector requires contraction dims to match, "
                    f"got {a.shape[-1]} and {b.shape[-1]}"
                )
            # end if

            if a.shape[:-2] != b.shape[:-1]:
                raise ValueError(
                    f"MatMul requires batch dimensions to match, "
                    f"got {a.shape[:-2]} and {b.shape[:-1]}"
                )
            # end if

            return True
        # end if

        # -------- Vector @ Matrix --------
        if a.rank == 1 and b.rank >= 2:
            if a.shape[-1] != b.shape[-2]:
                raise ValueError(
                    f"Vector-matrix requires contraction dims to match, "
                    f"got {a.shape[-1]} and {b.shape[-2]}"
                )
            # end if

            if a.shape[:-1] != b.shape[:-2]:
                raise ValueError(
                    f"MatMul requires batch dimensions to match, "
                    f"got {a.shape[:-1]} and {b.shape[:-2]}"
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
            return Shape(batch + (dims_a[-2], dims_b[-1]))
        # end if

        # Matrix @ Vector
        if rank_a >= 2 and rank_b == 1:
            batch = dims_a[:-2]
            return Shape(batch + (dims_a[-2],))
        # end if

        # Vector @ Matrix
        if rank_a == 1 and rank_b >= 2:
            batch = dims_b[:-2]
            return Shape(batch + (dims_b[-1],))
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

    NAME = "dot"
    ARITY = 2

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, b = operands

        # Same rank
        if a.rank != b.rank:
            raise ValueError(f"Dot requires same rank operands, got {a.rank} and {b.rank}")
        # end if

        # Last dim equivalent
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(f"Dot requires last dim to match, got {a.shape[-1]} and {b.shape[-1]}")
        # end if

        # Equivalent batch dims
        if a.rank > 1 and a.shape[:-1] != b.shape[:-1]:
            raise ValueError(f"Dot requires batch dims to match, got {a.shape[:-1]} and {b.shape[:-1]}")
        # end if

        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        A, B = operands
        dims_a = A.shape.dims
        rank_a = len(dims_a)

        # Batch dim
        batch_dim = ()
        if rank_a > 1:
            batch_dim = dims_a[:-1]
        # end if

        return Shape(batch_dim)
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, b = operands
        if a.ndim == 1:
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

    NAME = "outer"
    ARITY = 2

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, b = operands

        # Same rank
        if a.rank != b.rank:
            raise ValueError(f"Dot requires same rank operands, got {a.rank} and {b.rank}")
        # end if

        # Equivalent batch dims
        if a.rank > 1 and a.shape[:-1] != b.shape[:-1]:
            raise ValueError(f"Dot requires batch dims to match, got {a.shape[:-1]} and {b.shape[:-1]}")
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
        batch_dim = ()
        if rank_a > 1:
            batch_dim = dims_a[:-1]
        # end if

        return Shape(batch_dim + (dims_a[-1], dims_b[-1]))
    # end def infer_shape

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, b = operands
        if a.ndim == 1:
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

    NAME = "trace"
    ARITY = 1

    @classmethod
    def check_shapes(cls, operands: Operands) -> bool:
        a, = operands

        # Same rank
        if a.rank < 2:
            raise ValueError(f"Trace requires rank >= 2, got {a.rank}")
        # end if

        # Square matrices
        if a.shape[-1] != a.shape[-2]:
            raise ValueError(f"Trace requires square matrix, got {a.shape[-1]}x{a.shape[-2]}")
        # end if

        return True
    # end def check_shapes

    @classmethod
    def infer_shape(cls, operands: Operands) -> Shape:
        A, = operands
        dims_a = A.shape.dims
        rank_a = len(dims_a)

        # Batch dim
        batch_dim = ()
        if rank_a > 2:
            batch_dim = dims_a[:-2]
        # end if

        return Shape(batch_dim)
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

    NAME = "transpose"
    ARITY = 1

    def __init__(self, axes: Optional[Union["MathExpr", List[int]]] = None, **kwargs: Any):
        """
        Transpose
        """
        super(Transpose, self).__init__()
        from ..math_expr import Variable
        if axes and isinstance(axes, list):
            from ..utils import random_const_name, const
            self._check_axes(axes)
            axes = const(random_const_name(f"{self.__class__.__name__.lower()}-axes-"), axes, dtype=DType.INT32)
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

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
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
        axes = self._axes.eval().tolist() if self._axes else None
        return a.shape.transpose(axes=axes)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        return True
    # end def check_shapes

# end class Transpose


class Det(LinearAlgebraOperator):
    """
    Compute the determinant of a square matrix.
    """

    NAME = "det"
    ARITY = 1

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        return a.eval().det()
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
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
        new_shape = a.shape.copy()
        new_shape = new_shape[:-2]
        return Shape(new_shape)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        a: 'MathExpr' = operands[0]
        return a.rank >= 2 and a.shape[-1] == a.shape[-2]
    # end def check_shapes

# end class Det


class Inverse(LinearAlgebraOperator):
    """
    Compute the inverse of a square matrix.
    """

    NAME = "inverse"
    ARITY = 1

    def infer_shape(self, operands: Operands) -> Shape:
        a: 'MathExpr' = operands[0]
        return a.shape.copy()
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        """
        Infer the dtype from the input dtype.
        """
        return DType.FLOAT32
    # end def infer_dtype

    def check_shapes(self, operands: Operands) -> bool:
        a: 'MathExpr' = operands[0]
        return a.rank >= 2 and a.shape[-1] == a.shape[-2]
    # end def check_shapes

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        a, = operands
        return a.eval().inverse()
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("Inverse does not support backward.")
    # end def backward

# end class Inverse


operator_registry.register(MatMul)
operator_registry.register(Dot)
operator_registry.register(Outer)
operator_registry.register(Trace)
operator_registry.register(Transpose)
operator_registry.register(Det)
operator_registry.register(Inverse)
