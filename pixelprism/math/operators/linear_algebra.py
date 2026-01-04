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
from abc import ABC
from typing import Sequence

import numpy as np

from ..dtype import DType
from ..shape import Shape
from .base import Operands, Operand, operator_registry, Operator

__all__ = [
    "MatMul",
    "Dot",
    "Outer",
]


class LinearAlgebraOperator(Operator, ABC):
    """
    Linear algebra operator.
    """

    @classmethod
    def infer_dtype(cls, operands: Operands) -> DType:
        """
        Promote operand dtypes.
        """
        a, b = operands
        return DType.promote(a.dtype, b.dtype)
    # end def infer_dtype

# end class MatMul


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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        a, b = values
        return a @ b
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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        a, b = values
        if a.ndim == 1:
            return np.einsum("i,i->", a, b)
        else:
            return np.einsum("...i,...i->...", a, b)
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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        a, b = values
        if a.ndim == 1:
            return np.einsum("i,j->ij", a, b)
        else:
            return np.einsum("...i,...j->...ij", a, b)
        # end if
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError(f"{self.NAME} does not support backward.")
    # end def _backward

# end class Outer


operator_registry.register(MatMul())
operator_registry.register(Dot())
operator_registry.register(Outer())

