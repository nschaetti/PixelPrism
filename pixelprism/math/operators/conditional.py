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
Conditional operators.
"""

from __future__ import annotations

import numpy as np

from ..dtype import DType, to_numpy, promote
from ..shape import Shape
from ..tensor import Tensor
from ..math_expr import MathNode
from .base import Operands, Operator, operator_registry

__all__ = [
    "Where",
    "IfOperator",
]


class Where(Operator):
    """
    Elementwise conditional operator mirroring ``np.where`` semantics.
    """

    NAME = "where"
    ARITY = 3

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != self.ARITY:
            raise ValueError(f"{self.NAME} expects exactly 3 operands, got {len(operands)}")
        # end if
        cond, x, y = operands
        if cond.dtype != DType.B:
            raise TypeError(f"{self.NAME} condition must have dtype BOOL, got {cond.dtype}")
        # end if
        if cond.shape != x.shape or cond.shape != y.shape:
            raise ValueError(f"{self.NAME} operands must share the same shape, "
                             f"got {cond.shape}, {x.shape}, {y.shape}")
        # end if
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        cond, _, _ = operands
        return cond.shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        _, x, y = operands
        return promote(x.dtype, y.dtype)
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        cond, x, y = operands
        cond_val = cond.eval().value
        x_val = x.eval().value
        y_val = y.eval().value
        result = np.where(cond_val, x_val, y_val)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self.infer_dtype(operands))))
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Where does not support backward yet.")
    # end def _backward

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: str | None = None
    ) -> bool:
        return False
    # end def contains

    def __str__(self) -> str:
        return f"{self.NAME}()"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class Where


operator_registry.register(Where)


class IfOperator(Operator):
    """
    Scalar conditional operator selecting between two branches.
    """

    NAME = "if"
    ARITY = 2

    def __init__(self, cond: MathNode):
        super().__init__(cond=cond)
        self._cond = cond
        if self._cond.dtype != DType.B or self._cond.rank != 0:
            raise ValueError("If condition must be a scalar boolean expression.")
        # end if
    # end def __init__

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: str | None = None
    ) -> bool:
        return self._cond.contains(expr, by_ref=by_ref, look_for=look_for)
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) != 2:
            raise ValueError("If expects exactly two branch operands.")
        then_expr, else_expr = operands
        if then_expr.shape != else_expr.shape:
            raise ValueError("If branches must share identical shapes.")
        if then_expr.dtype != else_expr.dtype:
            raise ValueError("If branches must share identical dtypes.")
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return self.check_operands(operands)
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return operands[0].shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        then_expr, else_expr = operands
        cond_value = bool(np.asarray(self._cond.eval().value).item())
        branch = then_expr if cond_value else else_expr
        return branch.eval()
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("If does not support backward.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}(cond={self._cond.name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class IfOperator


operator_registry.register(IfOperator)
