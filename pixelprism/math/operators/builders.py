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
Tensor building operators.
"""

# Imports
from abc import ABC
from typing import Optional, Sequence, Union
import numpy as np

from ..dtype import DType
from ..math_expr import MathExpr
from ..shape import Shape
from ..tensor import Tensor
from .base import Operands, Operator, ParametricOperator, operator_registry


Element = Union[MathExpr, float, int]
Elements = Sequence[Element]


__all__ = [
    "Builder",
    "ParametricBuilder",
    "BuildTensor",
]


class Builder(Operator, ABC):
    """Tensor builder operator."""
    pass
# end class TensorBuilder


class ParametricBuilder(Builder, ParametricOperator, ABC):
    """Parametric tensor builder operator."""
    pass
# end def ParametricTensorBuilder


class BuildTensor(ParametricBuilder):
    """Tensor builder from list."""

    NAME = "build_tensor"
    ARITY = -1
    IS_VARIADIC = True

    @classmethod
    def check_arity(cls, operands: Operands) -> bool:
        """Allow a variable number of operands, requiring at least one."""
        return len(operands) >= 1
    # end def check_arity

    def __init__(
            self,
            input_shape: Optional[Shape] = None
    ):
        super(BuildTensor, self).__init__(
            shape=input_shape
        )
        self._input_shape = input_shape
    # end def __init__

    # region PROPERTY

    @property
    def input_shape(self) -> Optional[Shape]:
        return self._input_shape
    # end def shape

    @property
    def n_elements(self) -> Optional[int]:
        """Get the number of elements in the tensor."""
        if self._input_shape is None:
            return None
        # end if
        return self._input_shape.size
    # end def n_elements

    # endregion PROPERTY

    # region PUBLIC

    @classmethod
    def check_parameters(cls, input_shape: Shape) -> bool:
        return input_shape.n_dims >= 1
    # end def check_parameters

    def contains(
            self,
            expr: MathExpr,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return False
    # end def contains

    def check_operands(self, operands: Operands) -> bool:
        if len(operands) < 1:
            return False
        # end if
        if self.n_elements is not None and len(operands) != self.n_elements:
            return False
        # end if
        return True
    # end def check_operands

    def infer_dtype(self, operands: Operands) -> DType:
        dtype = operands[0].dtype
        for operand in operands[1:]:
            dtype = DType.promote(dtype, operand.dtype)
        # end for
        return dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        if self._input_shape is None:
            return Shape(dims=(len(operands),))
        # end if
        return self._input_shape.copy()
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        for operand in operands:
            if operand.rank != 0:
                return False
            # end if
        # end for
        if self.n_elements is not None and len(operands) != self.n_elements:
            return False
        # end if
        return True
    # end def check_shapes

    # endregion PUBLIC

    # region PRIVATE

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        dtype = self.infer_dtype(operands)
        numpy_dtype = dtype.to_numpy()
        values = [np.asarray(operand.eval().value, dtype=numpy_dtype) for operand in operands]
        data = np.stack(values, axis=0)
        if self._input_shape is not None:
            data = data.reshape(self._input_shape.dims)
        # end if
        return Tensor(data=np.asarray(data, dtype=numpy_dtype), dtype=dtype)
    # end def _eval

    def _backward(self, out_grad: MathExpr, node: MathExpr) -> Sequence[MathExpr]:
        raise NotImplementedError("BuildTensor does not support backward propagation.")
    # end def _backward

    # endregion PRIVATE

    # region OVERRIDE

    def __str__(self) -> str:
        return f"{self.NAME}(input_shape={self._input_shape})"
    # end def __str__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_shape={self._input_shape})"
    # end def __repr__

    # endregion OVERRIDE

# end class BuildTensor


operator_registry.register(BuildTensor)
