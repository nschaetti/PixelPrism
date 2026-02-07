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
Common neural network activation operators.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from ..tensor import Tensor
from ..dtype import DType, to_numpy
from .base import Operands, operator_registry
from .elementwise import UnaryElementwiseOperator

__all__ = [
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Softplus",
    "GELU",
]


class _UnaryActivation(UnaryElementwiseOperator):
    """Utilities shared across simple activations."""

    @staticmethod
    def _tensor_data(operand):
        tensor = operand.eval()
        dtype = tensor.dtype
        numpy_dtype = to_numpy(dtype)
        data = tensor.value.astype(numpy_dtype, copy=False)
        return tensor, data, numpy_dtype
    # end def _tensor_data

# end class _UnaryActivation


class ReLU(_UnaryActivation):
    """Rectified Linear activation."""

    NAME = "relu"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor, data, numpy_dtype = self._tensor_data(value)
        result = np.maximum(data, 0)
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=tensor.dtype)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("ReLU does not support backward.")
    # end def _backward

# end class ReLU


class LeakyReLU(_UnaryActivation):
    """Leaky ReLU activation."""

    NAME = "leaky_relu"

    def __init__(self, alpha: float = 0.01):
        super().__init__(alpha=alpha)
        self._alpha = float(alpha)
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor, data, numpy_dtype = self._tensor_data(value)
        result = np.where(data >= 0, data, self._alpha * data)
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=tensor.dtype)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("LeakyReLU does not support backward.")
    # end def _backward

# end class LeakyReLU


class Sigmoid(_UnaryActivation):
    """Logistic sigmoid activation."""

    NAME = "sigmoid"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor, data, numpy_dtype = self._tensor_data(value)
        result = 1.0 / (1.0 + np.exp(-data))
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=tensor.dtype)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Sigmoid does not support backward.")
    # end def _backward

# end class Sigmoid


class Softplus(_UnaryActivation):
    """Softplus activation."""

    NAME = "softplus"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor, data, numpy_dtype = self._tensor_data(value)
        result = np.log1p(np.exp(data))
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=tensor.dtype)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Softplus does not support backward.")
    # end def _backward

# end class Softplus


class GELU(_UnaryActivation):
    """Gaussian Error Linear Unit using tanh approximation."""

    NAME = "gelu"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor, data, numpy_dtype = self._tensor_data(value)
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cubic = data ** 3
        inner = sqrt_2_over_pi * (data + 0.044715 * cubic)
        result = 0.5 * data * (1.0 + np.tanh(inner))
        return Tensor(data=np.asarray(result, dtype=numpy_dtype), dtype=tensor.dtype)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("GELU does not support backward.")
    # end def _backward

# end class GELU


operator_registry.register(ReLU)
operator_registry.register(LeakyReLU)
operator_registry.register(Sigmoid)
operator_registry.register(Softplus)
operator_registry.register(GELU)
