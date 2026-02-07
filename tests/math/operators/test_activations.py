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

import numpy as np
import pytest

import pixelprism.math as pm
from pixelprism.math.dtype import to_numpy
from pixelprism.math.functional.activations import (
    relu,
    leaky_relu,
    sigmoid,
    softplus,
    gelu,
)
from pixelprism.math.operators.activations import ReLU


def _const(name, values, dtype):
    arr = np.asarray(values, dtype=to_numpy(dtype))
    return pm.const(name=name, data=arr.copy(), dtype=dtype), arr
# end def _const


def test_relu_preserves_shape_dtype():
    tensor, np_values = _const("relu_input", [-1.0, 2.0], pm.DType.R)
    expr = relu(tensor)
    expected = np.maximum(np_values, 0.0)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == tensor.shape.dims
    assert expr.dtype == tensor.dtype
# end test_relu_preserves_shape_dtype


def test_leaky_relu_alpha_options():
    tensor, np_values = _const("leaky_input", [-2.0, 1.0], pm.DType.R)
    expr_default = leaky_relu(tensor)
    expr_custom = leaky_relu(tensor, alpha=0.2)
    np.testing.assert_allclose(expr_default.eval().value, np.where(np_values >= 0, np_values, 0.01 * np_values))
    np.testing.assert_allclose(expr_custom.eval().value, np.where(np_values >= 0, np_values, 0.2 * np_values))
# end test_leaky_relu_alpha_options


def test_sigmoid_matches_numpy():
    tensor, np_values = _const("sigmoid_input", [-1.0, 0.0, 1.0], pm.DType.R)
    expr = sigmoid(tensor)
    expected = 1.0 / (1.0 + np.exp(-np_values))
    np.testing.assert_allclose(expr.eval().value, expected)
# end test_sigmoid_matches_numpy


def test_softplus_matches_numpy():
    tensor, np_values = _const("softplus_input", [-1.0, 0.5], pm.DType.R)
    expr = softplus(tensor)
    expected = np.log1p(np.exp(np_values))
    np.testing.assert_allclose(expr.eval().value, expected)
# end test_softplus_matches_numpy


def test_gelu_matches_reference():
    tensor, np_values = _const("gelu_input", [-1.0, 0.0, 1.0], pm.DType.R)
    expr = gelu(tensor)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    expected = 0.5 * np_values * (1.0 + np.tanh(sqrt_2_over_pi * (np_values + 0.044715 * np_values ** 3)))
    np.testing.assert_allclose(expr.eval().value, expected)
# end test_gelu_matches_reference


def test_activation_backward_not_implemented():
    op = ReLU()
    with pytest.raises(NotImplementedError):
        op.backward(None, None)
# end test_activation_backward_not_implemented
