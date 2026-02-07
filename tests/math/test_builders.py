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
from pixelprism.math.functional import builders as B


def test_build_tensor_vector_from_scalars():
    """Flat sequences without an explicit shape should produce 1-D tensors."""
    elements = [
        pm.const("builder_a", data=1.5, dtype=pm.DType.R),
        2,
        3.25,
    ]
    expr = B.build_tensor(elements)

    np.testing.assert_allclose(expr.eval().value, np.array([1.5, 2.0, 3.25], dtype=np.float64))
    assert expr.shape.dims == (3,)
    assert expr.dtype == pm.DType.R
# end def test_build_tensor_vector_from_scalars


def test_build_tensor_with_variables_and_expressions():
    """Variables and symbolic expressions should integrate seamlessly."""
    with pm.new_context():
        state = pm.var("builder_state", dtype=pm.DType.R, shape=())
        pm.set_value(state.name, 0.5)

        elements = [
            state,
            pm.const("builder_bias", data=-1, dtype=pm.DType.Z),
            state + 1.0,
            4.25,
        ]
        expr = B.build_tensor(elements)

        np.testing.assert_allclose(expr.eval().value, np.array([0.5, -1.0, 1.5, 4.25], dtype=np.float32))
        assert expr.shape.dims == (4,)
        assert expr.dtype == pm.DType.R
# end def test_build_tensor_with_variables_and_expressions


def test_build_tensor_respects_explicit_shape():
    """Providing a shape parameter should reshape the flat buffer."""
    elements = [
        pm.const("builder_m0", data=1, dtype=pm.DType.Z),
        2.0,
        3.0,
        pm.const("builder_m3", data=-4, dtype=pm.DType.R),
    ]
    expr = B.build_tensor(elements, shape=(2, 2))

    expected = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float64)
    np.testing.assert_allclose(expr.eval().value, expected)
    assert expr.shape.dims == (2, 2)
    assert expr.dtype == pm.DType.R
# end def test_build_tensor_respects_explicit_shape


def test_build_tensor_rejects_non_scalar_inputs():
    """Attempting to pack tensors with rank > 0 should raise an error."""
    vector = pm.const("builder_vector", data=[1.0, 2.0], dtype=pm.DType.R)
    with pytest.raises(TypeError):
        B.build_tensor([vector, 3.0])
# end def test_build_tensor_rejects_non_scalar_inputs
