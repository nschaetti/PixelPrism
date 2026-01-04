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
import numpy as np
import pytest

from pixelprism.math import utils, DType
from pixelprism.math.functional import reduction as R


def _vector_operand():
    return utils.vector(
        name="vec",
        value=[1.0, -2.0, 3.0, 0.5],
        dtype=DType.FLOAT32
    )
# end def _vector_operand


def _matrix_operand():
    return utils.matrix(
        name="mat",
        value=[[1.0, 2.5, -3.0], [4.0, 0.0, 1.5]],
        dtype=DType.FLOAT32
    )
# end def _matrix_operand


def _tensor_operand():
    return utils.tensor(
        name="ten",
        data=[[[1.0, -1.0], [2.0, 3.0]], [[-0.5, 0.5], [4.0, -2.0]]],
        dtype=DType.FLOAT32
    )
# end def _tensor_operand


OPERAND_FACTORIES = (
    ("vector", _vector_operand),
    ("matrix", _matrix_operand),
    ("tensor", _tensor_operand),
)


REDUCTION_CASES = (
    ("sum", R.sum, np.sum),
    ("mean", R.mean, np.mean),
    ("std", R.std, np.std),
)


@pytest.mark.parametrize("op_name, op_func, np_func", REDUCTION_CASES)
@pytest.mark.parametrize("operand_name, operand_factory", OPERAND_FACTORIES)
def test_reduction_scalar_result(op_name, op_func, np_func, operand_name, operand_factory):
    """
    Each reduction should collapse arbitrary tensor ranks to a scalar.
    """
    operand = operand_factory()
    expr = op_func(operand)

    operand_values = operand.eval()
    expected = np.array(np_func(operand_values), dtype=operand_values.dtype)

    np.testing.assert_allclose(expr.eval(), expected)
    assert expr.shape.dims == ()
    assert expr.dtype == operand.dtype
# end test_reduction_scalar_result
