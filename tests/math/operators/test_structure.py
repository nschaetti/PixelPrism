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

import pixelprism.math as pm
from pixelprism.math.math_expr import SliceExpr
from pixelprism.math.operators.structure import Getitem


def _const_matrix(name="getitem_matrix"):
    """Create a 3x4 float32 constant and its NumPy payload for slicing tests."""
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    expr = pm.const(name=name, data=values.copy(), dtype=pm.DType.FLOAT32)
    return expr, values
# end def _const_matrix


def _slice(start, stop, step=1):
    """Helper that mirrors Python slicing semantics via SliceExpr."""
    return SliceExpr.create(start=start, stop=stop, step=step)
# end def _slice


def _const_vector(name="getitem_vector"):
    """Create a 1-D tensor for slice coverage tests."""
    values = np.arange(10, dtype=np.float32)
    expr = pm.const(name=name, data=values.copy(), dtype=pm.DType.FLOAT32)
    return expr, values
# end def _const_vector


def test_getitem_mixed_slice_and_index_evaluation():
    """
    Ensure Getitem matches NumPy semantics for slice + integer indexing.
    """
    expr, np_values = _const_matrix()
    indices = [_slice(0, 2, step=1), 1]
    operator = Getitem(indices=indices)

    assert operator.check_operands([expr]) is True
    assert operator.check_parameters(indices)
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = np_values[0:2, 1]
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_dtype([expr]) == expr.dtype
    assert operator.infer_shape([expr]).dims == tuple(np.asarray(expected).shape)
# end test_getitem_mixed_slice_and_index_evaluation


def test_getitem_detects_out_of_bounds_indices():
    """
    Getitem should flag indices falling outside the operand extent.
    """
    expr, _ = _const_matrix()
    first_indices = [-4]
    operator = Getitem(indices=first_indices)

    assert operator.check_operands([expr]) is True
    assert operator.check_parameters(first_indices)
    assert operator.check_shapes([expr]) is False

    operator = Getitem(indices=[3])
    assert operator.check_shapes([expr]) is False
# end test_getitem_detects_out_of_bounds_indices


POS_NEG_STARTS = (1, -3)
STOP_CASES = (4, -1, None)
STEP_CASES = (1, -1, None)


@pytest.mark.parametrize("start", POS_NEG_STARTS)
@pytest.mark.parametrize("stop", STOP_CASES)
@pytest.mark.parametrize("step", STEP_CASES)
def test_getitem_slice_combinations(start, stop, step):
    """
    Validate Getitem for every start/stop/step slice combination requested.
    """
    expr, values = _const_vector()
    slice_expr = SliceExpr.create(start=start, stop=stop, step=step)
    indices = [slice_expr]
    operator = Getitem(indices=indices)

    assert operator.check_operands([expr]) is True
    assert operator.check_parameters(indices)
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = values[slice(start, stop, step)]
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_dtype([expr]) == expr.dtype
    assert operator.infer_shape([expr]).dims == tuple(np.asarray(expected).shape)
# end test_getitem_slice_combinations
