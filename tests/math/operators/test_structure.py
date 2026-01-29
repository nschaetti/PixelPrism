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
from pixelprism.math.operators.structure import (
    Concatenate,
    Getitem,
    Flatten,
    HStack,
    VStack,
    Squeeze,
    Unsqueeze,
)


def _const_matrix(name="getitem_matrix"):
    """Create a 3x4 float32 constant and its NumPy payload for slicing tests."""
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    expr = pm.const(name=name, data=values.copy(), dtype=pm.DType.FLOAT32)
    return expr, values
# end def _const_matrix


def _const_tensor3d(name="getitem_tensor3d"):
    """Create a 3x4x5 tensor to stress-test chained slice handling."""
    values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    expr = pm.const(name=name, data=values.copy(), dtype=pm.DType.FLOAT32)
    return expr, values
# end def _const_tensor3d


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


def _const_scalar(name="flatten_scalar"):
    """Create a scalar constant for flatten tests."""
    value = np.array(3.5, dtype=np.float32)
    expr = pm.const(name=name, data=value.copy(), dtype=pm.DType.FLOAT32)
    return expr, value
# end def _const_scalar


def _const_tensor_with_unit_dims(name="structure_unit_tensor"):
    """Create a tensor with interleaved unit dimensions."""
    values = np.arange(12, dtype=np.float32).reshape(1, 3, 1, 4)
    expr = pm.const(name=name, data=values.copy(), dtype=pm.DType.FLOAT32)
    return expr, values
# end def _const_tensor_with_unit_dims


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


def test_getitem_single_slice_two_dimensional_tensor():
    """Slicing a 2-D tensor on its first axis should preserve trailing axes."""
    expr, values = _const_matrix()
    indices = [_slice(1, 3)]
    operator = Getitem(indices=indices)

    assert operator.check_operands([expr])
    assert operator.check_parameters(indices)
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = values[1:3]
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
# end test_getitem_single_slice_two_dimensional_tensor


def test_getitem_two_slices_covering_both_2d_axes():
    """Stacking two slice expressions should mirror chained NumPy slicing."""
    expr, values = _const_matrix()
    indices = [_slice(0, 3, 2), _slice(1, 4)]
    operator = Getitem(indices=indices)

    assert operator.check_operands([expr])
    assert operator.check_parameters(indices)
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = values[0:3:2, 1:4]
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
# end test_getitem_two_slices_covering_both_2d_axes


def test_getitem_three_slices_on_3d_tensor():
    """Ensure infer_shape matches actual slice chaining on 3-D inputs."""
    expr, values = _const_tensor3d()
    indices = [_slice(0, 3, 2), _slice(1, 4, 2), _slice(0, 5, 3)]
    operator = Getitem(indices=indices)

    assert operator.check_operands([expr])
    assert operator.check_parameters(indices)
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = values[0:3:2, 1:4:2, 0:5:3]
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
# end test_getitem_three_slices_on_3d_tensor


def test_flatten_scalar_becomes_single_element_vector():
    """Flatten should lift a scalar into a length-1 vector."""
    expr, value = _const_scalar()
    operator = Flatten()

    assert operator.check_operands([expr])
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = np.ravel(value)
    np.testing.assert_allclose(tensor.value, expected)
    assert tensor.dtype == expr.dtype
    assert operator.infer_dtype([expr]) == expr.dtype
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
# end test_flatten_scalar_becomes_single_element_vector


def test_flatten_vector_is_noop():
    """Flattening a 1-D tensor should preserve values and shape."""
    expr, values = _const_vector()
    operator = Flatten()

    assert operator.check_operands([expr])

    tensor = operator.eval([expr])
    expected = values.flatten()
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
    assert operator.infer_dtype([expr]) == expr.dtype
# end test_flatten_vector_is_noop


def test_flatten_matrix_matches_numpy_ravel():
    """Flatten should follow NumPy row-major order for matrices."""
    expr, values = _const_matrix()
    operator = Flatten()

    tensor = operator.eval([expr])
    expected = values.flatten()
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
    assert operator.infer_dtype([expr]) == expr.dtype
# end test_flatten_matrix_matches_numpy_ravel


def test_flatten_three_dimensional_tensor():
    """Flattening higher-rank tensors should collapse to vector."""
    expr, values = _const_tensor3d()
    operator = Flatten()

    tensor = operator.eval([expr])
    expected = values.flatten()
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
    assert operator.infer_dtype([expr]) == expr.dtype
# end test_flatten_three_dimensional_tensor


def test_squeeze_removes_all_unit_axes_by_default():
    """Squeeze defaults to removing every dimension equal to one."""
    expr, values = _const_tensor_with_unit_dims()
    operator = Squeeze()

    assert operator.check_operands([expr])
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = np.squeeze(values)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
    assert operator.infer_dtype([expr]) == expr.dtype
# end test_squeeze_removes_all_unit_axes_by_default


def test_squeeze_with_explicit_axes_only_removes_selected_dims():
    """Squeeze with axes removes only the requested unit dimensions."""
    values = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
    expr = pm.const(name="squeeze_axes", data=values.copy(), dtype=pm.DType.FLOAT32)
    operator = Squeeze(axes=[1])

    assert operator.check_operands([expr])
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = np.squeeze(values, axis=1)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
# end test_squeeze_with_explicit_axes_only_removes_selected_dims


def test_unsqueeze_inserts_size_one_axes():
    """Unsqueeze should expand tensors at the requested positions."""
    expr, values = _const_matrix()
    operator = Unsqueeze(axes=[0, 2])

    assert operator.check_operands([expr])
    assert operator.check_shapes([expr])

    tensor = operator.eval([expr])
    expected = np.expand_dims(values, axis=0)
    expected = np.expand_dims(expected, axis=2)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
    assert operator.infer_dtype([expr]) == expr.dtype
# end test_unsqueeze_inserts_size_one_axes


def test_unsqueeze_supports_negative_axes():
    """Negative axes count from the end when inserting dimensions."""
    expr, values = _const_vector()
    operator = Unsqueeze(axes=[-1])

    tensor = operator.eval([expr])
    expected = np.expand_dims(values, axis=-1)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape([expr]).dims == tuple(expected.shape)
# end test_unsqueeze_supports_negative_axes

#
# Concatenate
#

def test_concatenate_axis0_matches_numpy_result():
    """Concatenate should follow numpy along the leading axis."""
    values_a = np.arange(6, dtype=np.float32).reshape(2, 3)
    values_b = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_a = pm.const("concat_a", data=values_a.copy(), dtype=pm.DType.FLOAT32)
    expr_b = pm.const("concat_b", data=values_b.copy(), dtype=pm.DType.FLOAT32)
    operator = Concatenate(axis=0)
    operands = [expr_a, expr_b]
    assert operator.check_operands(operands)
    assert operator.check_shapes(operands)
    tensor = operator.eval(operands)
    expected = np.concatenate([values_a, values_b], axis=0)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape(operands).dims == tuple(expected.shape)
    assert operator.infer_dtype(operands) == pm.DType.FLOAT32
# end test_concatenate_axis0_matches_numpy_result


def test_concatenate_negative_axis_three_inputs():
    """Concatenate should handle multiple operands and negative axes."""
    left = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    mid = np.arange(8, 16, dtype=np.float32).reshape(2, 2, 2)
    right = np.arange(16, 24, dtype=np.float32).reshape(2, 2, 2)
    exprs = [
        pm.const(f"concat_three_{idx}", data=values.copy(), dtype=pm.DType.FLOAT32)
        for idx, values in enumerate((left, mid, right))
    ]
    operator = Concatenate(axis=-1)
    assert operator.check_shapes(exprs)
    tensor = operator.eval(exprs)
    expected = np.concatenate([left, mid, right], axis=-1)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape(exprs).dims == tuple(expected.shape)
# end test_concatenate_negative_axis_three_inputs


def test_concatenate_axis_none_flattens_inputs():
    """Axis None should flatten every operand before concatenation."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(3, dtype=np.float32)
    expr_a = pm.const("concat_flat_a", data=a.copy(), dtype=pm.DType.FLOAT32)
    expr_b = pm.const("concat_flat_b", data=b.copy(), dtype=pm.DType.FLOAT32)
    operator = Concatenate(axis=None)
    operands = [expr_a, expr_b]
    tensor = operator.eval(operands)
    expected = np.concatenate([a.flatten(), b.flatten()], axis=0)
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.check_shapes(operands)
    assert operator.infer_shape(operands).dims == tuple(expected.shape)
# end test_concatenate_axis_none_flattens_inputs


def test_concatenate_detects_mismatched_shapes():
    """Concatenate should reject tensors that differ on non-concatenated axes."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(8, dtype=np.float32).reshape(2, 4)
    expr_a = pm.const("concat_err_a", data=a.copy(), dtype=pm.DType.FLOAT32)
    expr_b = pm.const("concat_err_b", data=b.copy(), dtype=pm.DType.FLOAT32)
    operator = Concatenate(axis=0)
    with pytest.raises(ValueError):
        operator.check_shapes([expr_a, expr_b])
    # end with
# end test_concatenate_detects_mismatched_shapes


#
# hstack
#


def test_hstack_concatenates_along_second_axis():
    """HStack should mirror numpy.hstack for rank >= 2 tensors."""
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_left = pm.const("hstack_left", data=left.copy(), dtype=pm.DType.FLOAT32)
    expr_right = pm.const("hstack_right", data=right.copy(), dtype=pm.DType.FLOAT32)
    operator = HStack()
    operands = [expr_left, expr_right]

    assert operator.check_shapes(operands)

    tensor = operator.eval(operands)
    expected = np.hstack([left, right])
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape(operands).dims == tuple(expected.shape)
    assert operator.infer_dtype(operands) == pm.DType.FLOAT32
# end test_hstack_concatenates_along_second_axis


def test_hstack_requires_second_axis():
    """HStack should reject operands that cannot expose axis 1."""
    vec_a = np.arange(3, dtype=np.float32)
    vec_b = np.arange(3, 6, dtype=np.float32)
    expr_a = pm.const("hstack_vec_a", data=vec_a.copy(), dtype=pm.DType.FLOAT32)
    expr_b = pm.const("hstack_vec_b", data=vec_b.copy(), dtype=pm.DType.FLOAT32)
    operator = HStack()

    with pytest.raises(ValueError):
        operator.check_shapes([expr_a, expr_b])
# end test_hstack_requires_second_axis


#
# vstack
#


def test_vstack_concatenates_along_first_axis():
    """VStack mirrors numpy.vstack on the leading axis."""
    upper = np.arange(6, dtype=np.float32).reshape(2, 3)
    lower = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    expr_upper = pm.const("vstack_upper", data=upper.copy(), dtype=pm.DType.FLOAT32)
    expr_lower = pm.const("vstack_lower", data=lower.copy(), dtype=pm.DType.FLOAT32)
    operator = VStack()
    operands = [expr_upper, expr_lower]

    assert operator.check_shapes(operands)

    tensor = operator.eval(operands)
    expected = np.vstack([upper, lower])
    np.testing.assert_allclose(tensor.value, expected)
    assert operator.infer_shape(operands).dims == tuple(expected.shape)
# end test_vstack_concatenates_along_first_axis


def test_vstack_detects_shape_mismatch():
    """VStack should raise when non-leading axes differ."""
    top = np.arange(6, dtype=np.float32).reshape(2, 3)
    bottom = np.arange(12, dtype=np.float32).reshape(2, 6)
    expr_top = pm.const("vstack_top", data=top.copy(), dtype=pm.DType.FLOAT32)
    expr_bottom = pm.const("vstack_bottom", data=bottom.copy(), dtype=pm.DType.FLOAT32)
    operator = VStack()

    with pytest.raises(ValueError):
        operator.check_shapes([expr_top, expr_bottom])
# end test_vstack_detects_shape_mismatch
