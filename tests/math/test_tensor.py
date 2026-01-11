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

from pixelprism.math import utils
from pixelprism.math.dtype import DType
from pixelprism.math.shape import Shape
from pixelprism.math.tensor import Tensor, _convert_data_to_numpy_array, _numpy_dtype_to_dtype


def test_numpy_dtype_to_dtype_supported_and_invalid_types():
    """
    Verify `_numpy_dtype_to_dtype` handles supported and unsupported numpy dtypes.

    Returns
    -------
    None
    """
    cases = [
        (np.dtype(np.float32), DType.FLOAT32),
        (np.dtype(np.float64), DType.FLOAT64),
        (np.dtype(np.int32), DType.INT32),
        (np.dtype(np.int64), DType.INT64),
    ]
    for dtype, expected in cases:
        assert _numpy_dtype_to_dtype(dtype) is expected
    # end for

    with pytest.raises(ValueError):
        _numpy_dtype_to_dtype(np.dtype(np.bool_))
    # end with
# end test test_numpy_dtype_to_dtype_supported_and_invalid_types


def test_convert_data_returns_arrays_with_expected_dtype():
    """
    Ensure `_convert_data` keeps numpy arrays and converts lists to float32 arrays.

    Returns
    -------
    None
    """
    arr = np.array([1, 2, 3], dtype=np.float32)
    assert _convert_data_to_numpy_array(arr) is arr

    converted = _convert_data_to_numpy_array([4, 5, 6], dtype=np.float64)
    converted2 = _convert_data_to_numpy_array([4.0, 5.0, 6.0])
    assert isinstance(converted, np.ndarray)
    assert converted.dtype == np.float64
    assert converted2.dtype == np.float64
    np.testing.assert_array_equal(converted, np.array([4, 5, 6], dtype=np.float64))
    np.testing.assert_array_equal(converted2, np.array([4, 5, 6], dtype=np.float64))
# end test test_convert_data_returns_arrays_with_expected_dtype


def test_tensor_basic_metadata_from_numpy_data():
    """
    Validate Tensor exposes metadata for numpy-backed tensors.

    Returns
    -------
    None
    """
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    tensor = Tensor(data=data)

    np.testing.assert_array_equal(tensor.value, data)
    assert tensor.dtype == DType.FLOAT64
    assert tensor.shape.dims == (2, 3)
    assert tensor.dims == (2, 3)
    assert tensor.ndim == 2
    assert tensor.rank == 2
    assert tensor.size == 6
    assert tensor.n_elements == 6
    assert tensor.mutable is True
    assert tensor.is_mutable is True
# end test test_tensor_basic_metadata_from_numpy_data


def test_tensor_from_numpy_and_from_list_factories():
    """
    Ensure Tensor factory helpers preserve or convert buffers as expected.

    Returns
    -------
    None
    """
    arr = np.arange(4, dtype=np.float32)
    tensor = Tensor.from_numpy(arr)
    assert tensor.value is arr
    assert tensor.dtype == DType.FLOAT32

    cast_source = np.arange(3, dtype=np.float64)
    cast_tensor = Tensor.from_numpy(cast_source, dtype=DType.FLOAT32)
    assert cast_tensor.value is not cast_source
    assert cast_tensor.dtype == DType.FLOAT32
    np.testing.assert_array_equal(cast_tensor.value, cast_source.astype(np.float32))

    listed = Tensor.from_list([1, 2, 3], dtype=DType.INT32)
    assert listed.dtype == DType.INT32
    assert listed.shape.dims == (3,)
# end test test_tensor_from_numpy_and_from_list_factories


def test_tensor_constructor_requires_dtype_for_python_lists():
    """
    Creating tensors from plain Python lists without dtype should error.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        Tensor(data=[1, 2, 3])
    # end with
# end test test_tensor_constructor_requires_dtype_for_python_lists


def test_tensor_zeros_factory_respects_dtype():
    """
    Validate Tensor.zeros honors provided dtype and Shape.

    Returns
    -------
    None
    """
    shape = Shape((2, 2))
    tensor = Tensor.zeros(shape, dtype=DType.FLOAT32)
    np.testing.assert_array_equal(tensor.value, np.zeros((2, 2), dtype=np.float32))
    assert tensor.dtype == DType.FLOAT32
    assert tensor.shape.dims == (2, 2)

    default_dtype_tensor = Tensor.zeros(shape)
    assert default_dtype_tensor.dtype == DType.FLOAT64
# end test test_tensor_zeros_factory_respects_dtype


def test_tensor_set_converts_inputs_to_dtype():
    """
    Confirm Tensor.set enforces the current dtype regardless of input dtype.

    Returns
    -------
    None
    """
    tensor = Tensor(data=[1, 2, 3], dtype=DType.FLOAT32)
    tensor.set(np.array([4, 5, 6], dtype=np.int32))
    assert tensor.dtype == DType.FLOAT32
    assert tensor.value.dtype == np.float32
    np.testing.assert_array_equal(tensor.value, np.array([4, 5, 6], dtype=np.float32))

    tensor.set([7, 8, 9])
    assert tensor.dtype == DType.FLOAT32
    assert tensor.value.dtype == np.float32
    np.testing.assert_array_equal(tensor.value, np.array([7, 8, 9], dtype=np.float32))
# end test test_tensor_set_converts_inputs_to_dtype


def test_tensor_copy_preserves_data_and_optional_mutability_override():
    """
    Check Tensor.copy duplicates data and honors explicit mutability flags.

    Returns
    -------
    None
    """
    tensor = Tensor(data=np.array([1, 2], dtype=np.float32), mutable=False)
    clone = tensor.copy()
    writable = tensor.copy(mutable=True)

    assert clone.mutable is False
    assert writable.mutable is True
    assert clone is not tensor
    assert writable is not tensor
    assert clone.value is not tensor.value
    np.testing.assert_array_equal(clone.value, tensor.value)
    np.testing.assert_array_equal(writable.value, tensor.value)
    assert clone.dtype is tensor.dtype
    assert clone.shape.dims == tensor.shape.dims
# end test test_tensor_copy_preserves_data_and_optional_mutability_override


def test_tensor_mutable_flag_controls_properties():
    """
    Confirm mutable constructor flag toggles both mutable and is_mutable properties.

    Returns
    -------
    None
    """
    frozen = Tensor(data=np.array([1.0]), mutable=False)
    assert frozen.mutable is False
    assert frozen.is_mutable is False

    writable = Tensor(data=np.array([1.0]))
    assert writable.mutable is True
    assert writable.is_mutable is True
# end test test_tensor_mutable_flag_controls_properties


def test_tensor_int_conversion_errors_depend_on_rank():
    """
    Ensure int() conversion only works for scalar tensors.

    Returns
    -------
    None
    """
    vector = Tensor(data=np.array([1, 2], dtype=np.int32))
    with pytest.raises(TypeError):
        int(vector)
    # end with

    scalar = Tensor(data=np.array(5, dtype=np.int32))
    assert int(scalar) == 5
# end test test_tensor_int_conversion_errors_depend_on_rank


def test_tensor_float_conversion_errors_depend_on_rank():
    """
    Ensure float() conversion is limited to scalar tensors.

    Returns
    -------
    None
    """
    vector = Tensor(data=np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(TypeError):
        float(vector)
    # end with

    scalar = Tensor(data=np.array(3.5, dtype=np.float32))
    assert float(scalar) == pytest.approx(3.5)
# end test test_tensor_float_conversion_errors_depend_on_rank


def test_tensor_string_and_repr(monkeypatch):
    """
    Validate Tensor string representations include core metadata.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to align Shape.__str__ output with __repr__ to simplify assertions.

    Returns
    -------
    None
    """
    monkeypatch.setattr(Shape, "__str__", lambda self: self.__repr__())
    tensor = Tensor(data=[1, 2], dtype=DType.FLOAT32)
    frozen = Tensor(data=[3], dtype=DType.FLOAT32, mutable=False)

    text = str(tensor)
    assert "tensor(" in text
    assert "dtype=DType.FLOAT32" in text
    assert "shape=" in text
    assert "mutable=True" in text

    frozen_text = str(frozen)
    assert "mutable=False" in frozen_text
    assert repr(frozen) == frozen_text
# end test test_tensor_string_and_repr


def test_tensor_creation_helpers_cover_all_paths():
    """
    Exercise every tensor creation helper to ensure they yield correct data/mutability.

    Returns
    -------
    None
    """
    base = np.arange(4, dtype=np.float32).reshape(2, 2)
    tensor = utils.tensor(data=base, dtype=DType.FLOAT32, mutable=False)
    assert isinstance(tensor, Tensor)
    assert tensor.mutable is False
    assert tensor.value is base

    scalar = utils.scalar(3.5, dtype=np.float32, mutable=False)
    assert scalar.rank == 0
    assert scalar.mutable is False

    vector = utils.vector([1, 2, 3], dtype=np.float64)
    np.testing.assert_array_equal(vector.value, np.array([1, 2, 3], dtype=np.float64))

    matrix = utils.matrix([[1, 0], [0, 1]], dtype=np.int32)
    np.testing.assert_array_equal(matrix.value, np.identity(2, dtype=np.int32))

    empty = utils.empty((1, 3), dtype=np.float64)
    assert empty.shape.dims == (1, 3)
    assert empty.dtype == DType.FLOAT64

    zeros = utils.zeros((2, 2), dtype=np.float32)
    ones = utils.ones(3, dtype=np.float64)
    full = utils.full((2,), value=7, dtype=np.int32)
    nan_tensor = utils.nan((1, 2))
    np.testing.assert_array_equal(zeros.value, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(ones.value, np.ones(3, dtype=np.float64))
    np.testing.assert_array_equal(full.value, np.full((2,), 7, dtype=np.int32))
    assert np.isnan(nan_tensor.value).all()

    identity = utils.I(3, dtype=np.float64)
    diag = utils.diag([1, 2, 3], dtype=np.float32)
    np.testing.assert_array_equal(identity.value, np.eye(3, dtype=np.float64))
    np.testing.assert_array_equal(diag.value, np.diag([1, 2, 3]).astype(np.float32))

    eye_like = utils.eye_like(tensor)
    zeros_like = utils.zeros_like(base, dtype=np.float64)
    ones_like = utils.ones_like(base, dtype=DType.FLOAT32)
    np.testing.assert_array_equal(eye_like.value, np.eye(2))
    assert eye_like.dtype == DType.FLOAT32
    np.testing.assert_array_equal(zeros_like.value, np.zeros_like(base, dtype=np.float64))
    assert zeros_like.dtype == DType.FLOAT64
    np.testing.assert_array_equal(ones_like.value, np.ones_like(base, dtype=np.float32))
    assert ones_like.dtype == DType.FLOAT32
# end test test_tensor_creation_helpers_cover_all_paths


def test_tensor_elementwise_operators_match_numpy():
    """
    Ensure Tensor arithmetic mirrors numpy semantics for scalars and arrays.

    Returns
    -------
    None
    """
    left = Tensor(data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    right = Tensor(data=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    array = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    np.testing.assert_array_equal((left + right).value, left.value + right.value)
    np.testing.assert_array_equal((left - 1.0).value, left.value - 1.0)
    np.testing.assert_array_equal((2.0 - left).value, 2.0 - left.value)
    np.testing.assert_array_equal((left * array).value, left.value * array)
    np.testing.assert_array_equal((array / left).value, array / left.value)
    np.testing.assert_array_equal((left ** 2).value, left.value ** 2)

    matmul_result = left @ right
    np.testing.assert_array_equal(matmul_result.value, np.matmul(left.value, right.value))
# end test test_tensor_elementwise_operators_match_numpy
