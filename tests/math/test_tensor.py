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


def test_tensor_basic_metadata_and_eval():
    """
    Validate Tensor exposes metadata accessors and evaluation overrides.

    Returns
    -------
    None
    """
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    tensor = Tensor(name="weights", data=data)

    np.testing.assert_array_equal(tensor.value, data)
    np.testing.assert_array_equal(tensor.eval(), data)
    override = data + 1
    np.testing.assert_array_equal(tensor.eval(weights=override), override)

    assert tensor.dtype == DType.FLOAT64
    assert tensor.shape.dims == (2, 3)
    assert tensor.dims == (2, 3)
    assert tensor.ndim == 2
    assert tensor.rank == 2
    assert tensor.size == 6
    assert tensor.n_elements == 6
    assert tensor.mutable
    assert tensor.is_mutable
# end test test_tensor_basic_metadata_and_eval


def test_tensor_set_converts_inputs_to_dtype():
    """
    Confirm Tensor.set enforces current dtype regardless of input dtype.

    Returns
    -------
    None
    """
    tensor = Tensor(name="cast", data=[1, 2, 3], dtype=DType.FLOAT32)
    tensor.set(np.array([4, 5, 6], dtype=np.int32))
    assert tensor.dtype == DType.FLOAT32
    assert tensor.data.dtype == np.float32
    np.testing.assert_array_equal(tensor.value, np.array([4, 5, 6], dtype=np.float32))

    tensor.set([7, 8, 9])
    assert tensor.dtype == DType.FLOAT32
    assert tensor.data.dtype == np.float32
    np.testing.assert_array_equal(tensor.value, np.array([7, 8, 9], dtype=np.float32))
# end test test_tensor_set_converts_inputs_to_dtype


def test_tensor_set_immutable_raises():
    """
    Validate immutable tensors reject `set` operations.

    Returns
    -------
    None
    """
    tensor = Tensor(name="frozen", data=[1, 2, 3], mutable=False, dtype=DType.FLOAT32)
    with pytest.raises(RuntimeError):
        tensor.set([4, 5, 6])
    # end with
    np.testing.assert_array_equal(tensor.value, np.array([1, 2, 3], dtype=np.float32))
# end test test_tensor_set_immutable_raises


def test_tensor_copy_preserves_data_and_optional_mutability_override():
    """
    Check Tensor.copy duplicates data and honors explicit mutability flags.

    Returns
    -------
    None
    """
    tensor = Tensor(name="original", data=np.array([1, 2], dtype=np.float32), mutable=True)
    clone = tensor.copy(name="clone")
    frozen = tensor.copy(name="frozen", mutable=False)

    assert clone.name == "clone"
    assert frozen.name == "frozen"
    assert clone.mutable is True
    assert frozen.mutable is False
    assert clone is not tensor
    assert frozen is not tensor
    np.testing.assert_array_equal(clone.value, tensor.value)
    np.testing.assert_array_equal(frozen.value, tensor.value)
    assert clone.dtype is tensor.dtype
    assert clone.shape.dims == tensor.shape.dims
# end test test_tensor_copy_preserves_data_and_optional_mutability_override


def test_tensor_int_conversion_errors_depend_on_rank():
    """
    Ensure int() conversion only works for scalar tensors.

    Returns
    -------
    None
    """
    vector = Tensor(name="vector", data=[1, 2], dtype=DType.FLOAT32)
    with pytest.raises(TypeError):
        int(vector)
    # end with

    scalar = Tensor(name="scalar", data=np.array(5, dtype=np.float32))
    assert int(scalar) == 5
# end test test_tensor_int_conversion_errors_depend_on_rank


def test_tensor_float_conversion_errors_depend_on_rank():
    """
    Ensure float() conversion is limited to scalar tensors.

    Returns
    -------
    None
    """
    vector = Tensor(name="vector_float", data=[1.0, 2.0], dtype=DType.FLOAT32)
    with pytest.raises(TypeError):
        float(vector)
    # end with

    scalar = Tensor(name="scalar_float", data=np.array(3.5, dtype=np.float32))
    assert float(scalar) == pytest.approx(3.5)
# end test test_tensor_float_conversion_errors_depend_on_rank


def test_tensor_string_and_repr(monkeypatch):
    """
    Validate Tensor string representations for mutable and immutable tensors.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to align Shape.__str__ output with __repr__ to simplify assertions.

    Returns
    -------
    None
    """
    monkeypatch.setattr(Shape, "__str__", lambda self: self.__repr__())
    tensor = Tensor(name="pretty", data=[1, 2], dtype=DType.FLOAT32)
    frozen = Tensor(name="frozen", data=[3], dtype=DType.FLOAT32, mutable=False)

    text = str(tensor)
    assert text.startswith("tensor(pretty")
    assert "dtype=DType.FLOAT32" in text
    assert "shape=" in text

    frozen_text = str(frozen)
    assert frozen_text.startswith("ctensor(frozen")
    assert repr(frozen) == frozen_text
# end test test_tensor_string_and_repr
