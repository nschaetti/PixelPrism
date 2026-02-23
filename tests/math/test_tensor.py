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
from pixelprism.math import typing_utils
from pixelprism.math.dtype import DType, from_numpy
from pixelprism.math.shape import Shape
from pixelprism.math.tensor import Tensor, _convert_data_to_numpy_array, _numpy_dtype_to_dtype, einsum


def test_numpy_dtype_to_dtype_supported_and_invalid_types():
    """
    Verify `_numpy_dtype_to_dtype` handles supported and unsupported numpy dtypes.

    Returns
    -------
    None
    """
    cases = [
        (np.dtype(np.float32), DType.R),
        (np.dtype(np.float64), DType.R),
        (np.dtype(np.int32), DType.Z),
        (np.dtype(np.int64), DType.Z),
        (np.dtype(np.bool_), DType.B),
    ]
    for dtype, expected in cases:
        assert _numpy_dtype_to_dtype(dtype) is expected
    # end for
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
    assert tensor.dtype == DType.R
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
    assert tensor.dtype == DType.R

    cast_source = np.arange(3, dtype=np.float64)
    cast_tensor = Tensor.from_numpy(cast_source, dtype=DType.R)
    assert cast_tensor.value is not cast_source
    assert cast_tensor.dtype == DType.R
    np.testing.assert_array_equal(cast_tensor.value, cast_source.astype(np.float32))

    listed = Tensor.from_list([1, 2, 3], dtype=DType.Z)
    assert listed.dtype == DType.Z
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
    tensor = Tensor.zeros(shape, dtype=DType.R)
    np.testing.assert_array_equal(tensor.value, np.zeros((2, 2), dtype=np.float32))
    assert tensor.dtype == DType.R
    assert tensor.shape.dims == (2, 2)

    default_dtype_tensor = Tensor.zeros(shape)
    assert default_dtype_tensor.dtype == DType.R
# end test test_tensor_zeros_factory_respects_dtype


def test_tensor_set_converts_inputs_to_dtype():
    """
    Confirm Tensor.set enforces the current dtype regardless of input dtype.

    Returns
    -------
    None
    """
    tensor = Tensor(data=[1, 2, 3], dtype=DType.R)
    tensor.set(np.array([4, 5, 6], dtype=np.int32))
    assert tensor.dtype == DType.R
    assert tensor.value.dtype == np.float32
    np.testing.assert_array_equal(tensor.value, np.array([4, 5, 6], dtype=np.float32))

    tensor.set([7, 8, 9])
    assert tensor.dtype == DType.R
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
    tensor = Tensor(data=[1, 2], dtype=DType.R)
    frozen = Tensor(data=[3], dtype=DType.R, mutable=False)

    text = str(tensor)
    assert "tensor(" in text
    assert "dtype=DType.R" in text
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
    tensor = utils.tensor(data=base, dtype=DType.R, mutable=False)
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
    assert empty.dtype == DType.R

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
    ones_like = utils.ones_like(base, dtype=DType.R)
    np.testing.assert_array_equal(eye_like.value, np.eye(2))
    assert eye_like.dtype == DType.R
    np.testing.assert_array_equal(zeros_like.value, np.zeros_like(base, dtype=np.float64))
    assert zeros_like.dtype == DType.R
    np.testing.assert_array_equal(ones_like.value, np.ones_like(base, dtype=np.float32))
    assert ones_like.dtype == DType.R
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

    def numpy_binary_expected(op, other, reverse=False):
        other_arr = other.value if isinstance(other, Tensor) else np.asarray(other)
        if reverse:
            return op(other_arr, left.value)
        return op(left.value, other_arr)
    # end def numpy_binary_expected

    cases = [
        (left + right, numpy_binary_expected(np.add, right)),
        (left - 1.0, numpy_binary_expected(np.subtract, 1.0)),
        (2.0 - left, numpy_binary_expected(np.subtract, 2.0, reverse=True)),
        (left * array, numpy_binary_expected(np.multiply, array)),
        (array / left, numpy_binary_expected(np.divide, array, reverse=True)),
        (left ** 2, numpy_binary_expected(np.power, 2)),
        (left @ right, np.matmul(left.value, right.value)),
    ]
    for tensor_result, expected in cases:
        np.testing.assert_array_equal(tensor_result.value, expected)
        assert tensor_result.dims == expected.shape
        assert tensor_result.dtype == from_numpy(expected.dtype)
# end test test_tensor_elementwise_operators_match_numpy


def test_tensor_getitem_preserves_dtype_and_shape():
    """
    Verify Tensor.__getitem__ slices return Tensors with aligned dtype/shape metadata.

    Returns
    -------
    None
    """
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    tensor = Tensor(data=data)

    first_row = tensor[0]
    first_col = tensor[:, 0]
    sub_block = tensor[1:, 2:]

    np.testing.assert_array_equal(first_row.value, data[0])
    assert first_row.dtype == DType.Z
    assert first_row.dims == data[0].shape

    np.testing.assert_array_equal(first_col.value, data[:, 0])
    assert first_col.dtype == DType.Z
    assert first_col.dims == data[:, 0].shape

    np.testing.assert_array_equal(sub_block.value, data[1:, 2:])
    assert sub_block.dtype == DType.Z
    assert sub_block.dims == data[1:, 2:].shape
# end test test_tensor_getitem_preserves_dtype_and_shape


def test_tensor_math_methods_match_numpy():
    """
    Ensure dedicated Tensor math helpers mirror numpy for representative inputs.

    Returns
    -------
    None
    """
    general_data = np.array([-0.75, -0.25, 0.5, 2.0], dtype=np.float64)
    positive_data = np.array([0.5, 1.5, 2.5, 5.0], dtype=np.float64)
    bounded_data = np.array([-0.8, -0.2, 0.2, 0.8], dtype=np.float64)
    hyperbolic_domain = np.array([1.1, 1.5, 3.0], dtype=np.float64)
    degree_data = np.array([0.0, 30.0, 90.0, 180.0], dtype=np.float64)

    def assert_unary(data: np.ndarray, method: str, numpy_fn):
        tensor = Tensor(data=data.copy())
        result = getattr(tensor, method)()
        expected = numpy_fn(data)
        np.testing.assert_allclose(result.value, expected)
        assert result.dims == data.shape
        assert result.dtype == from_numpy(expected.dtype)
    # end def assert_unary

    unary_cases = [
        (general_data, "square", np.square),
        (positive_data, "sqrt", np.sqrt),
        (general_data, "cbrt", np.cbrt),
        (general_data, "reciprocal", np.reciprocal),
        (general_data, "exp", np.exp),
        (general_data, "exp2", np.exp2),
        (general_data, "expm1", np.expm1),
        (positive_data, "log", np.log),
        (positive_data, "log2", np.log2),
        (positive_data, "log10", np.log10),
        (positive_data, "log1p", np.log1p),
        (general_data, "sin", np.sin),
        (general_data, "cos", np.cos),
        (general_data, "tan", np.tan),
        (bounded_data, "arcsin", np.arcsin),
        (bounded_data, "arccos", np.arccos),
        (bounded_data, "arctan", np.arctan),
        (general_data, "sinh", np.sinh),
        (general_data, "cosh", np.cosh),
        (general_data, "tanh", np.tanh),
        (general_data, "arcsinh", np.arcsinh),
        (hyperbolic_domain, "arccosh", np.arccosh),
        (bounded_data, "arctanh", np.arctanh),
        (degree_data, "deg2rad", np.deg2rad),
        (general_data, "rad2deg", np.rad2deg),
        (general_data, "absolute", np.abs),
        (general_data, "sign", np.sign),
        (general_data, "floor", np.floor),
        (general_data, "ceil", np.ceil),
        (general_data, "trunc", np.trunc),
        (general_data, "rint", np.rint),
    ]

    for data, method, numpy_fn in unary_cases:
        assert_unary(data, method, numpy_fn)
    # end for

    tensor_positive = Tensor(data=positive_data.copy())
    np.testing.assert_allclose(tensor_positive.pow(3).value, np.power(positive_data, 3))
    np.testing.assert_allclose(tensor_positive.square().value, np.square(positive_data))

    rounded = tensor_positive.round(decimals=2)
    np.testing.assert_allclose(rounded.value, np.round(positive_data, 2))

    clipped = tensor_positive.clip(min_value=1.0, max_value=3.0)
    np.testing.assert_allclose(clipped.value, np.clip(positive_data, 1.0, 3.0))
    assert clipped.dtype == DType.R
# end test test_tensor_math_methods_match_numpy


def test_tensor_math_functions_exposed_in_module():
    """
    Confirm pixelprism.math exposes tensor math helpers and enforces Tensor inputs.

    Returns
    -------
    None
    """
    import pixelprism.math as ppmath

    data = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    tensor = Tensor(data=data.copy())

    np.testing.assert_allclose(ppmath.log(tensor).value, np.log(data))
    np.testing.assert_allclose(ppmath.pow(tensor, 2).value, np.power(data, 2))
    np.testing.assert_allclose(
        ppmath.clip(tensor, min_value=1.0).value,
        np.clip(data, 1.0, None)
    )

    with pytest.raises(TypeError):
        ppmath.log(data)
    # end with
# end test test_tensor_math_functions_exposed_in_module


def test_tensor_einsum_matches_numpy_and_out_parameter():
    """
    Ensure tensor.einsum mirrors numpy einsum and respects Tensor outputs.
    """
    left = Tensor(data=np.arange(6, dtype=np.float32).reshape(2, 3))
    right = Tensor(data=np.arange(12, dtype=np.float32).reshape(3, 4))
    result = einsum("ik,kj->ij", left, right)
    expected = np.einsum("ik,kj->ij", left.value, right.value)
    np.testing.assert_allclose(result.value, expected)
    assert result.dtype == DType.R

    out = Tensor(data=np.zeros((2, 4), dtype=np.float32))
    returned = einsum("ik,kj->ij", left, right, out=out)
    assert returned is out
    np.testing.assert_allclose(out.value, expected)

    scalar = einsum("ij->", Tensor(data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)))
    np.testing.assert_allclose(scalar.value, np.einsum("ij->", np.array([[1.0, 2.0], [3.0, 4.0]])))
    assert scalar.dtype == DType.R
# end test test_tensor_einsum_matches_numpy_and_out_parameter


def test_tensor_random_sampling_factories_cover_classic_distributions():
    """Validate Tensor random factories for shape, dtype, and value constraints.

    Returns
    -------
    None
    """
    normal = Tensor.normal((4, 3), loc=0.0, scale=1.0)
    uniform = Tensor.uniform((4, 3), low=-2.0, high=5.0)
    integers = Tensor.randint((4, 3), low=2, high=7)
    poisson = Tensor.poisson((4, 3), lam=3.0)
    bernoulli = Tensor.bernoulli((4, 3), p=0.25)

    assert normal.shape.dims == (4, 3)
    assert uniform.shape.dims == (4, 3)
    assert integers.shape.dims == (4, 3)
    assert poisson.shape.dims == (4, 3)
    assert bernoulli.shape.dims == (4, 3)

    assert normal.dtype == DType.R
    assert uniform.dtype == DType.R
    assert integers.dtype == DType.Z
    assert poisson.dtype == DType.Z
    assert bernoulli.dtype == DType.Z

    assert np.all(uniform.value >= -2.0)
    assert np.all(uniform.value < 5.0)
    assert np.all(integers.value >= 2)
    assert np.all(integers.value < 7)
    assert np.all(np.isin(bernoulli.value, [0, 1]))
# end test test_tensor_random_sampling_factories_cover_classic_distributions


def test_tensor_random_sampling_module_exports_and_validation():
    """Ensure math-level random factories are exported and validated.

    Returns
    -------
    None
    """
    import pixelprism.math as ppmath

    sampled = ppmath.bernoulli((2, 2), p=0.6)
    assert isinstance(sampled, Tensor)
    assert sampled.dtype == DType.Z
    assert sampled.shape.dims == (2, 2)

    with pytest.raises(ValueError):
        Tensor.bernoulli((2, 2), p=-0.1)
    # end with
    with pytest.raises(ValueError):
        Tensor.uniform((2, 2), low=1.0, high=1.0)
    # end with
    with pytest.raises(ValueError):
        Tensor.poisson((2, 2), lam=-1.0)
    # end with
# end test test_tensor_random_sampling_module_exports_and_validation
