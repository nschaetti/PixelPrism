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

from pixelprism.math import utils
from pixelprism.math.dtype import DType
from pixelprism.math.shape import Shape


def test_tensor_wrapper_reuses_numpy_buffer_and_respects_mutability():
    """
    Ensure `utils.tensor` reuses the numpy buffer and applies mutability flags.

    Returns
    -------
    None
    """
    data = np.arange(4, dtype=np.float32).reshape(2, 2)
    tensor = utils.tensor(data=data, dtype=DType.FLOAT32, mutable=False)
    np.testing.assert_array_equal(data, tensor.value)
    assert tensor.mutable is False
    assert tensor.dtype == DType.FLOAT32
    assert tensor.shape.dims == (2, 2)

    data2 = np.arange(4, dtype=np.float64).reshape(2, 2)
    tensor2 = utils.tensor(data=data2, dtype=DType.FLOAT64, mutable=False)
    np.testing.assert_array_equal(data2, tensor2.value)
    assert tensor2.mutable is False
    assert tensor2.dtype == DType.FLOAT64
    assert tensor2.shape.dims == (2, 2)
# end test test_tensor_wrapper_reuses_numpy_buffer_and_respects_mutability


def test_scalar_vector_matrix_rank_and_dtype():
    """
    Validate scalar, vector, and matrix helper constructors set rank and dtype.

    Returns
    -------
    None
    """
    scalar = utils.scalar(3.5, dtype=np.float32)
    vector = utils.vector([1, 2, 3], dtype=np.float64)
    matrix = utils.matrix([[1, 0], [0, 1]], dtype=np.int32)

    assert scalar.rank == 0
    assert scalar.dtype == DType.FLOAT32
    assert vector.rank == 1
    assert vector.shape.dims == (3,)
    assert vector.dtype == DType.FLOAT64
    assert matrix.rank == 2
    assert matrix.shape.dims == (2, 2)
    assert matrix.dtype == DType.INT32
# end test test_scalar_vector_matrix_rank_and_dtype


def test_empty_accepts_shape_objects_and_dtype_override():
    """
    Check that `utils.empty` handles `Shape` inputs and dtype overrides.

    Returns
    -------
    None
    """
    shape = Shape((2, 3, 1))
    tensor = utils.empty(shape, dtype=np.float64)

    assert tensor.shape.dims == (2, 3, 1)
    assert tensor.dtype == DType.FLOAT64
    assert tensor.mutable is True
# end test test_empty_accepts_shape_objects_and_dtype_override


def test_zeros_ones_full_and_nan_initialization():
    """
    Verify fill helpers (zeros, ones, full, nan) produce expected values.

    Returns
    -------
    None
    """
    zeros = utils.zeros([2, 2], dtype=np.float32, mutable=False)
    ones = utils.ones(3, dtype=np.float64)
    full = utils.full((2,), value=7, dtype=np.int32)
    nan_tensor = utils.nan((2, 1))

    assert zeros.mutable is False
    np.testing.assert_array_equal(zeros.value, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(ones.value, np.ones(3, dtype=np.float64))
    np.testing.assert_array_equal(full.value, np.full((2,), 7, dtype=np.int32))
    assert np.isnan(nan_tensor.value).all()
# end test test_zeros_ones_full_and_nan_initialization


def test_identity_helpers_I_and_diag():
    """
    Confirm identity constructors `I` and `diag` create correct matrices.

    Returns
    -------
    None
    """
    identity = utils.I(3, dtype=np.float64)
    diag = utils.diag([1, 2, 3], dtype=np.float32)

    assert identity.mutable is False
    np.testing.assert_array_equal(identity.value, np.eye(3, dtype=np.float64))
    np.testing.assert_array_equal(diag.value, np.diag([1, 2, 3]).astype(np.float32))
# end test test_identity_helpers_I_and_diag


def test_eye_like_and_dtype_override():
    """
    Ensure `eye_like` honors reference shapes and optional dtype overrides.

    Returns
    -------
    None
    """
    base = utils.tensor(
        data=np.zeros((4, 4), dtype=np.float32),
        dtype=DType.FLOAT32
    )
    eye = utils.eye_like(base)
    eye64 = utils.eye_like(base, dtype=np.float64)

    np.testing.assert_array_equal(eye.value, np.eye(4, dtype=np.float32))
    assert eye.dtype == DType.FLOAT32
    np.testing.assert_array_equal(eye64.value, np.eye(4, dtype=np.float64))
    assert eye64.dtype == DType.FLOAT64
# end test test_eye_like_and_dtype_override


def test_zeros_like_and_ones_like_clone_shape_and_dtype():
    """
    Check `zeros_like` and `ones_like` mirror shapes and respect dtype arguments.

    Returns
    -------
    None
    """
    base = np.arange(6, dtype=np.int32).reshape(2, 3)
    zeros_like = utils.zeros_like(base, dtype=np.float64)
    ones_like = utils.ones_like(base, dtype=DType.INT32)

    np.testing.assert_array_equal(zeros_like.value, np.zeros_like(base, dtype=np.float64))
    assert zeros_like.dtype == DType.FLOAT64
    np.testing.assert_array_equal(ones_like.value, np.ones_like(base, dtype=np.int32))
    assert ones_like.dtype == DType.INT32
# end test test_zeros_like_and_ones_like_clone_shape_and_dtype
