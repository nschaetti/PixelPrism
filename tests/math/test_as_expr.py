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

from pixelprism.math import as_expr, DType, Constant
from pixelprism.math.math_base import MathBase, MathNode, MathLeaf
from pixelprism.math.tensor import Tensor


def test_as_expr_returns_math_expr_unchanged():
    """
    Ensure MathExpr inputs are returned without wrapping or copying.

    Returns
    -------
    None
    """
    tensor = Constant(name="identity", data=Tensor.from_numpy(np.array([1.0]), dtype=DType.R))
    result = as_expr(tensor)
    assert isinstance(result, MathBase)
    assert isinstance(result, MathNode)
    assert isinstance(result, MathLeaf)
    assert result.name == "identity"
    assert isinstance(result, Constant)
# end test test_as_expr_returns_math_expr_unchanged


def test_as_expr_converts_python_scalar_with_dtype_and_mutability():
    """
    Scalars should be converted into a 0-D tensor honoring dtype/mutable.

    Returns
    -------
    None
    """
    result = as_expr(3.5, dtype=DType.R)
    assert isinstance(result, Constant)
    assert result.dtype is DType.R
    assert result.shape.dims == ()
    np.testing.assert_array_equal(result.eval().value, np.array(3.5, dtype=np.float32))
# end test test_as_expr_converts_python_scalar_with_dtype_and_mutability


def test_as_expr_converts_numpy_scalar_with_dtype_override():
    """
    NumPy scalars should be treated as scalar inputs and cast to dtype.

    Returns
    -------
    None
    """
    result = as_expr(np.float32(2.25), dtype=DType.R)
    assert isinstance(result, Constant)
    assert result.dtype is DType.R
    assert result.shape.dims == ()
    np.testing.assert_array_equal(result.eval().value, np.array(2.25, dtype=np.float32))
# end test test_as_expr_converts_numpy_scalar_with_dtype_override


def test_as_expr_wraps_numpy_array_with_and_without_dtype_override():
    """
    NumPy arrays should be wrapped, with optional dtype override.

    Returns
    -------
    None
    """
    data = np.array([[1, 2], [3, 4]], dtype=np.int64)
    result = as_expr(data, dtype=DType.R)
    assert isinstance(result, Constant)
    assert result.dtype is DType.R
    assert result.shape.dims == (2, 2)
    np.testing.assert_array_equal(result.eval().value, np.array(data, dtype=np.float32))

    data = np.array([[1, 2], [3, 4]], dtype=np.int64)
    result = as_expr(data)
    assert isinstance(result, Constant)
    assert result.dtype is DType.Z
    assert result.shape.dims == (2, 2)
    np.testing.assert_array_equal(result.eval().value, np.array(data, dtype=np.int64))
# end test test_as_expr_wraps_numpy_array_with_and_without_dtype_override


def test_as_expr_converts_nested_list_with_dtype():
    """
    Nested lists should be converted via numpy.asarray using dtype.

    Returns
    -------
    None
    """
    data = [[1, 2], [3, 4]]
    result = as_expr(data, dtype=DType.Z)
    assert isinstance(result, Constant)
    assert result.dtype is DType.Z
    assert result.shape.dims == (2, 2)
    np.testing.assert_array_equal(result.eval().value, np.array(data, dtype=np.int32))
# end test test_as_expr_converts_nested_list_with_dtype


def test_as_expr_converts_nested_list_default_dtype_float():
    """
    Nested lists should default to R when dtype is None.

    Returns
    -------
    None
    """
    data = [[1, 2], [3, 4]]
    result = as_expr(data, dtype=None)
    assert isinstance(result, MathBase)
    assert result.dtype is DType.R
    assert result.shape.dims == (2, 2)
    np.testing.assert_array_equal(result.eval().value, np.array(data, dtype=np.float32))
# end test test_as_expr_converts_nested_list_default_dtype_float


def test_as_expr_rejects_unsupported_types():
    """
    Unsupported inputs should raise a TypeError with a clear message.

    Returns
    -------
    None
    """
    with pytest.raises(TypeError, match="Cannot convert"):
        as_expr({"unsupported": True})
    # end with
# end test test_as_expr_rejects_unsupported_types
