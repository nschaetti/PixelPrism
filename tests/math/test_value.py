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
"""Tests for pixelprism.math.Value."""
import pytest

# Imports
from pixelprism.math import Shape, Value, DType


def test_value_get():
    """Ensure Value exposes get/set/copy semantics."""
    shape = Shape((2, 2))
    value = Value([[1, 2], [3, 4]], shape, dtype=DType.INT32)
    assert value.shape is shape
    assert value.dtype == DType.INT32
    assert value.get()[0][0] == 1
# end def test_value_get


def test_value_set():
    """Ensure Value can be set to a scalar."""
    shape = Shape((2, 2))
    value = Value([[1, 2], [3, 4]], shape, dtype=DType.INT32)
    value.set([[5, 6], [7, 8]])
    assert value.get()[0][0] == 5
    assert value.get()[1][1] == 8
    assert value.dtype == DType.INT32
# end def test_value_set


def test_value_copy():
    """Ensure Value can be copied."""
    shape = Shape((2, 2))
    value = Value([[1, 2], [3, 4]], shape, dtype=DType.INT32)
    clone = value.copy()
    assert clone.get() == value.get()
    assert clone is not value
# end def test_value_copy


def test_value_mutable():
    """Ensure Value is mutable."""
    shape = Shape((2, 2))
    value = Value([[1, 2], [3, 4]], shape, dtype=DType.INT32)
    value.set([[5, 6], [7, 8]])
    assert value.get()[0][0] == 5
# end def test_value_mutable


def test_value_immutable():
    """Ensure Value is immutable."""
    shape = Shape((2, 2))
    value = Value([[1, 2], [3, 4]], shape, mutable=False, dtype=DType.INT32)
    with pytest.raises(RuntimeError, match="Trying to modify an immutable Value."):
        value.set([[5, 6], [7, 8]])
    # end with
# end def test_value_immutable


def test_value_dim_mismatch():
    """Ensure Value raises an error when the shape does not match."""
    shape = Shape((1, 1))
    with pytest.raises(ValueError, match="Dimension mismatch at index 0: 2 vs 1."):
        Value([[1, 2], [3, 4]], shape, dtype=DType.INT32)
    # end with
# end def test_value_dim_mismatch

